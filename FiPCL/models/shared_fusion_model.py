"""FiPCL encoders, task-specific fusion, Fisher weights and prototype banks."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline import SEEncoder, load_pretrained_mri
from model.biomarker_encoder import ClinicalDataEncoder
from model.advanced_fusion import AdvancedMultimodalFusion
from .fipcl_components import (
    FisherInformationTracker, PrototypeBank, ModalityUncertaintyEstimator,
)

MODALITY_KEYS = ['clinic', 'blood', 'prot']
MRI_SHARING_GROUPS = {
    'shared': {'clinic': 'clinic', 'blood': 'clinic', 'prot': 'clinic'},
    'independent': {'clinic': 'clinic', 'blood': 'blood', 'prot': 'prot'},
    'clinic_blood': {'clinic': 'clinic', 'blood': 'clinic', 'prot': 'prot'},
    'blood_prot': {'clinic': 'clinic', 'blood': 'blood', 'prot': 'blood'},
    'clinic_prot': {'clinic': 'clinic', 'blood': 'blood', 'prot': 'clinic'},
}

# Chunk size for encoding single-modality MRI batches on the pseudo-pair path.
# <=0 disables chunking and restores the original single-shot encoding.
ENCODE_CHUNK = int(os.environ.get('FIPCL_ENCODE_CHUNK', '8'))


class IndependentFiPCLModel(nn.Module):
    """Three task branches with explicit encoder and task-specific prototype ownership."""

    def __init__(
        self,
        input_dims: dict,
        embed_dim: int = 256,
        num_classes: int = 3,
        num_prototypes_per_class: int = 4,
        dropout: float = 0.4,
        device: str = 'cuda',
        pretrained_mri: bool = False,
        share_mri_encoder: bool = False,
        blood_branch_dropout: float | None = None,
        blood_mri_restrict: bool = False,
        blood_share_mri_with: str | None = None,
        prot_branch_dropout: float | None = None,
        nacc_num_labels: int = 6,
        mri_sharing: str | None = None,
        tabular_sharing: str = 'independent',
        clinic_encoder_type: str | None = 'mlp',
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.nacc_num_labels = nacc_num_labels
        self.device = device
        self.modality_keys = MODALITY_KEYS
        if mri_sharing is None:
            mri_sharing = ('shared' if share_mri_encoder else
                           {'clinic': 'clinic_blood', 'prot': 'blood_prot'}.get(
                               blood_share_mri_with, 'independent'))
        if mri_sharing not in MRI_SHARING_GROUPS:
            raise ValueError(f'Unknown MRI sharing mode: {mri_sharing}')
        if tabular_sharing not in ('legacy', 'independent', 'shared'):
            raise ValueError(f'Unknown tabular sharing mode: {tabular_sharing}')
        if clinic_encoder_type not in (None, 'se', 'mlp'):
            raise ValueError(f'Unknown Clinic encoder: {clinic_encoder_type}')
        if clinic_encoder_type == 'mlp' and tabular_sharing != 'independent':
            raise ValueError('An explicit Clinic MLP requires independent tabular encoders')
        if clinic_encoder_type == 'se' and tabular_sharing == 'legacy':
            raise ValueError('Legacy tabular mode requires its original Clinic MLP')
        self.clinic_encoder_type = clinic_encoder_type or ('mlp' if tabular_sharing == 'legacy' else 'se')
        self.mri_sharing = mri_sharing
        self.tabular_sharing = tabular_sharing
        self.share_mri_encoder = (mri_sharing == 'shared')
        # Branch-specific dropout: Blood / Prot can be set higher for overfitting branches.
        blood_dropout = dropout if blood_branch_dropout is None else blood_branch_dropout
        prot_dropout = dropout if prot_branch_dropout is None else prot_branch_dropout
        self.blood_branch_dropout = blood_dropout
        self.prot_branch_dropout = prot_dropout
        self.blood_mri_restrict = blood_mri_restrict
        # Hybrid mode: under share_mri_encoder=False, let Blood reuse another
        # branch's MRI encoder so it inherits gradients from a bigger sample pool.
        if blood_share_mri_with is not None and blood_share_mri_with not in MODALITY_KEYS:
            raise ValueError(f"blood_share_mri_with must be one of {MODALITY_KEYS}")
        if blood_share_mri_with == 'blood':
            blood_share_mri_with = None
        self.blood_share_mri_with = blood_share_mri_with

        # ========== MRI Encoder(s) + Per-Branch Projectors ==========
        def _build_mri_encoder(restrict: bool = False):
            """Build an MRI encoder. If `restrict`, only unfreeze slice_attention
            (projection/classifier stay frozen) — used for low-sample branches
            like Blood (n_train ~ 160) to curb overfitting under indep-MRI mode.
            """
            unfreeze_keys = ['slice_attention'] if restrict else ['slice_attention', 'projection', 'classifier']
            if pretrained_mri:
                enc = load_pretrained_mri(device, freeze=True, num_classes=num_classes)
                for name, param in enc.named_parameters():
                    if any(x in name for x in unfreeze_keys):
                        param.requires_grad = True
                return enc
            from model.slice_attention_resnet import Slice3DResNet
            return Slice3DResNet(
                num_slices=39, input_size=182, num_classes=3,
                pretrained=False, dropout=0.0,
            ).to(device)

        if self.share_mri_encoder:
            self.mri_encoder = _build_mri_encoder()
        else:
            # One independent MRI encoder per modality branch.
            # Use a ModuleDict so per-branch params are nicely namespaced.
            encoders, owners = {}, {}
            for mk in MODALITY_KEYS:
                owner = MRI_SHARING_GROUPS[mri_sharing][mk]
                if owner not in owners:
                    owners[owner] = _build_mri_encoder(
                        restrict=(blood_mri_restrict and mk == 'blood'),
                    )
                encoders[mk] = owners[owner]
            self.mri_encoders = nn.ModuleDict(encoders)
        # Separate projectors per modality (small MLPs, modality-adaptive)
        per_branch_dropout = {'clinic': dropout, 'blood': blood_dropout, 'prot': prot_dropout}
        for mk in MODALITY_KEYS:
            mk_drop = per_branch_dropout[mk]
            setattr(self, f'mri_projector_{mk}', nn.Sequential(
                nn.Linear(512, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(mk_drop),
            ))

        # ========== Bio Encoders (separate per modality) ==========
        self.clinic_encoder = ClinicalDataEncoder(
            input_dim=input_dims.get('clinic', 100),
            embed_dim=embed_dim,
            dropout=0.0 if tabular_sharing == 'legacy' else dropout,
            num_classes=num_classes,
        )
        self.clinic_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blood_encoder = SEEncoder(
            input_dim=input_dims.get('blood', 100),
            embed_dim=embed_dim,
            dropout=blood_dropout,
        )
        self.blood_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(blood_dropout),
        )

        self.prot_encoder = SEEncoder(
            input_dim=input_dims.get('prot', 100),
            embed_dim=embed_dim,
            dropout=prot_dropout,
        )
        self.prot_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(prot_dropout),
        )

        # Controlled sharing experiment: identical SE architecture in both modes.
        # Input adapters stay separate because feature dimensions/semantics differ.
        if tabular_sharing != 'legacy':
            for mk in MODALITY_KEYS:
                if mk == 'clinic' and self.clinic_encoder_type == 'mlp':
                    continue
                setattr(self, f'{mk}_encoder', SEEncoder(
                    input_dims[mk], embed_dim, dropout=dropout))
            if tabular_sharing == 'shared':
                for mk in ('blood', 'prot'):
                    encoder = getattr(self, f'{mk}_encoder')
                    for part in ('se1', 'mlp', 'se2'):
                        setattr(encoder, part, getattr(self.clinic_encoder, part))

        # ========== 3 Independent Fusions + Classifiers ==========
        self.fusion_clinic = AdvancedMultimodalFusion(
            embed_dim=embed_dim, num_heads=4,
            dropout=dropout, num_classes=num_classes,
        )
        self.fusion_blood = AdvancedMultimodalFusion(
            embed_dim=embed_dim, num_heads=4,
            dropout=blood_dropout, num_classes=num_classes,
        )
        self.fusion_prot = AdvancedMultimodalFusion(
            embed_dim=embed_dim, num_heads=4,
            dropout=prot_dropout, num_classes=num_classes,
        )

        # ========== NACC multi-label classifier (shares clinic encoders) ==========
        self.nacc_classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, nacc_num_labels),
        )

        # NACC bio encoder (separate from clinic_encoder due to different feature dims)
        _nacc_input_dim = input_dims.get('nacc', input_dims.get('clinic', 100))
        self.nacc_bio_encoder = SEEncoder(
            input_dim=_nacc_input_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # ========== GLOBAL Prototype Banks (v8: no clinic prototypes) ==========
        # MRI prototypes: used when pairing with single blood/proteomics data
        if tabular_sharing == 'legacy':
            self.mri_prototypes = PrototypeBank(embed_dim, num_classes, num_prototypes_per_class)
        else:
            # Projectors are task-specific even when the backbone is shared.
            # Their coordinates cannot be pooled into a global prototype bank.
            self.mri_prototype_banks = nn.ModuleDict({
                mk: PrototypeBank(embed_dim, num_classes, num_prototypes_per_class)
                for mk in MODALITY_KEYS
            })
        # Blood prototypes: used when pairing with single MRI data
        self.blood_prototypes = PrototypeBank(embed_dim, num_classes, num_prototypes_per_class)
        # Proteomics prototypes: used when pairing with single MRI data
        self.prot_prototypes = PrototypeBank(embed_dim, num_classes, num_prototypes_per_class)

        # Fisher trackers per modality pair (plain Python objects)
        self.fisher_trackers = {
            mk: FisherInformationTracker(embed_dim, num_modalities=2, device=device)
            for mk in self.modality_keys
        }

        # Fisher modality weights (buffers so .to() moves them)
        for mk in self.modality_keys:
            self.register_buffer(f'fisher_weight_mri_{mk}', torch.tensor(0.5))
            self.register_buffer(f'fisher_weight_bio_{mk}', torch.tensor(0.5))

        # Uncertainty estimators per modality pair
        self.uncertainty_mri = nn.ModuleDict({
            mk: ModalityUncertaintyEstimator(embed_dim) for mk in self.modality_keys
        })
        self.uncertainty_bio = nn.ModuleDict({
            mk: ModalityUncertaintyEstimator(embed_dim) for mk in self.modality_keys
        })

    # ============================================================
    # Device management
    # ============================================================

    def to(self, *args, **kwargs):
        for mk in self.modality_keys:
            self.fisher_trackers[mk].device = str(torch.device(*args, **kwargs))
        return super().to(*args, **kwargs)

    # ============================================================
    # Modality detection
    # ============================================================

    def _detect_modality(self, batch):
        for mk in self.modality_keys:
            if mk in batch:
                return mk
        return None

    # ============================================================
    # Encoding (v8: shared MRI encoder)
    # ============================================================

    def _encode_mri(self, mri_slices, modality_key='clinic'):
        """Encode MRI using shared encoder (or per-branch encoder) + per-branch projector."""
        if self.share_mri_encoder:
            encoder = self.mri_encoder
        else:
            encoder = self.mri_encoders[modality_key]
        _, feat = encoder(mri_slices, return_intermediate=True, return_weights=False)
        mri_proj = getattr(self, f'mri_projector_{modality_key}')
        return mri_proj(feat)

    def _encode_bio(self, bio_data, modality_key, mask=None):
        if modality_key == 'clinic' and self.clinic_encoder_type == 'mlp':
            _, features, _ = self.clinic_encoder(bio_data, return_causal=False)
            return self.clinic_projector(features)
        if self.tabular_sharing != 'legacy':
            if mask is not None:
                bio_data = bio_data * mask
            features = getattr(self, f'{modality_key}_encoder')(bio_data)
            return getattr(self, f'{modality_key}_projector')(features)
        if modality_key == 'clinic':
            _, feat, _ = self.clinic_encoder(bio_data, return_causal=False)
            return self.clinic_projector(feat)
        elif modality_key == 'blood':
            feat = self.blood_encoder(bio_data)
            return self.blood_projector(feat)
        elif modality_key == 'prot':
            if mask is not None:
                bio_data = bio_data * mask
            feat = self.prot_encoder(bio_data)
            return self.prot_projector(feat)
        else:
            raise ValueError(f"Unknown modality: {modality_key}")

    def mri_prototype_bank(self, modality_key):
        if self.tabular_sharing == 'legacy':
            return self.mri_prototypes
        return self.mri_prototype_banks[modality_key]

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, batch, return_all=False, update_prototypes=False):
        modality_key = self._detect_modality(batch)
        if modality_key is None:
            raise ValueError("No known modality key found in batch")
        return self._forward_pair(batch, modality_key, return_all, update_prototypes)

    def forward_pseudo_pair(self, mri_emb, bio_emb, labels, modality_key):
        """Forward pre-computed pseudo-pair embeddings through shared fusion.

        Args:
            mri_emb: (B, D) MRI embeddings (real or prototype)
            bio_emb: (B, D) bio embeddings (real or prototype)
            labels: (B,) class labels
            modality_key: str modality key

        Returns:
            dict with fusion_logits, mri_logits, bio_logits, embeddings
        """
        fusion = getattr(self, f'fusion_{modality_key}')
        outputs = fusion(mri_emb, bio_emb, return_all=True)
        outputs['modality_key'] = modality_key
        outputs['mri_emb'] = mri_emb
        outputs['bio_emb'] = bio_emb
        return outputs

    def _forward_pair(self, batch, modality_key, return_all=False, update_prototypes=False):
        mri_slices = batch['mri_slices']
        bio_data = batch[modality_key]
        mask = batch.get('mask', None)
        labels = batch.get('label', None)

        mri_emb = self._encode_mri(mri_slices, modality_key)
        bio_emb = self._encode_bio(bio_data, modality_key, mask)

        fusion = getattr(self, f'fusion_{modality_key}')
        outputs = fusion(mri_emb, bio_emb, return_all=True)
        outputs['modality_key'] = modality_key
        outputs['mri_emb'] = mri_emb
        outputs['bio_emb'] = bio_emb

        if update_prototypes and labels is not None:
            with torch.no_grad():
                self.mri_prototype_bank(modality_key).update(mri_emb.detach(), labels)
                # Update bio prototypes only for blood and prot (not clinic)
                if modality_key == 'blood':
                    self.blood_prototypes.update(bio_emb.detach(), labels)
                elif modality_key == 'prot':
                    self.prot_prototypes.update(bio_emb.detach(), labels)

        if not return_all:
            return {'fusion_logits': outputs['fusion_logits']}
        return outputs

    def forward_nacc(self, batch):
        """Forward NACC batch through shared MRI encoder + NACC bio encoder + NACC head.

        Returns raw logits (num_labels,) for BCEWithLogitsLoss.
        """
        mri_slices = batch['mri_slices']
        bio_data = batch.get('clinic', batch.get('tabular'))
        mri_emb = self._encode_mri(mri_slices, 'clinic')
        # Use NACC-specific bio encoder (different feature dimensionality)
        bio_emb = self.nacc_bio_encoder(bio_data)
        combined = torch.cat([mri_emb, bio_emb], dim=1)
        logits = self.nacc_classifier(combined)
        return logits

    # ============================================================
    # Embeddings (for prototype update)
    # ============================================================

    @torch.no_grad()
    def get_embeddings(self, batch):
        modality_key = self._detect_modality(batch)
        mri_emb = self._encode_mri(batch['mri_slices'], modality_key)
        bio_emb = self._encode_bio(batch[modality_key], modality_key, batch.get('mask'))
        return mri_emb, bio_emb, modality_key

    # ============================================================
    # Fisher-related methods
    # ============================================================

    def compute_fisher_regularization(self, mri_emb, bio_emb, modality_key):
        traces = self.fisher_trackers[modality_key].get_trace()
        if traces[0] == 0 and traces[1] == 0:
            return torch.tensor(0.0, device=mri_emb.device)

        cos_sim = F.cosine_similarity(mri_emb, bio_emb, dim=1)
        align_loss = (1.0 - cos_sim).mean()

        total = traces[0] + traces[1] + 1e-8
        balance = (traces[0] * traces[1]) / (total * total + 1e-8) * 4.0
        balance = min(balance, 1.0)

        return 0.1 * (balance ** 0.5) * align_loss

    def compute_prototype_loss(self, mri_emb, bio_emb, labels, modality_key, temperature=0.07):
        """Prototype contrastive loss within the task's projector coordinates."""
        loss_mri = self.mri_prototype_bank(modality_key).compute_prototype_loss(
            mri_emb, labels, temperature=temperature)
        if modality_key == 'blood':
            loss_bio = self.blood_prototypes.compute_prototype_loss(bio_emb, labels, temperature=temperature)
        elif modality_key == 'prot':
            loss_bio = self.prot_prototypes.compute_prototype_loss(bio_emb, labels, temperature=temperature)
        else:
            # clinic: no bio prototypes
            loss_bio = torch.tensor(0.0, device=mri_emb.device)
        return loss_mri + loss_bio

    def update_fisher_weights(self, modality_key):
        weights = self.fisher_trackers[modality_key].get_modality_weights(temperature=0.5)
        getattr(self, f'fisher_weight_mri_{modality_key}').fill_(weights[0].item())
        getattr(self, f'fisher_weight_bio_{modality_key}').fill_(weights[1].item())

    def compute_fisher_guided_fusion_loss(self, mri_logits, bio_logits, labels, modality_key, class_weights=None):
        w_mri = getattr(self, f'fisher_weight_mri_{modality_key}').item()
        w_bio = getattr(self, f'fisher_weight_bio_{modality_key}').item()
        if labels.dim() == 2 and labels.shape[1] > 1:
            # Multi-label: use BCE
            loss_mri = F.binary_cross_entropy_with_logits(mri_logits, labels.float())
            loss_bio = F.binary_cross_entropy_with_logits(bio_logits, labels.float())
        else:
            ce_kwargs = {"weight": class_weights} if class_weights is not None else {}
            loss_mri = F.cross_entropy(mri_logits, labels, **ce_kwargs)
            loss_bio = F.cross_entropy(bio_logits, labels, **ce_kwargs)
        return w_mri * loss_mri + w_bio * loss_bio

    # ============================================================
    # Prototype Orthogonality Loss (v8)
    # ============================================================

    def compute_orthogonality_loss(self):
        """L_ortho = ||P P^T - I||_F for each prototype bank.

        Pushes prototypes apart within each class for better diversity.
        """
        mri_banks = ([self.mri_prototypes] if self.tabular_sharing == 'legacy'
                     else list(self.mri_prototype_banks.values()))
        banks = mri_banks + [self.blood_prototypes, self.prot_prototypes]
        loss = torch.tensor(0.0, device=banks[0].prototypes.device)
        for bank in banks:
            for c in range(self.num_classes):
                P = bank.prototypes[c]  # (K, D) where K = num_protos
                P_norm = F.normalize(P, dim=1)
                gram = torch.mm(P_norm, P_norm.T)  # (K, K)
                target = torch.eye(P.size(0), device=gram.device)
                loss += torch.norm(gram - target, p='fro')
        return loss / (self.num_classes * len(banks))

    # ============================================================
    # Pseudo-pair loss (v8: bidirectional, no clinic)
    # ============================================================

    def compute_pseudo_pair_loss_bio_to_mri(self, single_batch, epoch, warmup_epochs=3):
        """Single bio data → nearest MRI prototype → MRI-bio fusion.

        Used for: single blood → MRI, single proteomics → MRI.
        NOT used for clinic (clinic has no pseudo-pairing).
        """
        device = next(self.parameters()).device
        labels = single_batch['label'].to(device)
        B = labels.size(0)

        mk = self._detect_modality(single_batch)
        if mk is None or mk == 'clinic':
            return torch.tensor(0.0, device=device)

        bio_data = single_batch[mk].to(device)
        mask = single_batch.get('mask')
        if mask is not None:
            mask = mask.to(device)
        bio_emb = self._encode_bio(bio_data, mk, mask)

        # Find nearest same-class MRI prototype for each sample
        mri_protos_list = []
        for i in range(B):
            c = labels[i].item()
            all_c_protos = self.mri_prototype_bank(mk).get_all_prototypes(c)  # (K, D)
            sims = F.cosine_similarity(bio_emb[i:i+1], all_c_protos, dim=1)
            mri_protos_list.append(all_c_protos[sims.argmax()])
        mri_protos = torch.stack(mri_protos_list).to(bio_emb.device)

        # Confidence: cosine similarity between bio and matched MRI prototype
        conf = F.cosine_similarity(bio_emb, mri_protos, dim=1)
        conf = torch.sigmoid(conf / 0.5).clamp(0.1, 1.0)

        # Forward through fusion
        pout = self.forward_pseudo_pair(mri_protos, bio_emb, labels, mk)

        # Weighted loss
        loss = (F.cross_entropy(pout['fusion_logits'], labels, reduction='none') * conf).mean()
        loss += 0.5 * (F.cross_entropy(pout['mri_logits'], labels, reduction='none') * conf).mean()
        loss += 0.5 * (F.cross_entropy(pout['bio_logits'], labels, reduction='none') * conf).mean()

        # MSE alignment loss
        mse = (bio_emb - mri_protos).pow(2).mean(dim=1)
        loss += 0.1 * (mse * conf).mean()

        return loss

    def compute_pseudo_pair_loss_mri_to_bio(self, mri_batch, target_modality, epoch, warmup_epochs=3):
        """Single MRI data → nearest Blood/Proteomics prototype → MRI-bio fusion.

        Args:
            mri_batch: dict with 'mri_slices' and 'label' (single MRI data)
            target_modality: 'blood' or 'prot'
            epoch: current epoch
            warmup_epochs: warmup epochs
        """
        device = next(self.parameters()).device
        labels = mri_batch['label'].to(device)
        B = labels.size(0)

        mri_slices = mri_batch['mri_slices'].to(device)
        # The slice-attention encoder keeps every slice activation (~0.55 GB per volume
        # in backward), so a 32-sample pseudo batch needs >17 GB if encoded in one shot.
        # Chunk it and recompute activations in backward (checkpointing) to keep the peak
        # close to the paired batch size.
        # NOTE: chunking changes the BatchNorm statistics of the MRI encoder within a
        # pseudo batch, which perturbs training (RNG consumption and BN moments) and
        # therefore the final metrics. Set FIPCL_ENCODE_CHUNK=0 to restore the original
        # single-shot encoding bit-for-bit when memory allows.
        chunk = B if ENCODE_CHUNK <= 0 else min(ENCODE_CHUNK, B)
        if chunk >= B:
            mri_emb = self._encode_mri(mri_slices, target_modality)
        else:
            parts = [
                torch.utils.checkpoint.checkpoint(
                    self._encode_mri, mri_slices[i:i + chunk], target_modality,
                    use_reentrant=False,
                )
                for i in range(0, B, chunk)
            ]
            mri_emb = torch.cat(parts, dim=0)

        # Select target prototype bank
        if target_modality == 'blood':
            target_bank = self.blood_prototypes
        elif target_modality == 'prot':
            target_bank = self.prot_prototypes
        else:
            return torch.tensor(0.0, device=mri_emb.device)

        # Find nearest same-class bio prototype for each MRI sample
        bio_protos_list = []
        for i in range(B):
            c = labels[i].item()
            all_c_protos = target_bank.get_all_prototypes(c)  # (K, D)
            sims = F.cosine_similarity(mri_emb[i:i+1], all_c_protos, dim=1)
            bio_protos_list.append(all_c_protos[sims.argmax()])
        bio_protos = torch.stack(bio_protos_list).to(mri_emb.device)

        # Confidence
        conf = F.cosine_similarity(mri_emb, bio_protos, dim=1)
        conf = torch.sigmoid(conf / 0.5).clamp(0.1, 1.0)

        # Forward through fusion (MRI as MRI, prototype as bio)
        pout = self.forward_pseudo_pair(mri_emb, bio_protos, labels, target_modality)

        # Weighted loss
        loss = (F.cross_entropy(pout['fusion_logits'], labels, reduction='none') * conf).mean()
        loss += 0.5 * (F.cross_entropy(pout['mri_logits'], labels, reduction='none') * conf).mean()
        loss += 0.5 * (F.cross_entropy(pout['bio_logits'], labels, reduction='none') * conf).mean()

        # MSE alignment
        mse = (mri_emb - bio_protos).pow(2).mean(dim=1)
        loss += 0.1 * (mse * conf).mean()

        return loss


# ============================================================
# Factory
# ============================================================

def create_independent_fipcl_model(input_dims, embed_dim=256, num_classes=3,
                                    dropout=0.4, device='cuda', pretrained_mri=False,
                                    share_mri_encoder=False,
                                    blood_branch_dropout=None,
                                    blood_mri_restrict=False,
                                    blood_share_mri_with=None,
                                    prot_branch_dropout=None,
                                    nacc_num_labels=6, mri_sharing=None,
                                    tabular_sharing='independent', clinic_encoder_type='mlp'):
    return IndependentFiPCLModel(
        input_dims=input_dims, embed_dim=embed_dim, num_classes=num_classes,
        num_prototypes_per_class=4, dropout=dropout,
        device=device, pretrained_mri=pretrained_mri,
        share_mri_encoder=share_mri_encoder,
        blood_branch_dropout=blood_branch_dropout,
        blood_mri_restrict=blood_mri_restrict,
        blood_share_mri_with=blood_share_mri_with,
        prot_branch_dropout=prot_branch_dropout,
        nacc_num_labels=nacc_num_labels,
        mri_sharing=mri_sharing, tabular_sharing=tabular_sharing,
        clinic_encoder_type=clinic_encoder_type,
    )
