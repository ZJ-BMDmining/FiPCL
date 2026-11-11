"""Fisher reliability traces, class prototype banks and uncertainty estimation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# Fisher Information Matrix
# ============================================================

class FisherInformationTracker:
    """Track empirical Fisher Information via gradient trace per modality.

    Stores scalar Fisher traces with EMA smoothing to prevent wild fluctuations.
    Uses clamped softmax weights to prevent one modality from completely dominating.

    Used for:
    - Modality importance weighting: w_m = softmax(tr_m / temperature)
    - Fisher stability monitoring for convergence
    """

    def __init__(self, embed_dim=256, num_modalities=3, device='cuda',
                 ema_alpha=0.99, weight_clip=0.9):
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.device = device
        self.ema_alpha = ema_alpha
        self.weight_clip = weight_clip
        # Initialize EMA state BEFORE reset() so it persists across resets
        self._ema_traces = None
        self.reset()

    def reset(self):
        """Reset batch accumulator but preserve EMA state for cross-epoch smoothing."""
        self.fim_traces = [0.0 for _ in range(self.num_modalities)]
        self.count = 0
        self._ema_dirty = False

    def accumulate(self, trace_values):
        """Accumulate scalar Fisher trace values with EMA smoothing.

        Args:
            trace_values: list of scalar floats, one per modality
        """
        for i, tv in enumerate(trace_values):
            self.fim_traces[i] += tv
        self.count += 1
        self._ema_dirty = True

    def _get_avg_traces(self):
        """Get averaged traces with log-scaling to reduce variance."""
        if self.count == 0:
            return self.fim_traces
        avg = [t / self.count for t in self.fim_traces]
        # Apply log-scaling to stabilize extreme values
        log_scaled = [np.sign(t) * np.log1p(abs(t)) for t in avg]
        return log_scaled

    def get_trace(self, modality_idx=None):
        """Return averaged Fisher trace with EMA smoothing.

        EMA is only updated when count > 0 (real Fisher data available).
        When called during warmup (count=0), returns zeros without
        polluting the EMA state.
        """
        raw = self._get_avg_traces()
        if self.count > 0 and self._ema_dirty:
            if self._ema_traces is None:
                self._ema_traces = raw[:]
            else:
                for i in range(len(raw)):
                    self._ema_traces[i] = (self.ema_alpha * self._ema_traces[i]
                                           + (1 - self.ema_alpha) * raw[i])
            self._ema_dirty = False
        # When count==0 and no EMA state yet, return zeros
        if self._ema_traces is None:
            return [0.0 for _ in range(self.num_modalities)]
        if modality_idx is not None:
            return self._ema_traces[modality_idx]
        return self._ema_traces

    def peek_trace(self, modality_idx=None):
        """Return current trace WITHOUT updating EMA (for weight computation)."""
        if self._ema_traces is None:
            return [0.0 for _ in range(self.num_modalities)]
        if modality_idx is not None:
            return self._ema_traces[modality_idx]
        return self._ema_traces

    def get_modality_weights(self, temperature=0.5):
        """Compute modality weights with clipping to prevent dominance.

        w_m = clip(softmax(tr(F_m) / temperature), weight_clip)
        """
        traces = torch.tensor(self.peek_trace(), device=self.device)
        traces = torch.clamp(traces, min=1e-8)
        weights = F.softmax(traces / temperature, dim=0)
        # Clip weights to [1-weight_clip, weight_clip] to prevent dominance
        weights = torch.clamp(weights, 1 - self.weight_clip, self.weight_clip)
        weights = weights / (weights.sum() + 1e-8)
        return weights

    def get_top_eigen(self, modality_idx, k=4):
        """Return identity-based pseudo eigenvalues (for compatibility).

        Since we only track traces, return the trace as the top eigenvalue.
        """
        tr = self.get_trace(modality_idx)
        eigval = torch.tensor([tr], device=self.device)
        eigvec = torch.eye(1, device=self.device)
        return eigval, eigvec

    def compute_fisher_alignment(self, mod_a, mod_b, k=4):
        """Fisher alignment from trace balance.

        Simplified: balance = min(tr_a/(tr_a+tr_b), tr_b/(tr_a+tr_b))
        """
        tr_a = self.get_trace(mod_a)
        tr_b = self.get_trace(mod_b)
        total = tr_a + tr_b + 1e-8
        return min(tr_a / total, tr_b / total)

    def fisher_stability(self, prev_traces):
        """Compute Fisher stability: relative change in traces."""
        if prev_traces is None:
            return 1.0
        curr = self.get_trace()
        max_delta = 0.0
        for i in range(min(len(curr), len(prev_traces))):
            delta = abs(curr[i] - prev_traces[i]) / (abs(prev_traces[i]) + 1e-8)
            max_delta = max(max_delta, delta)
        return max_delta

    def compute_fisher_balance(self, mod_a=0, mod_b=1):
        """Compute Fisher balance B between two modalities.

        B = 4 * λ_a * λ_b / (λ_a + λ_b)^2  ∈ [0, 1]

        B→1 when modalities are balanced, B→0 when one dominates.
        """
        traces = self.get_trace()
        if len(traces) < max(mod_a, mod_b) + 1:
            return 0.5
        la, lb = traces[mod_a], traces[mod_b]
        if la == 0 and lb == 0:
            return 1.0
        total = la + lb + 1e-8
        return min(4.0 * la * lb / (total * total), 1.0)

    def compute_fisher_threshold(self, warmup_epochs, current_epoch,
                                 mod_a=0, mod_b=1, base_min=0.02, base_range=0.15,
                                 progress_epochs=40):
        """Compute Fisher-informed progressive confidence threshold.

        τ_conf(e) = (1 - 0.5 * B_fisher) * (base_min + base_range * progress)

        Args:
            warmup_epochs: number of warmup epochs
            current_epoch: current training epoch
            mod_a, mod_b: modality indices for balance computation
            base_min: minimum threshold (default 0.1)
            base_range: range of threshold increase (default 0.3)
            progress_epochs: epochs over which threshold ramps up

        Returns:
            float: Fisher-informed confidence threshold
        """
        progress = min(1.0, (current_epoch - warmup_epochs) / progress_epochs)
        if current_epoch < warmup_epochs:
            progress = 0.0

        base_threshold = base_min + base_range * progress

        balance = self.compute_fisher_balance(mod_a, mod_b)
        # Adjust threshold based on Fisher balance:
        # Unbalanced → higher threshold, Balanced → use base
        fisher_factor = 1.0 - 0.5 * balance
        return float(fisher_factor * base_threshold)


# ============================================================
# Prototype Bank (CrossTab-inspired)
# ============================================================

class PrototypeBank(nn.Module):
    """Maintain 4 prototypes per class for a single modality.

    Prototypes are EMA-updated representative embeddings. Following the
    CrossTab approach, prototypes capture intra-class variation and serve
    as anchors for cross-modal pseudo-pairing.

    Each modality has its own PrototypeBank with 4 prototypes per class.
    """

    def __init__(self, embed_dim=256, num_classes=3, num_prototypes_per_class=4, ema_momentum=0.99):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_protos = num_prototypes_per_class
        self.ema_momentum = ema_momentum

        self.register_buffer('prototypes',
                             torch.randn(num_classes, num_prototypes_per_class, embed_dim) * 0.02)
        self.register_buffer('initialized_flags', torch.zeros(num_classes, dtype=torch.bool))

    @property
    def initialized(self):
        return self.initialized_flags.tolist()

    @initialized.setter
    def initialized(self, values):
        self.initialized_flags.copy_(torch.tensor(values, dtype=torch.bool))

    @torch.no_grad()
    def update(self, embeddings, labels):
        """Update prototypes via EMA using new batch embeddings.

        Args:
            embeddings: (B, embed_dim) encoded features
            labels: (B,) class labels OR (B, C) multi-hot labels
        """
        if labels.dim() == 2 and labels.shape[1] == self.num_classes:
            self.update_multilabel(embeddings, labels)
            return
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            class_embs = embeddings[mask]  # (N_c, embed_dim)

            if not self.initialized_flags[c]:
                # Initialize via k-means++ style: pick diverse points
                self._kmeans_plusplus_init(class_embs, c)
                self.initialized_flags[c] = True
            else:
                # Assign each sample to nearest prototype
                dists = torch.cdist(class_embs.unsqueeze(0),
                                    self.prototypes[c].unsqueeze(0)).squeeze(0)  # (N_c, P)
                assignments = dists.argmin(dim=1)  # (N_c,)

                for p in range(self.num_protos):
                    assigned = class_embs[assignments == p]
                    if assigned.size(0) > 0:
                        new_proto = assigned.mean(dim=0)
                        old_proto = self.prototypes[c, p].clone()
                        self.prototypes[c, p] = (self.ema_momentum * old_proto +
                                                  (1 - self.ema_momentum) * new_proto)
                        self.prototypes[c, p] = F.normalize(self.prototypes[c, p], dim=0)

    def _kmeans_plusplus_init(self, embeddings, class_idx):
        """Initialize prototypes using k-means++ seeding for diversity."""
        n = embeddings.size(0)
        if n <= self.num_protos:
            self.prototypes[class_idx, :n] = F.normalize(embeddings, dim=1)
            if n < self.num_protos:
                self.prototypes[class_idx, n:] = F.normalize(
                    embeddings[torch.randint(0, n, (self.num_protos - n,))], dim=1
                )
            return

        # Pick first centroid randomly
        first_idx = torch.randint(0, n, (1,)).item()
        centroids = [F.normalize(embeddings[first_idx:first_idx+1], dim=1).squeeze(0)]
        centroid_indices = [first_idx]

        for _ in range(1, self.num_protos):
            dists = []
            for c_vec in centroids:
                d = torch.norm(embeddings - c_vec.unsqueeze(0), dim=1)
                dists.append(d)
            min_dists = torch.stack(dists, dim=1).min(dim=1)[0]
            total_distance = min_dists.sum()
            if total_distance <= 1e-8:
                # Repeated embeddings legitimately have zero distance to every
                # seeded centroid; multinomial cannot sample an all-zero vector.
                next_idx = torch.randint(n, (1,)).item()
            else:
                next_idx = torch.multinomial(min_dists / total_distance, 1).item()
            centroids.append(F.normalize(embeddings[next_idx:next_idx+1], dim=1).squeeze(0))
            centroid_indices.append(next_idx)

        for p, c_vec in enumerate(centroids):
            self.prototypes[class_idx, p] = c_vec

    def get_prototype(self, class_idx, proto_idx=None):
        """Get a prototype for a given class. Random if proto_idx not specified."""
        if proto_idx is None:
            proto_idx = torch.randint(0, self.num_protos, (1,)).item()
        return self.prototypes[class_idx, proto_idx]

    def get_all_prototypes(self, class_idx):
        """Return all prototypes for a class: (num_protos, embed_dim)."""
        return self.prototypes[class_idx]

    def compute_prototype_loss(self, embeddings, labels, temperature=0.07):
        """Prototype-based contrastive loss.

        Pulls embeddings toward their class prototypes and pushes away from others.
        Supports both scalar labels (B,) and multi-hot labels (B, C).
        """
        if labels.dim() == 2 and labels.shape[1] == self.num_classes:
            return self.compute_prototype_loss_multilabel(embeddings, labels, temperature)
        B = embeddings.size(0)
        all_protos = self.prototypes.view(-1, self.embed_dim)  # (C*P, D)
        proto_labels = torch.arange(self.num_classes, device=embeddings.device).repeat_interleave(self.num_protos)

        sim = torch.mm(F.normalize(embeddings, dim=1),
                       F.normalize(all_protos, dim=1).T) / temperature  # cosine / tau
        mask = (labels.unsqueeze(1) == proto_labels.unsqueeze(0)).float()  # (B, C*P)
        mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)

        loss = -(mask * F.log_softmax(sim, dim=1)).sum(dim=1).mean()
        return loss

    @torch.no_grad()
    def update_multilabel(self, embeddings, multilabels):
        """Update prototypes with multi-label support.

        Args:
            embeddings: (B, embed_dim)
            multilabels: (B, C) multi-hot float tensor
        """
        for c in range(self.num_classes):
            mask = multilabels[:, c] > 0.5
            if mask.sum() == 0:
                continue
            class_embs = embeddings[mask]
            if not self.initialized_flags[c]:
                self._kmeans_plusplus_init(class_embs, c)
                self.initialized_flags[c] = True
            else:
                dists = torch.cdist(class_embs.unsqueeze(0),
                                    self.prototypes[c].unsqueeze(0)).squeeze(0)
                assignments = dists.argmin(dim=1)
                for p in range(self.num_protos):
                    assigned = class_embs[assignments == p]
                    if assigned.size(0) > 0:
                        new_proto = assigned.mean(dim=0)
                        old_proto = self.prototypes[c, p].clone()
                        self.prototypes[c, p] = (self.ema_momentum * old_proto +
                                                  (1 - self.ema_momentum) * new_proto)
                        self.prototypes[c, p] = F.normalize(self.prototypes[c, p], dim=0)

    def compute_prototype_loss_multilabel(self, embeddings, multilabels, temperature=0.07):
        """Prototype contrastive loss for multi-label.

        Pulls embeddings toward prototypes of ALL active classes.
        """
        B = embeddings.size(0)
        all_protos = self.prototypes.view(-1, self.embed_dim)  # (C*P, D)
        proto_labels = torch.arange(self.num_classes, device=embeddings.device).repeat_interleave(self.num_protos)

        sim = torch.mm(F.normalize(embeddings, dim=1),
                       F.normalize(all_protos, dim=1).T) / temperature
        # mask[b, c*P+p] = 1 if multilabels[b, c] > 0.5
        mask = (multilabels[:, proto_labels] > 0.5).float()  # (B, C*P)
        mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)

        loss = -(mask * F.log_softmax(sim, dim=1)).sum(dim=1).mean()
        return loss


# ============================================================
# Uncertainty Estimator
# ============================================================

class ModalityUncertaintyEstimator(nn.Module):
    """Estimate per-modality uncertainty from Fisher trace and prediction entropy.

    Combines:
    - Fisher-based uncertainty: low Fisher trace -> high uncertainty
    - Prediction entropy: high entropy -> high uncertainty
    """

    def __init__(self, embed_dim=256):
        super().__init__()
        self.uncertainty_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )

    def forward(self, embedding):
        """Return log-uncertainty for each sample."""
        return self.uncertainty_net(embedding)  # (B, 1)

    @staticmethod
    def compute_fisher_uncertainty(fisher_traces, epsilon=1e-8):
        """Compute Fisher-based uncertainty from trace values (v8: 70/30 weighting).

        Uses Fisher magnitude (how much info is being learned) as primary signal.
        Returns scalar in [0,1] representing modality-level uncertainty.
        
        u_fisher = |log(λ_m / λ_other)| / (1 + |log(λ_m / λ_other)|) * (λ_m + λ_other) / (1 + λ_m + λ_other)
        """
        import math
        if len(fisher_traces) < 2:
            return [0.5]
        la = max(fisher_traces[0], epsilon)
        lb = max(fisher_traces[1], epsilon)
        log_ratio = abs(math.log(la / lb + epsilon))
        imbalance = log_ratio / (1.0 + log_ratio)
        total = la + lb
        magnitude = total / (1.0 + total)
        # 70% magnitude + 30% imbalance
        u = 0.7 * magnitude + 0.3 * imbalance
        u = max(0.02, min(0.98, float(u)))
        return [u, u]

    @staticmethod
    def compute_entropy_uncertainty(logits):
        """Compute prediction entropy uncertainty from logits.

        u_entropy = H(y_hat) / log(C)  ∈ [0, 1]

        Args:
            logits: (B, C) model output logits

        Returns:
            (B,) tensor of entropy-based uncertainties
        """
        import math
        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=1)
        C = logits.size(1)
        return entropy / math.log(C)
