"""Tabular SE encoder and MRI weight-loading helper used by FiPCL."""

import torch
import torch.nn as nn
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.slice_attention_resnet import Slice3DResNet


# ============================================================
# SE Encoder (from single_modality Blood/Proteomics scripts)
# ============================================================


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.squeeze(x.unsqueeze(-1)).squeeze(-1)
        return x * self.excite(y)


class SEEncoder(nn.Module):
    """Squeeze-and-Excitation encoder for tabular biomarkers (blood, proteomics)."""

    def __init__(self, input_dim, embed_dim, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.se1 = SqueezeExcitation(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.se2 = SqueezeExcitation(embed_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.se1(x)
        x = self.mlp(x)
        return self.se2(x)


# ============================================================
# Pretrained MRI loader
# ============================================================


def _fresh_mri_encoder(device="cuda", num_classes=3):
    """Randomly initialised slice-attention MRI encoder (no pretrained weights)."""
    return Slice3DResNet(
        num_slices=39, input_size=182, num_classes=num_classes,
        pretrained=False, dropout=0.0,
    ).to(device)


def load_pretrained_mri(device="cuda", freeze=True, num_classes=6):
    """Load MRI encoder with pretrained weights from single_modality checkpoint.

    The pretrained checkpoint was trained for 3-class (NC/MCI/AD). When num_classes
    differs, the classifier head is randomly initialized while backbone, attention,
    and projection layers inherit pretrained weights via strict=False.
    """
    ckpt_path = PROJECT_ROOT / "checkpoints" / "single_modality" / "MRI.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Pretrained MRI encoder not found at {ckpt_path}.\n"
            "Place the requested MRI checkpoint at that path, "
            "or build a randomly initialised encoder with pretrained_mri=False."
        )
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    mri_encoder = Slice3DResNet(
        num_slices=39, input_size=182, num_classes=num_classes, pretrained=False, dropout=0.0
    )

    mri_state = {
        k.replace("mri_encoder.", ""): v
        for k, v in ckpt.items()
        if k.startswith("mri_encoder.")
    }

    use_strict = (num_classes == 3)
    if use_strict:
        mri_encoder.load_state_dict(mri_state, strict=True)
    else:
        # Classifier shape differs (3 vs 6 classes) — filter mismatched keys
        model_state = mri_encoder.state_dict()
        filtered_state = {}
        skipped_keys = []
        for k, v in mri_state.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                skipped_keys.append(k)
        mri_encoder.load_state_dict(filtered_state, strict=False)
        if skipped_keys:
            print(f"  [MRI] num_classes={num_classes}: {len(skipped_keys)} params randomly initialized "
                  f"(classifier head reshaped {num_classes}-way), "
                  f"{len(filtered_state)} params from pretrained")

    if freeze:
        for param in mri_encoder.parameters():
            param.requires_grad = False
        mri_encoder.eval()

    return mri_encoder
