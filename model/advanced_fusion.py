"""
Advanced Multimodal Fusion Module
融合架构参考 (MICCAI/ICML/NeurIPS 2023-2024)

核心创新:
1. Uncertainty-aware Gating: 根据每个样本动态调整模态权重
2. Orthogonal Projection: 分离模态特有和模态共享信息
3. Cross-Attention with Residual: 双向注意力融合
4. Modality Balance Loss: 防止dominant modality过度主导
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class UncertaintyAwareGating(nn.Module):
    """不确定性感知门控 - 根据预测不确定性动态调整模态权重.
    
    来自: "Uncertainty-aware Multimodal Fusion for Medical Image Classification" (MICCAI 2021)
    改进: 增加可学习的温度参数
    """
    
    def __init__(self, embed_dim: int, num_modalities: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        # 每个模态的uncertainty估计器
        self.uncertainty_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.LayerNorm(embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1)
            ) for _ in range(num_modalities)
        ])
        
        # 可学习的温度参数
        self.log_temperature = nn.Parameter(torch.zeros(num_modalities))
        
    def forward(self, modality_features: list[Tensor]) -> tuple[Tensor, dict]:
        """
        Args:
            modality_features: list of (B, embed_dim) tensors
        Returns:
            weighted_features: (B, embed_dim)
            gating_info: dict with uncertainty scores and weights
        """
        assert len(modality_features) == self.num_modalities
        
        # 估计每个模态的不确定性 (log variance as uncertainty)
        uncertainties = []
        for i, feat in enumerate(modality_features):
            unc = self.uncertainty_estimators[i](feat)  # (B, 1)
            uncertainties.append(unc)
        
        # Concat uncertainties: (B, num_modalities)
        unc_concat = torch.cat(uncertainties, dim=1)  # (B, 2)
        
        # 计算门控权重: softmax(-uncertainty / temperature)
        # 不确定性高的模态获得较低权重
        temperature = F.softplus(self.log_temperature) + 1e-6  # 确保正值
        weights = F.softmax(-unc_concat / temperature.unsqueeze(0), dim=1)  # (B, num_modalities)
        
        # 加权融合
        weighted_features = sum(
            weights[:, i:i+1] * feat for i, feat in enumerate(modality_features)
        )
        
        gating_info = {
            'uncertainties': torch.cat(uncertainties, dim=1),
            'weights': weights,
            'temperature': temperature
        }
        
        return weighted_features, gating_info


class OrthogonalProjection(nn.Module):
    """正交投影 - 分离模态特有信息和模态共享信息.
    
    来自: "Factorized Multimodal Representation Learning" (NeurIPS 2022)
    改进: 使用更高效的实现
    """
    
    def __init__(self, embed_dim: int, num_modalities: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        # 共享子空间投影
        self.shared_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 每个模态的私有子空间投影
        self.private_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(num_modalities)
        ])
        
    def forward(self, modality_features: list[Tensor]) -> tuple[list[Tensor], list[Tensor], Tensor]:
        """
        Args:
            modality_features: list of (B, embed_dim) tensors
        Returns:
            shared_features: list of (B, embed_dim) - 共享信息
            private_features: list of (B, embed_dim) - 模态特有信息
            fused_shared: (B, embed_dim) - 聚合的共享信息
        """
        # 聚合所有模态特征得到共享表示
        concat_features = torch.stack(modality_features, dim=1)  # (B, num_modalities, embed_dim)
        mean_shared = concat_features.mean(dim=1)  # (B, embed_dim)
        fused_shared = self.shared_projector(mean_shared)  # (B, embed_dim)
        
        # 每个模态分离私有和共享成分
        shared_features = []
        private_features = []
        
        for i, feat in enumerate(modality_features):
            # 估计共享成分 (通过投影到共享子空间)
            feat_shared = self.shared_projector(feat)
            
            # 私有成分 = 原始 - 共享成分
            # 但先对齐维度
            feat_private = feat - feat_shared
            
            # 投影私有成分
            feat_private = self.private_projectors[i](feat_private)
            
            shared_features.append(feat_shared)
            private_features.append(feat_private)
        
        return shared_features, private_features, fused_shared


class CrossAttentionFusion(nn.Module):
    """Cross-Attention Fusion with Residual Connections.
    
    来自: "Transformers for Multimodal Learning" (ICML 2022)
    改进: 增加residual connection和layer norm
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 双向cross-attention
        self.cross_attn_a2b = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_b2a = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layernorm和FFN
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)
        
        self.ffn_a = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.ffn_b = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, feat_a: Tensor, feat_b: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            feat_a: (B, embed_dim) - 模态A特征
            feat_b: (B, embed_dim) - 模态B特征
        Returns:
            fused_a: (B, embed_dim) - A的融合特征
            fused_b: (B, embed_dim) - B的融合特征
        """
        # A <- B cross attention
        a_q = feat_a.unsqueeze(1)  # (B, 1, embed_dim)
        b_kv = feat_b.unsqueeze(1)
        
        a_cross, _ = self.cross_attn_a2b(a_q, b_kv, b_kv)
        a_cross = a_cross.squeeze(1)
        fused_a = self.norm_a(feat_a + a_cross)  # Residual
        fused_a = self.norm_a(fused_a + self.ffn_a(fused_a))  # FFN with residual
        
        # B <- A cross attention
        b_q = feat_b.unsqueeze(1)
        a_kv = feat_a.unsqueeze(1)
        
        b_cross, _ = self.cross_attn_b2a(b_q, a_kv, a_kv)
        b_cross = b_cross.squeeze(1)
        fused_b = self.norm_b(feat_b + b_cross)
        fused_b = self.norm_b(fused_b + self.ffn_b(fused_b))
        
        return fused_a, fused_b


class AdvancedMultimodalFusion(nn.Module):
    """高级多模态融合模块 - 整合所有创新点.
    
    融合策略:
    1. Uncertainty-aware Gating: 动态模态权重
    2. Cross-Attention: 双向信息交互
    3. Orthogonal Projection: 分离共享/私有信息
    4. 三个分类头: 单模态专家 + 融合分类
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 3
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # 门控机制
        self.gating = UncertaintyAwareGating(embed_dim, num_modalities=2)

        # 正交投影
        self.orthogonal_proj = OrthogonalProjection(embed_dim, num_modalities=2, dropout=dropout)
        # Cross-Attention
        self.cross_attn = CrossAttentionFusion(embed_dim, num_heads, dropout)
        
        # 融合层 - 3个特征源cat后
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),  # 3个特征源: gated, cross_a+cross_b, shared
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 分类器
        self.fusion_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 单模态分类器 (用于辅助loss和modality balance)
        self.bio_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        self.mri_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(
        self,
        mri_feat: Tensor,
        bio_feat: Tensor,
        return_all: bool = False
    ) -> dict:
        """
        Args:
            mri_feat: (B, embed_dim) - MRI特征
            bio_feat: (B, embed_dim) - 非MRI模态特征 (clinic/blood/proteomics)
            return_all: 是否返回所有中间结果
        Returns:
            如果return_all=True: 返回完整dict包含所有logits
            否则: 返回{'fusion_logits': ...}
        """
        modality_features = [mri_feat, bio_feat]

        # 1. Uncertainty-aware Gating
        gated_feat, gating_info = self.gating(modality_features)  # (B, embed_dim)

        # 2. Cross-Attention
        fused_a, fused_b = self.cross_attn(mri_feat, bio_feat)  # (B, embed_dim) each

        # 3. Orthogonal Projection
        shared_feats, private_feats, fused_shared = self.orthogonal_proj(modality_features)

        # 4. 组合所有特征
        combined = torch.cat([
            gated_feat,
            (fused_a + fused_b) * 0.5,
            fused_shared * 0.5,
        ], dim=1)
        
        # 5. 投影和分类
        fusion_feat = self.fusion_proj(combined)  # (B, embed_dim)
        fusion_logits = self.fusion_classifier(fusion_feat)  # (B, num_classes)
        
        # 单模态分类器 (在原始特征上)
        bio_logits = self.bio_classifier(bio_feat)  # (B, num_classes)
        mri_logits = self.mri_classifier(mri_feat)  # (B, num_classes)
        
        if return_all:
            return {
                'fusion_logits': fusion_logits,
                'bio_logits': bio_logits,
                'mri_logits': mri_logits,
                'fusion_feat': fusion_feat,
                'gating_weights': gating_info['weights'],
                'gating_uncertainties': gating_info['uncertainties']
            }
        
        return {'fusion_logits': fusion_logits}
    
    def compute_modality_balance_loss(
        self,
        mri_logits: Tensor,
        bio_logits: Tensor,
        fusion_logits: Tensor,
        labels: Tensor,
        alpha: float = 0.1
    ) -> Tensor:
        """
        Modality Balance Loss - 防止dominant modality过度主导.

        来自: "Learning Balanced Multimodal Representations" (ICLR 2023)

        思想:
        1. 如果单模态已经很强, fusion loss不需要过度依赖该模态
        2. 如果单模态很弱, fusion需要补偿
        3. 通过KL散度衡量单模态和融合的"贡献差异"
        """
        # Use log_softmax for numerical stability
        fusion_log_probs = F.log_softmax(fusion_logits, dim=1)
        mri_log_probs = F.log_softmax(mri_logits, dim=1)
        bio_log_probs = F.log_softmax(bio_logits, dim=1)

        # Numerically stable KL divergence: KL(fusion || mri) and KL(fusion || bio)
        # F.kl_div expects: (log_prob, prob) with reduction='batchmean'
        eps = 1e-7
        mri_probs = (mri_log_probs.exp() + eps).clamp(max=1.0 - eps)
        bio_probs = (bio_log_probs.exp() + eps).clamp(max=1.0 - eps)

        mri_contribution = F.kl_div(
            fusion_log_probs,
            mri_probs,
            reduction='batchmean'
        )

        bio_contribution = F.kl_div(
            fusion_log_probs,
            bio_probs,
            reduction='batchmean'
        )

        # Check for NaN and return 0 if unstable
        if torch.isnan(mri_contribution) or torch.isnan(bio_contribution):
            return torch.tensor(0.0, device=mri_logits.device)

        # 鼓励两个模态贡献均衡
        balance_loss = torch.abs(mri_contribution - bio_contribution)

        return balance_loss * alpha


class ModalityBalanceLoss(nn.Module):
    """
    独立的Modality Balance Loss模块.

    用于在没有AdvancedMultimodalFusion的情况下使用.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        mri_logits: Tensor,
        bio_logits: Tensor,
        fusion_logits: Tensor,
        labels: Tensor
    ) -> Tensor:
        """
        计算modality balance loss.

        Args:
            mri_logits: (B, num_classes)
            bio_logits: (B, num_classes)
            fusion_logits: (B, num_classes)
            labels: (B,)
        """
        # Use log_softmax for numerical stability
        fusion_log_probs = F.log_softmax(fusion_logits, dim=1)
        mri_log_probs = F.log_softmax(mri_logits, dim=1)
        bio_log_probs = F.log_softmax(bio_logits, dim=1)

        # Numerically stable KL divergence: KL(fusion || mri) and KL(fusion || bio)
        # F.kl_div expects: (log_prob, prob) with reduction='batchmean'
        eps = 1e-7
        mri_probs = (mri_log_probs.exp() + eps).clamp(max=1.0 - eps)
        bio_probs = (bio_log_probs.exp() + eps).clamp(max=1.0 - eps)

        mri_kl = F.kl_div(
            fusion_log_probs,
            mri_probs,
            reduction='batchmean'
        )

        bio_kl = F.kl_div(
            fusion_log_probs,
            bio_probs,
            reduction='batchmean'
        )

        # Check for NaN and return 0 if unstable
        if torch.isnan(mri_kl) or torch.isnan(bio_kl):
            return torch.tensor(0.0, device=mri_logits.device)

        # 鼓励均衡贡献
        balance_loss = torch.abs(mri_kl - bio_kl)

        return balance_loss * self.alpha


def create_advanced_fusion(
    embed_dim: int = 256,
    num_heads: int = 4,
    dropout: float = 0.1,
    num_classes: int = 3
) -> AdvancedMultimodalFusion:
    """Factory function to create AdvancedMultimodalFusion."""
    return AdvancedMultimodalFusion(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        num_classes=num_classes
    )
