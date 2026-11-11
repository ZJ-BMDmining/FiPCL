"""
切片注意力 3D ResNet 编码器
使用 2D ResNet-18 作为单切片特征提取器，配合切片注意力聚合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Optional
from torchvision.models import resnet18, ResNet18_Weights


class SlicePositionEmbedding(nn.Module):
    """
    切片位置编码：为 39 张切片添加可学习的位置信息
    """

    def __init__(self, num_slices: int = 39, embed_dim: int = 512):
        super().__init__()
        self.num_slices = num_slices
        self.embedding = nn.Parameter(torch.randn(1, num_slices, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) 切片特征序列
        Returns:
            (B, N, C) 添加位置编码后的特征
        """
        return x + self.embedding[:, :x.shape[1], :]


class SliceAttentionModule(nn.Module):
    """
    切片注意力模块：学习 39 张切片的重要性权重

    创新点：
    1. 使用自注意力机制聚合切片信息
    2. 可解释的切片权重可视化
    3. 支持多尺度切片特征
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_slices: int = 39,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_slices = num_slices
        self.embed_dim = embed_dim

        # 切片间自注意力
        self.slice_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # 可学习的查询向量（用于聚合）
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(
        self,
        slice_features: torch.Tensor,  # (B, N, C)
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            slice_features: 每张切片的特征 (B, num_slices, embed_dim)
            return_weights: 是否返回注意力权重

        Returns:
            aggregated: 聚合后的特征 (B, embed_dim)
            weights: 切片注意力权重 (B, num_slices)
        """
        B, N, C = slice_features.shape

        # 1. 添加 cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        x = torch.cat([cls_tokens, slice_features], dim=1)  # (B, N+1, C)

        # 2. 自注意力
        attn_output, attn_weights = self.slice_attention(x, x, x)

        # 3. 残差连接 + LayerNorm
        x = x + self.norm1(attn_output)

        # 4. MLP
        mlp_out = self.mlp(x)
        x = x + self.norm2(mlp_out)

        # 5. 提取 cls token 作为聚合特征
        aggregated = x[:, 0, :]  # (B, C)

        if return_weights:
            # cls token 对其他切片的注意力权重
            weights = attn_weights[:, 0, 1:]  # (B, N)
            return aggregated, weights

        return aggregated, None


class AdaptiveSliceAggregation(nn.Module):
    """
    自适应切片聚合：结合注意力加权平均和最大池化
    """

    def __init__(self, embed_dim: int = 512, num_slices: int = 39):
        super().__init__()

        # 注意力权重预测
        self.attention_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )

        self.num_slices = num_slices

    def forward(self, slice_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slice_features: (B, N, C)
        Returns:
            aggregated: (B, C)
        """
        # 1. 预测每张切片的注意力权重
        weights = self.attention_predictor(slice_features)  # (B, N, 1)
        weights = weights.squeeze(-1)  # (B, N)
        weights = F.normalize(weights, p=1, dim=1)  # 归一化

        # 2. 加权平均
        weighted_avg = torch.sum(
            slice_features * weights.unsqueeze(-1),
            dim=1
        )  # (B, C)

        # 3. 最大池化（捕捉最显著特征）
        max_pool = torch.max(slice_features, dim=1)[0]  # (B, C)

        # 4. 融合
        aggregated = weighted_avg + max_pool

        return aggregated


class Slice3DResNet(nn.Module):
    """
    基于 2D ResNet 的切片序列编码器

    架构:
    1. 输入：39 张切片 (B, 39, 1, 182, 182)
    2. 逐片特征提取：使用轻量级 2D CNN
    3. 切片注意力聚合
    4. 输出：全局特征 (B, 512)
    """

    def __init__(
        self,
        num_slices: int = 39,
        input_size: int = 182,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3,
        aggregation_method: str = 'attention'  # 'attention' or 'adaptive'
    ):
        super().__init__()

        self.num_slices = num_slices
        self.aggregation_method = aggregation_method

        # 1. 单切片特征提取器 (轻量级 2D ResNet-18)
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        resnet = resnet18(weights=weights)

        # 修改第一层以适应单通道输入
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.slice_encoder = nn.Sequential(*list(resnet.children())[:-1])  # 移除 FC 层
        self.slice_embed_dim = 512  # ResNet-18 输出维度

        # 2. 切片注意力聚合
        if aggregation_method == 'attention':
            self.slice_attention = SliceAttentionModule(
                embed_dim=self.slice_embed_dim,
                num_heads=8,
                num_slices=num_slices,
                dropout=dropout
            )
        else:
            self.adaptive_agg = AdaptiveSliceAggregation(
                embed_dim=self.slice_embed_dim,
                num_slices=num_slices
            )

        # 3. 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.slice_embed_dim),
            nn.Dropout(dropout),
            nn.Linear(self.slice_embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # 4. 中间特征投影（用于多模态融合）
        self.projection = nn.Sequential(
            nn.Linear(self.slice_embed_dim, self.slice_embed_dim),
            nn.LayerNorm(self.slice_embed_dim)
        )

    def forward(
        self,
        slices: torch.Tensor,  # (B, N, 1, H, W)
        return_intermediate: bool = True,
        return_weights: bool = False
    ):
        """
        前向传播

        Args:
            slices: 39 张切片 (B, 39, 1, 182, 182)
            return_intermediate: 是否返回中间特征
            return_weights: 是否返回切片权重

        Returns:
            logits: 分类输出
            features: 中间特征 (用于融合)
            weights: 切片注意力权重 (可选)
        """
        B, N, C_in, H, W = slices.shape

        # 1. 重排为 (B*N, C, H, W) 进行逐片处理
        slices_flat = rearrange(slices, 'b n c h w -> (b n) c h w')

        # 2. 提取每片特征
        slice_features = self.slice_encoder(slices_flat)  # (B*N, 512, 1, 1)
        slice_features = slice_features.view(B * N, -1)  # (B*N, 512)

        # 3. 重塑为序列 (B, N, C)
        slice_features = rearrange(
            slice_features, '(b n) c -> b n c',
            b=B, n=N
        )

        # 4. 切片注意力聚合
        if self.aggregation_method == 'attention':
            aggregated, weights = self.slice_attention(
                slice_features,
                return_weights=True
            )  # (B, C)
        else:
            aggregated = self.adaptive_agg(slice_features)
            weights = None

        # 5. 投影到共享特征空间
        features = self.projection(aggregated)

        # 6. 分类
        logits = self.classifier(features)

        if return_intermediate and return_weights:
            return logits, features, weights
        elif return_intermediate:
            return logits, features
        elif return_weights:
            return logits, weights
        else:
            return logits


# 使用示例
if __name__ == "__main__":
    # 模拟输入：batch=2, 39 切片，单通道，182x182
    dummy_input = torch.randn(2, 39, 1, 182, 182)

    model = Slice3DResNet(num_slices=39, num_classes=4, pretrained=False)
    logits, features, weights = model(dummy_input, return_weights=True)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")  # (2, 4)
    print(f"Features shape: {features.shape}")     # (2, 512)
    print(f"Slice weights shape: {weights.shape}") # (2, 39)
    print(f"Weights sum (should be ~1): {weights.sum(dim=1)}")

    # 测试 adaptive 聚合
    model_adaptive = Slice3DResNet(
        num_slices=39,
        num_classes=4,
        pretrained=False,
        aggregation_method='adaptive'
    )
    logits2, features2 = model_adaptive(dummy_input, return_intermediate=True)
    print(f"\nAdaptive aggregation:")
    print(f"Output logits shape: {logits2.shape}")
    print(f"Features shape: {features2.shape}")
