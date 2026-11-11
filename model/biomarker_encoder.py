"""Clinical tabular MLP used by the Clinic branch."""
import torch
import torch.nn as nn


class ClinicalDataEncoder(nn.Module):
    """
    临床数据编码器 - Simple MLP
    处理：人口统计学、认知评估、神经心理学测试等
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 512,
        dropout: float = 0.2,
        num_classes: int = 4
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Simple MLP
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor, return_causal: bool = True):
        """
        Args:
            x: 临床数据特征 (B, input_dim)
            return_causal: 是否返回因果因子 (ignored for MLP, kept for compatibility)

        Returns:
            logits: 分类输出
            features: 编码后的特征 (B, embed_dim)
            causal: None (for compatibility)
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits, features, None

