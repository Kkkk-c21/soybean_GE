"""三个模型共用同一套网络组件：G-only / E-only / G+E。

输入约定（与现有 SoyDataset 对齐）：
    gene: [batch_size, num_snps]，每个 SNP 编码为 0、1、2
    env:  [batch_size, num_days, num_env_features]，例如 [B, 100, 18]
输出：
    prediction: [batch_size]，连续油分预测值

注意：基因型向量必须在所有样本中使用同一 SNP 顺序。
"""

import torch
from torch import nn
from torch.nn import functional as F


class GeneEncoder(nn.Module):
    """把一维 SNP 向量编码为固定长度的基因特征。

    采用大步长 1D 卷积，避免对几十万甚至更多 SNP
    直接使用一个巨大的全连接层。

    这是教学用的轻量 CNN 基线，并非特别针对 SNP 位点效应
    设计的最优遗传模型；对照实验可再添加 RR-BLUP 等方法。
    """

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            # [B, 1, padded_N] -> [B, 8, padded_N/64]
            # kernel_size=stride=64；forward 中会补齐尾端不足 64 的窗口。
            nn.Conv1d(1, 8, kernel_size=64, stride=64),
            nn.ReLU(),
            # 进一步提取相邻窗口间的局部关联。
            nn.Conv1d(8, 32, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            # 无论 SNP 数量是多少，都得到 [B, 32, 16]。
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

    def forward(self, gene: torch.Tensor) -> torch.Tensor:
        if gene.ndim != 2:
            raise ValueError(f"gene 应为 [B, N]，实际为 {tuple(gene.shape)}")
        if gene.shape[1] < 64:
            raise ValueError("当前基因 CNN 要求至少 64 个 SNP。")

        # npy 中的整数 0/1/2 转成 float32；float64不做额外编码或裁剪。int
        gene = gene.to(dtype=torch.float32)
        # Conv1d 默认会舍弃不足一个 kernel 的末尾 SNP，先补齐可避免漏掉位点。
        pad_size = (-gene.shape[1]) % 64
        if pad_size:
            gene = F.pad(gene, (0, pad_size), value=0.0)
        gene = gene.unsqueeze(1)  # [B, padded_N] -> [B, 1, padded_N]
        return self.network(gene)  # [B, feature_dim]


class EnvironmentEncoder(nn.Module):
    """用时序 1D CNN 编码环境因子（保留较粗的时间位置信息）。"""

    def __init__(self, num_env_features: int, feature_dim: int = 64):
        super().__init__()
        if num_env_features <= 0:
            raise ValueError("num_env_features 必须大于 0")

        self.num_env_features = num_env_features
        self.network = nn.Sequential(
            nn.Conv1d(num_env_features, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            # 将完整时间序列汇总为 10 个时间区间。
            nn.AdaptiveAvgPool1d(10),
            nn.Flatten(),
            nn.Linear(32 * 10, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

    def forward(self, env: torch.Tensor) -> torch.Tensor:
        if env.ndim != 3:
            raise ValueError(f"env 应为 [B, days, features]，实际为 {tuple(env.shape)}")
        if env.shape[-1] != self.num_env_features:
            raise ValueError(
                f"环境特征数不匹配：预期 {self.num_env_features}，"
                f"实际 {env.shape[-1]}；请核查 Dataset 的 env 轴顺序。"
            )

        env = env.to(dtype=torch.float32)
        env = env.transpose(1, 2)  # [B, days, F] -> [B, F, days]
        return self.network(env)  # [B, feature_dim]


class GxEModel(nn.Module):
    """统一接口，用 mode 选择启用哪些分支。

    mode='g_only'：只训练基因编码器
    mode='e_only'：只训练环境编码器
    mode='g_e'：    两个分支都训练，拼接后送入同结构的 MLP。

    三种模式分别新建模型、独立训练；共用的是代码结构，
    不是训练出来的权重。
    """

    MODES = ("g_only", "e_only", "g_e")

    def __init__(
        self,
        num_env_features: int,
        mode: str = "g_e",
        feature_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode 必须是 {self.MODES}，实际为 {mode!r}")

        self.mode = mode
        self.gene_encoder = (
            GeneEncoder(feature_dim) if mode in ("g_only", "g_e") else None
        )
        self.env_encoder = (
            EnvironmentEncoder(num_env_features, feature_dim)
            if mode in ("e_only", "g_e") else None
        )

        self.encode_dim = feature_dim * 2  if mode == "g_e" else feature_dim
        self.predictor = nn.Sequential(
            nn.Linear(self.encode_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, gene: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        gene_features = self.gene_encoder(gene) if self.gene_encoder else None
        env_features = self.env_encoder(env) if self.env_encoder else None

        combined = gene_features if env_features is None else (
            env_features if gene_features is None else torch.cat([gene_features, env_features], dim=-1)
        )
        return self.predictor(combined).squeeze(-1)  # [B]

if __name__ == "__main__":
    # 测试GxE模型前向传播
    n, num_snps, days, env_features = 24, 5837, 100, 18
    gene = torch.randint(0, 3, (n, num_snps))
    env = torch.randn(n, days, env_features)
    y = gene[:, :20].float().mean(dim=1) + env[:, :, 0].mean(dim=1)

    out = GxEModel(num_env_features=env_features, mode="g_e")(gene, env)
    print(out.shape)  # [B]

    # 测试 G-only 模式
    out_g = GxEModel(num_env_features=env_features, mode="g_only")(gene, env)
    print(out_g.shape)  # [B]

    # 测试 E-only 模式
    out_e = GxEModel(num_env_features=env_features, mode="e_only")(gene, env)
    print(out_e.shape)  # [B]
