# SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning
# Hojoon Lee, Dongyoon Hwang, Donghu Kim, Hyunseung Kim, Jun Jet Tai, Kaushik Subramanian,
# Peter R. Wurman, Jaegul Choo, Peter Stone, Takuma Seno (ICLR 2025)
# https://github.com/SonyResearch/simba

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSNorm(nn.Module):
    """Running-statistics observation normalization (SimBa eqs. 3-4).

    Matches the SonyResearch/simba ObservationNormalizer + RunningMeanStd:
    update stats from on-policy interaction observations, then re-normalize
    replay samples with the current statistics (do not store normalized obs).
    """

    def __init__(self, num_features, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.ones(num_features))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        obs = x.detach().reshape(-1, self.mean.shape[0]).to(dtype=self.mean.dtype)
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0, unbiased=False)
        batch_count = obs.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / tot_count
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / tot_count)
        self.count.copy_(tot_count)

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


class ResidualFFBlock(nn.Module):
    """Pre-LN residual feedforward block: x + MLP(LN(x)), 4x inverted bottleneck."""

    def __init__(self, hidden_dim, expansion=4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * expansion)
        self.fc2 = nn.Linear(hidden_dim * expansion, hidden_dim)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.norm(x)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class SimBaEncoder(nn.Module):
    """Linear embed -> L residual blocks -> post-LN."""

    def __init__(self, in_dim, hidden_dim, num_blocks, expansion=4):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        nn.init.orthogonal_(self.embed.weight, gain=1.0)
        nn.init.zeros_(self.embed.bias)
        self.blocks = nn.ModuleList(
            [ResidualFFBlock(hidden_dim, expansion=expansion) for _ in range(num_blocks)]
        )
        self.post_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.post_norm(x)
