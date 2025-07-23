import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (batch,) in [0, 1]
        Returns: (batch, dim) sinusoidal embedding
        """
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        exponents = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        angle_rates = t[:, None] * exponents[None, :]  # (B, half_dim)
        return torch.cat([torch.sin(angle_rates), torch.cos(angle_rates)], dim=-1)  # (B, dim)


class RectifiedFlowMLP(nn.Module):
    def __init__(self, input_dim=128, time_embed_dim=32, hidden_dim=512, depth=5):
        """
        Args:
            input_dim: Dimensionality of the fingerprint vector
            time_embed_dim: Dimensionality of time embedding
            hidden_dim: Width of each hidden layer
            depth: Number of hidden layers
        """
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        layers = []
        in_dim = input_dim + time_embed_dim

        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, input_dim))  # Final layer: predict velocity

        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        """
        x: (batch, input_dim) → noisy fingerprint
        t: (batch,) → timestep in [0, 1]
        """
        t_embed = self.time_embed(t)  # (batch, time_embed_dim)
        x_input = torch.cat([x, t_embed], dim=-1)
        return self.net(x_input)
