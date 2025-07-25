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


class AdaptiveNorm(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_scale = nn.Linear(cond_dim, dim)
        self.to_shift = nn.Linear(cond_dim, dim)

    def forward(self, x, cond):
        x = self.norm(x)
        scale = self.to_scale(cond)
        shift = self.to_shift(cond)
        return x * (1 + scale) + shift


# MLP Layer
class MLP(nn.Module):
    def __init__(self, dim, mlp_mult=4, cond_dim=None, dropout=0.):
        super(MLP, self).__init__()
        self.ff = Feedforward(dim=dim, mlp_mult=mlp_mult, dropout=dropout)
        self.norm = AdaptiveNorm(dim, cond_dim)

    def forward(self, x, cond=None):
        inp = x
        x = self.norm(x, cond)
        x = self.ff(x)
        return x+inp


class Feedforward(nn.Module):
    def __init__(self, dim, mlp_mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mlp_mult)
        dim_out = dim

        self.activation = nn.SiLU()
        self.to_mlp = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim//2, dim_out, bias=False)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = self.to_mlp(x)
        x1,x2 = x.chunk(2, dim=-1)
        x = self.activation(x1) * x2
        x = self.do(x)
        x = self.to_out(x)
        return x


class RectfiedFlowMLP(nn.Module):
    def __init__(self, input_dim, output_dim, cond_dim=None, dim=768, num_layers=12, mlp_mult=4, dropout=0.):
        super().__init__()

        self.linear_input = nn.Linear(input_dim, dim)
        self.norm_output = AdaptiveNorm(dim, cond_dim=cond_dim)
        self.linear_output = nn.Linear(dim, output_dim)

        if cond_dim is None:
            raise ValueError("Dimensionality of conditioning cond_dim must be provided!")

        self.layers = nn.ModuleList([
            MLP(dim, mlp_mult=mlp_mult, cond_dim=cond_dim, dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, x, cond):
        # x: noisy samples with shape (batch_size, ..., channels)
        # cond: conditioning information with shape (batch_size, channels).

        # get input features
        x = self.linear_input(x)

        for i in range(len(self.layers)):
            x = self.layers[i](x, cond)

        # get output
        x = self.norm_output(x, cond)
        x = self.linear_output(x)

        return x