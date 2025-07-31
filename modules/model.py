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
        x1, x2 = x.chunk(2, dim=-1)
        x = self.activation(x1) * x2
        x = self.do(x)
        x = self.to_out(x)
        return x


# MLP Layer with improved residual connection
class MLP(nn.Module):
    def __init__(self, dim, mlp_mult=4, cond_dim=None, dropout=0.):
        super(MLP, self).__init__()
        self.ff = Feedforward(dim=dim, mlp_mult=mlp_mult, dropout=dropout)
        if cond_dim is None:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = AdaptiveNorm(dim, cond_dim)
        
        # # layer scale for better training stability
        # self.layer_scale = nn.Parameter(torch.ones(dim) * 1e-6)

    def forward(self, x, cond=None):
        inp = x
        if cond is None:
            x = self.norm(x)
        else:
            # Apply conditional normalization
            x = self.norm(x, cond)
        x = self.ff(x)
        # # Apply layer scaling
        # x = x * self.layer_scale
        return x + inp


class RectifiedFlowMLP(nn.Module):
    def __init__(self, input_dim, output_dim, time_dim=256, dim=768, 
                 num_layers=12, mlp_mult=4, dropout=0.):
        super().__init__()
        
        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        
        # Input projection
        self.linear_input = nn.Linear(input_dim, dim)
        
        # MLP layers with time conditioning only
        self.layers = nn.ModuleList([
            MLP(dim, mlp_mult=mlp_mult, 
                cond_dim=time_dim if i == 0 else None, 
                dropout=dropout) 
            for i in range(num_layers)
        ])
        
        # Output layers
        # self.norm_output = AdaptiveNorm(dim, cond_dim=time_dim)
        self.norm_output = nn.LayerNorm(dim)
        self.linear_output = nn.Linear(dim, output_dim)
        
        # Initialize output layer to zero for better training start
        nn.init.zeros_(self.linear_output.weight)
        if self.linear_output.bias is not None:
            nn.init.zeros_(self.linear_output.bias)

    def forward(self, x, t):
        """
        x: noisy samples with shape (batch_size, ..., input_dim)
        t: time values with shape (batch_size,) in [0, 1]
        """
        # Get time embeddings
        t_emb = self.time_embedding(t)  # (batch_size, time_dim)
        
        # Process input
        x = self.linear_input(x)
        
        # Pass through MLP layers with time conditioning
        for layer in self.layers:
            x = layer(x, t_emb)
        
        # Output with time conditioning
        # x = self.norm_output(x, t_emb)
        x = self.norm_output(x)
        x = self.linear_output(x)
        
        return x


