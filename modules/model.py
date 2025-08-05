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
        t = (t * 2.) - 1.  # Rescale to [-1, 1]

        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        exponents = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        angle_rates = t[:, None] * exponents[None, :]  # (B, half_dim)
        return torch.cat([torch.sin(angle_rates), torch.cos(angle_rates)], dim=-1)  # (B, dim)
    
class TimeEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.embed = SinusoidalTimeEmbedding(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):  # t: shape [batch]
        raw = self.embed(t)  # shape: [batch, input_dim]
        return self.proj(raw)


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
        if isinstance(self.norm, AdaptiveNorm):
            x = self.norm(x, cond)
        else:
            x = self.norm(x)
        x = self.ff(x)
        # # Apply layer scaling
        # x = x * self.layer_scale
        return x + inp


class RectifiedFlowMLP(nn.Module):
    def __init__(self, input_dim, output_dim, time_dim=256, dim=768, 
                 num_layers=12, mlp_mult=4, dropout=0.0):
        super().__init__()
        
        # Time embedding
        # self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_embedding = TimeEmbedding(input_dim=time_dim, hidden_dim=time_dim)
        
        # Input projection
        self.linear_input = nn.Linear(input_dim, dim)
        
        # MLP layers with time conditioning only
        # self.layers = nn.ModuleList([
        #     MLP(dim, mlp_mult=mlp_mult, 
        #         cond_dim=time_dim if i == 0 else None, 
        #         dropout=dropout) 
        #     for i in range(num_layers)
        # ])
        self.layers = nn.ModuleList([
            MLP(dim, mlp_mult=mlp_mult, 
                cond_dim=time_dim, 
                dropout=dropout) 
            for i in range(num_layers)
        ])
        
        # Output layers
        self.norm_output = AdaptiveNorm(dim, cond_dim=time_dim)
        # self.norm_output = nn.LayerNorm(dim)
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
        x = self.norm_output(x, t_emb)
        # x = self.norm_output(x)
        x = self.linear_output(x)
        
        return x


class VanillaMLP(nn.Module):
    def __init__(self, input_dim=128, time_dim=32, hidden_dim=512, depth=5):
        """
        Args:
            input_dim: Dimensionality of the fingerprint vector
            time_embed_dim: Dimensionality of time embedding
            hidden_dim: Width of each hidden layer
            depth: Number of hidden layers
        """
        super().__init__()
        self.time_embedding = TimeEmbedding(input_dim=time_dim, hidden_dim=time_dim)

        layers = []
        in_dim = input_dim + time_dim

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