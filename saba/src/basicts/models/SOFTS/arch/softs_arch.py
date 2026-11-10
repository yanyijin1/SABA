import torch
from torch import nn
from torch.nn import functional as F

from ..config.softs_config import SOFTSConfig


class DataEmbeddingInverted(nn.Module):
    def __init__(self, input_len: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(input_len, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, C] -> [B, C, hidden]
        return self.dropout(self.value_embedding(x.permute(0, 2, 1)))


class STAR(nn.Module):
    """STar Aggregate-Redistribute module from SOFTS."""

    def __init__(self, d_series: int, d_core: int):
        super().__init__()
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, x: torch.Tensor):
        batch_size, channels, _ = x.shape
        combined_mean = self.gen2(F.gelu(self.gen1(x)))
        if self.training:
            ratio = F.softmax(combined_mean, dim=1).permute(0, 2, 1).reshape(-1, channels)
            indices = torch.multinomial(ratio, 1).view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices).repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)
        out = self.gen4(F.gelu(self.gen3(torch.cat([x, combined_mean], dim=-1))))
        return out, None


class SOFTSEncoderLayer(nn.Module):
    def __init__(self, star: STAR, hidden_size: int, intermediate_size: int, dropout: float, activation: str):
        super().__init__()
        self.attention = star
        self.conv1 = nn.Conv1d(hidden_size, intermediate_size, kernel_size=1)
        self.conv2 = nn.Conv1d(intermediate_size, hidden_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor):
        new_x, attn = self.attention(x)
        x = self.norm1(x + self.dropout(new_x))
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class SOFTSForForecasting(nn.Module):
    """SOFTS forecasting model adapted from the official NeurIPS 2024 implementation."""

    def __init__(self, config: SOFTSConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features
        self.use_norm = config.use_norm
        self.enc_embedding = DataEmbeddingInverted(config.input_len, config.hidden_size, config.dropout)
        self.encoder = nn.ModuleList([
            SOFTSEncoderLayer(
                STAR(config.hidden_size, config.d_core),
                config.hidden_size,
                config.intermediate_size,
                config.dropout,
                config.activation,
            ) for _ in range(config.num_layers)
        ])
        self.projection = nn.Linear(config.hidden_size, config.output_len, bias=True)

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        x = inputs
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        _, _, n_vars = x.shape
        enc_out = self.enc_embedding(x)
        for layer in self.encoder:
            enc_out, _ = layer(enc_out)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :n_vars]
        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.output_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.output_len, 1)
        return dec_out[:, -self.output_len:, :]
