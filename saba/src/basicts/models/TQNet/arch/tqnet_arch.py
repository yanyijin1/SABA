import torch
from torch import nn

from ..config.tqnet_config import TQNetConfig


class TQNetForForecasting(nn.Module):
    """TQNet forecasting model adapted from the official implementation."""

    def __init__(self, config: TQNetConfig):
        super().__init__()
        self.seq_len = config.input_len
        self.pred_len = config.output_len
        self.enc_in = config.num_features
        self.cycle_len = config.cycle
        self.model_type = config.model_type
        self.d_model = config.hidden_size
        self.dropout = config.dropout
        self.use_revin = config.use_revin

        self.use_tq = config.use_tq
        self.channel_aggre = config.channel_aggre

        if self.use_tq:
            self.temporalQuery = nn.Parameter(
                torch.zeros(self.cycle_len, self.enc_in),
                requires_grad=True,
            )

        if self.channel_aggre:
            self.channelAggregator = nn.MultiheadAttention(
                embed_dim=self.seq_len,
                num_heads=config.channel_aggre_heads,
                batch_first=True,
                dropout=config.channel_aggre_dropout,
            )

        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len),
        )

    def forward(
            self,
            inputs: torch.Tensor,
            cycle_index: torch.Tensor = None,
            inputs_timestamps: torch.Tensor = None,
            targets_timestamps: torch.Tensor = None,
            **kwargs) -> torch.Tensor:
        x = inputs
        if cycle_index is None:
            if targets_timestamps is not None:
                cycle_index = torch.floor(targets_timestamps[:, 0, 0] * self.cycle_len + 1e-6)
                cycle_index = torch.remainder(cycle_index, self.cycle_len).to(device=x.device, dtype=torch.long)
            elif inputs_timestamps is not None:
                cycle_index = torch.floor(inputs_timestamps[:, -1, 0] * self.cycle_len + 1.0 + 1e-6)
                cycle_index = torch.remainder(cycle_index, self.cycle_len).to(device=x.device, dtype=torch.long)
            else:
                cycle_index = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        else:
            cycle_index = cycle_index.to(device=x.device, dtype=torch.long).reshape(-1)

        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x_input = x.permute(0, 2, 1)

        if self.use_tq:
            offsets = torch.arange(self.seq_len, device=x.device).view(1, -1)
            gather_index = (cycle_index.view(-1, 1) + offsets) % self.cycle_len
            query_input = self.temporalQuery[gather_index].permute(0, 2, 1)
            if self.channel_aggre:
                channel_information = self.channelAggregator(
                    query=query_input,
                    key=x_input,
                    value=x_input,
                )[0]
            else:
                channel_information = query_input
        else:
            if self.channel_aggre:
                channel_information = self.channelAggregator(
                    query=x_input,
                    key=x_input,
                    value=x_input,
                )[0]
            else:
                channel_information = 0

        projected = self.input_proj(x_input + channel_information)
        hidden = self.model(projected)
        output = self.output_proj(hidden + projected).permute(0, 2, 1)

        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output
