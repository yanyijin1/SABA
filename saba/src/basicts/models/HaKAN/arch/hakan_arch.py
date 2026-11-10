import torch
from torch import nn
from torch import Tensor

from ..config.hakan_config import HaKANConfig
from .hahn import HahnPolynomials
from .revin import RevIN


class HahnKANBlock(nn.Module):
    def __init__(self, dim, len):
        super().__init__()
        self.intrapatch_kan = HahnPolynomials(dim, dim, 3, 1, 1, 7)
        self.interpatch_kan = HahnPolynomials(len, len, 3, 1, 1, 7)


    def forward(self, x):
        x = self.intrapatch_kan(x)
        x = x.permute(0,2,1)
        x = self.interpatch_kan(x)
        x = x.permute(0,2,1)
        return x

class Model(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int, d_model: int = 128, padding_patch: str = None, revin: bool = True, affine: bool = True, subtract_last: bool = False, num_blocks: int = 5, saba_dropout: float = 0.0):

        super().__init__()

        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.backbone = BackBone(patch_num=patch_num, patch_len=patch_len, d_model=d_model, num_blocks=num_blocks)

        self.head_nf = d_model * patch_num
        self.n_vars = c_in

        self.conv1 = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)

        self.revert = nn.Linear(d_model, c_in)
        self.head = Flatten_Head(self.head_nf, target_window, saba_dropout)


    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        if self.revin:
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)

        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]

        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window]

        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
        return z



class Flatten_Head(nn.Module):
    def __init__(self, nf, target_window, saba_dropout=0.0):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear1 = nn.Linear(nf, 336)
        self.saba_dropout = nn.Dropout(saba_dropout)
        self.linear2 = nn.Linear(336, target_window)

    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.saba_dropout(x)
        x = self.linear2(x)
        return x



class BackBone(nn.Module):
    def __init__(self, patch_num, patch_len, d_model=128, num_blocks=5):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = q_len
        self.W_pos = nn.Parameter(torch.randn(1, q_len, d_model))
        self.encoder = nn.ModuleList([HahnKANBlock(d_model, self.patch_num) for _ in range(num_blocks)])


    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        n_vars = x.shape[1]
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = u + self.W_pos[:, :self.seq_len, :]

        z = u
        for layer in self.encoder:
            x = layer(z)                                                         # z: [bs * nvars x patch_num x d_model]
            z = z + x

        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]

        return z


class HaKANForForecasting(nn.Module):
    """BasicTS adapter for the official HaKAN forecasting model."""

    saba_dropout_only = True
    saba_dropout_layer_names = ("model.head.saba_dropout",)

    def __init__(self, config: HaKANConfig):
        super().__init__()
        self.model = Model(
            c_in=config.num_features,
            context_window=config.input_len,
            target_window=config.output_len,
            patch_len=config.patch_len,
            stride=config.stride,
            d_model=config.hidden_size,
            padding_patch=config.padding_patch,
            revin=config.use_revin,
            affine=config.affine,
            subtract_last=config.subtract_last,
            num_blocks=config.num_blocks,
            saba_dropout=config.dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 4:
            if inputs.shape[-1] != 1:
                raise ValueError("HaKAN expects a single value channel in 4D BasicTS inputs.")
            inputs = inputs[..., 0]
        if inputs.ndim != 3:
            raise ValueError(f"HaKAN expects [batch, length, variables], got {tuple(inputs.shape)}.")
        # The official HaKAN RevIN layer normalizes the last (variable) axis,
        # so its public input contract is already [batch, length, variables].
        return self.model(inputs)
