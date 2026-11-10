from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class HaKANConfig(BasicTSModelConfig):
    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of variables."})
    patch_len: int = field(default=16, metadata={"help": "Patch length."})
    stride: int = field(default=8, metadata={"help": "Patch stride."})
    hidden_size: int = field(default=128, metadata={"help": "HaKAN embedding dimension."})
    num_blocks: int = field(default=5, metadata={"help": "Number of Hahn-KAN residual blocks."})
    padding_patch: str = field(default="end", metadata={"help": "Patch padding mode."})
    use_revin: bool = field(default=True, metadata={"help": "Enable reversible instance normalization."})
    affine: bool = field(default=True, metadata={"help": "Use affine RevIN parameters."})
    subtract_last: bool = field(default=False, metadata={"help": "Use the final value as the RevIN reference."})
    dropout: float = field(default=0.0, metadata={"help": "SABA bottleneck dropout before the output projection."})
