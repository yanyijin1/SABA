from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class SOFTSConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    hidden_size: int = field(default=128)
    d_core: int = field(default=64)
    intermediate_size: int = field(default=256)
    num_layers: int = field(default=2)
    dropout: float = field(default=0.1)
    activation: str = field(default="gelu")
    use_norm: bool = field(default=True)
