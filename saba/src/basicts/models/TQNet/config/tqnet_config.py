from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class TQNetConfig(BasicTSModelConfig):
    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of variables."})
    cycle: int = field(default=24, metadata={"help": "Cycle length for temporal queries."})
    model_type: str = field(default="mlp", metadata={"help": "Author option, kept for compatibility."})
    use_revin: int = field(default=1, metadata={"help": "Whether to use instance normalization."})
    hidden_size: int = field(default=512, metadata={"help": "Hidden size used by the MLP."})
    dropout: float = field(default=0.5, metadata={"help": "Dropout rate in the output projection."})
    use_tq: bool = field(default=True, metadata={"help": "Whether to use temporal query parameters."})
    channel_aggre: bool = field(default=True, metadata={"help": "Whether to use channel aggregation attention."})
    channel_aggre_heads: int = field(default=4, metadata={"help": "Number of channel aggregation attention heads."})
    channel_aggre_dropout: float = field(default=0.5, metadata={"help": "Dropout in channel aggregation attention."})
