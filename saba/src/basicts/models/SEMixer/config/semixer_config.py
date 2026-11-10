from dataclasses import dataclass, field
from typing import Sequence

from basicts.configs import BasicTSModelConfig


@dataclass
class SEMixerConfig(BasicTSModelConfig):
    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of variables."})
    hidden_size: int = field(default=128, metadata={"help": "Embedding dimension."})
    patch_len: int = field(default=None, metadata={"help": "Base patch length."})
    stride: int = field(default=None, metadata={"help": "Base patch stride."})
    reduce_dim: int = field(default=64, metadata={"help": "Multiscale fusion dimension."})
    scale_factors: Sequence[int] = field(default_factory=lambda: (1, 2, 4, 8), metadata={"help": "Multiscale patch factors."})
    eib_num: int = field(default=1, metadata={"help": "Progressive mixer blocks per later scale."})
    eib_num_1scale: int = field(default=1, metadata={"help": "Mixer blocks at the base scale."})
    head_dropout: float = field(default=0.0, metadata={"help": "Native prediction-head dropout."})
    affine: bool = field(default=False, metadata={"help": "Use affine RevIN parameters."})
    subtract_last: bool = field(default=False, metadata={"help": "Use the final value as the RevIN reference."})
    individual: bool = field(default=False, metadata={"help": "Use per-variable heads."})
    var_individual: bool = field(default=False, metadata={"help": "Use variable-independent heads."})
    var_decomp: bool = field(default=False, metadata={"help": "Use grouped variable heads."})
    var_sp_num: int = field(default=1, metadata={"help": "Number of grouped variable heads."})
    sample_num: int = field(default=5, metadata={"help": "Original SEMixer sample count setting."})
    connection_probability: float = field(default=0.85, metadata={"help": "Original RAM connection probability."})
    random_attention: bool = field(default=True, metadata={"help": "Enable original Random Attention Mechanism."})
    saba_dropout: float = field(default=0.0, metadata={"help": "SABA dropout after multiscale fusion."})

    def __post_init__(self) -> None:
        if self.input_len is None or self.output_len is None or self.num_features is None:
            raise ValueError("SEMixer requires input_len, output_len, and num_features.")
        base_patch = max(1, self.input_len // 16)
        self.patch_len = base_patch if self.patch_len is None else self.patch_len
        self.stride = base_patch if self.stride is None else self.stride
        if tuple(self.scale_factors) != (1, 2, 4, 8):
            raise ValueError("SEMixer currently implements the official four scales (1, 2, 4, 8).")
        if self.patch_len * max(self.scale_factors) > self.input_len:
            raise ValueError("SEMixer patch_len and scale_factors exceed input_len.")

        # Preserve the official implementation's attribute contract.
        self.seq_len = self.input_len
        self.pred_len = self.output_len
        self.c_in = self.num_features
        self.enc_in = self.num_features
        self.d_model = self.hidden_size
        self.multi_scale = True
        self.padding_patch = "end"
        self.Random_Attention_Mechanism = self.random_attention
        self.Self_Attention_Mechanism = False
        self.prob_attn = False
        self.logsparse = False
        self.peformer_attn = False
        self.reformer_attn = False
        self.autocorrelation = False
        self.fed_fourier_attn = False
        self.self_attn = False
        self.factor = 1
        self.output_attention = False
        self.dropout = 0.1
        self.test = False
