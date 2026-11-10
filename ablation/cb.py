"""Callback adapter for the isolated SABA ablation variants."""

from __future__ import annotations

from basicts.modules.saba import SABA
from basicts.runners.callback.saba_dropout import SABACallback

from abl import AblationSABA, ABLATION_VARIANTS


class AblationSABACallback(SABACallback):
    """Use an ablation allocator without changing the standard callback flow."""

    def __init__(self, *args, ablation_variant: str = "full", **kwargs):
        if ablation_variant not in ABLATION_VARIANTS:
            raise ValueError(
                f"Unknown ablation variant {ablation_variant!r}; "
                f"expected one of {ABLATION_VARIANTS}."
            )
        super().__init__(*args, **kwargs)
        self.ablation_variant = ablation_variant

    def on_train_start(self, runner, **kwargs) -> None:
        runner.logger.info(
            f"[SABA ablation={self.ablation_variant}] "
            f"response_ratio={self.response_ratio:.8f}, "
            f"model_gate={self.model_gate:.8f}, "
            f"p_min={self.p_min:.6f}, p_max={self.p_max:.6f}, "
            f"init_sensitivity={self.init_sensitivity:.4f}."
        )
        self.dropout_ts = AblationSABA(
            seq_len=None,
            p_min=self.p_min,
            p_max=self.p_max,
            init_alpha=self.init_alpha,
            init_sensitivity=self.init_sensitivity,
            detrend_method=self.detrend_method,
            use_instance_norm=self.use_instance_norm,
            use_sfm_anchor=self.use_sfm_anchor,
            scorer_type="spectral",
            model_gate=self.model_gate,
            response_ratio=self.response_ratio,
            ablation_variant=self.ablation_variant,
        ).to(next(runner.model.parameters()).device)
        self._wrap_model(runner)
