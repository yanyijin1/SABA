"""Isolated SABA probability variants used only for ablation experiments."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch.nn import functional as F

from basicts.modules.saba import SABA


ABLATION_VARIANTS = ("full", "no_backbone", "global_only", "recurring_only")


class AblationSABA(SABA):
    """SABA with one explicitly selected probability-path ablation."""

    def __init__(self, *args, ablation_variant: str = "full", **kwargs):
        if ablation_variant not in ABLATION_VARIANTS:
            raise ValueError(
                f"Unknown ablation variant {ablation_variant!r}; "
                f"expected one of {ABLATION_VARIANTS}."
            )
        super().__init__(*args, **kwargs)
        self.ablation_variant = ablation_variant

    def compute_dropout_rates(
        self,
        inputs: torch.Tensor,
        visualize: bool = False,
        max_samples: int = 4,
        return_score: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self._ensure_noise_scorer(inputs)
        details = self.noise_scorer(inputs, return_details=True)
        global_raw_score = details["noise_scores"]
        relative_score = self._batch_minmax(global_raw_score)
        context = self.temporal_reliability(
            details["original"], details["reconstructed"]
        )
        context_ratio = context["unsupported_ratio"]
        strength = self.context_strength.to(
            context_ratio.device, context_ratio.dtype
        )
        effective_context_ratio = 1.0 - strength * (1.0 - context_ratio)

        if self.ablation_variant == "full":
            context_score = relative_score * effective_context_ratio
            model_gate = self.model_gate
        elif self.ablation_variant == "no_backbone":
            # Neutral backbone response: R=1, hence eta_eff=eta and v=1.
            context_score = relative_score * context_ratio
            model_gate = torch.ones_like(self.model_gate)
        elif self.ablation_variant == "global_only":
            # Remove recurring-support modulation while retaining R_dm.
            context_score = relative_score
            model_gate = self.model_gate
        elif self.ablation_variant == "recurring_only":
            # Remove the global residual magnitude while retaining R_dm.
            context_score = effective_context_ratio
            model_gate = self.model_gate
        else:
            raise RuntimeError(f"Unhandled ablation variant {self.ablation_variant!r}")

        mapped = torch.tanh(context_score * F.softplus(self.sensitivity))
        probability_span = self.p_max - self.p_min
        unit_rates = self.p_min + probability_span * mapped
        final_rates = unit_rates * model_gate.to(
            unit_rates.device, unit_rates.dtype
        )

        self.last_raw_score = global_raw_score.detach()
        self.last_global_score = relative_score.detach()
        self.last_context_score = context_score.detach()
        self.last_context_ratio = context_ratio.detach()
        self.last_effective_context_ratio = effective_context_ratio.detach()
        self.last_unit_dropout_rates = unit_rates.detach()
        self.last_dropout_rates = final_rates.detach()
        self.last_period = context["period"].detach()
        self.last_support_fraction = context["support_fraction"].detach()
        self.last_dynamic_score = None
        self.last_stat_score = None

        if visualize:
            from basicts.modules.dropout_ts import visualize_noise_scoring

            visualize_noise_scoring(
                original=details["original"],
                reconstructed=details["reconstructed"],
                noise_scores=context_score,
                dropout_rates=final_rates,
                fft_magnitude_log1p=details.get("fft_magnitude_log1p"),
                mask=details.get("mask"),
                thresholds=details.get("threshold"),
                num_samples=max_samples,
            )
        if return_score:
            return context_score, final_rates
        return final_rates

