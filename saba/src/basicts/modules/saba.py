"""SABA: sample-adaptive and backbone-aware dropout.

Implements sample-wise temporal reliability estimation and sample-backbone
dropout allocation from the paper. Backbone sensitivity enters through a
precomputed response ratio and gate.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from .dropout_ts import DropoutTS, visualize_noise_scoring


def backbone_gate(response_ratio: Union[float, torch.Tensor], order: int = 2):
    """One-sided backbone gate: preserve the safe side and rapidly suppress the low side."""
    if order < 1:
        raise ValueError("order must be >= 1")
    if not torch.is_tensor(response_ratio):
        r = torch.tensor(float(response_ratio), dtype=torch.float32)
    else:
        r = response_ratio
    safe_r = r.clamp_min(torch.finfo(r.dtype).eps)
    low_value = torch.exp(1.0 - safe_r.pow(-order))
    out = torch.where(r >= 1.0, torch.ones_like(r), low_value)
    return torch.where(r <= 0.0, torch.zeros_like(out), out)


class TemporalReliability(nn.Module):
    """Sample-wise temporal reliability estimation.

    Spectral reconstruction gives global deviation. Recurring interval support
    keeps periodic residuals from being treated as unreliable.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def _detrend(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        t = torch.arange(length, device=x.device, dtype=x.dtype)
        tc = t - t.mean()
        mean = x.mean(dim=-1, keepdim=True)
        slope = ((x - mean) * tc.view(1, 1, -1)).sum(dim=-1, keepdim=True)
        slope = slope / tc.square().sum().clamp_min(self.eps)
        trend = mean + slope * tc.view(1, 1, -1)
        return x - trend

    def _estimate_period(self, detrended: torch.Tensor) -> torch.Tensor:
        batch_size, _, length = detrended.shape
        if length < 3:
            return torch.ones(batch_size, device=detrended.device, dtype=torch.long)
        amplitude = torch.fft.rfft(detrended, dim=-1).abs()
        amplitude = amplitude.clone()
        amplitude[..., 0] = 0.0
        normalized = amplitude / amplitude.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        aggregate = normalized.mean(dim=1)
        if aggregate.shape[-1] <= 1:
            return torch.ones(batch_size, device=detrended.device, dtype=torch.long)
        frequency_index = aggregate[:, 1:].argmax(dim=-1) + 1
        period = torch.round(length / frequency_index.to(detrended.dtype)).long()
        return period.clamp(min=1, max=max(1, length // 2)).detach()

    def _period_support(self, residual: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = residual.shape
        base = torch.arange(length, device=residual.device).view(1, -1)
        left = base - period.view(-1, 1)
        right = base + period.view(-1, 1)
        left_valid = (left >= 0).to(residual.dtype)
        right_valid = (right < length).to(residual.dtype)
        left_index = left.clamp(0, length - 1).view(batch_size, 1, length).expand(-1, channels, -1)
        right_index = right.clamp(0, length - 1).view(batch_size, 1, length).expand(-1, channels, -1)
        left_value = torch.gather(residual, dim=2, index=left_index)
        right_value = torch.gather(residual, dim=2, index=right_index)
        count = (left_valid + right_valid).view(batch_size, 1, length)
        support = (
            left_value * left_valid.view(batch_size, 1, length)
            + right_value * right_valid.view(batch_size, 1, length)
        ) / count.clamp_min(1.0)
        return torch.where(count > 0, support, residual)

    def forward(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, torch.Tensor]:
        if original.dim() != 3 or reconstructed.shape != original.shape:
            raise ValueError("original and reconstructed must have matching [B, C, T] shapes")
        detrended = self._detrend(original)
        period = self._estimate_period(detrended)
        channel_scale = detrended.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(self.eps)
        residual = (original - reconstructed).abs() / channel_scale
        support = self._period_support(residual, period)
        unsupported = F.relu(residual - support)
        global_score = residual.mean(dim=(1, 2))
        unsupported_score = unsupported.mean(dim=(1, 2))
        unsupported_ratio = (unsupported_score / global_score.clamp_min(self.eps)).clamp(0.0, 1.0)
        support_fraction = (
            torch.minimum(residual, support).sum(dim=(1, 2))
            / residual.sum(dim=(1, 2)).clamp_min(self.eps)
        ).clamp(0.0, 1.0)
        return {
            "period": period,
            "period_ratio": period.to(original.dtype) / float(original.shape[-1]),
            "global_residual": global_score,
            "unsupported_residual": unsupported_score,
            "unsupported_ratio": unsupported_ratio,
            "support_fraction": support_fraction,
        }


class SABA(DropoutTS):
    """Sample-backbone dropout allocation.

    Combines sample reliability with the fixed backbone response gate.
    The probability formula is unchanged from the deployed implementation.
    """

    def __init__(
        self,
        *args,
        model_gate: float = 1.0,
        response_ratio: float = 1.0,
        context_eps: float = 1e-6,
        **kwargs,
    ):
        kwargs.setdefault("init_sensitivity", 1.0)
        kwargs.setdefault("scorer_type", "spectral")
        super().__init__(*args, **kwargs)
        if not 0.0 <= float(self.p_min) < float(self.p_max) < 1.0:
            raise ValueError("Expected 0 <= p_min < p_max < 1.")
        if self.scorer_type != "spectral":
            raise ValueError("SABA initial version requires scorer_type='spectral'.")
        if not 0.0 <= float(model_gate) <= 1.0:
            raise ValueError("model_gate must be in [0, 1].")
        if float(response_ratio) < 0.0:
            raise ValueError("response_ratio must be nonnegative.")
        self.temporal_reliability = TemporalReliability(eps=context_eps)
        self.register_buffer(
            "response_ratio",
            torch.tensor(float(response_ratio), dtype=torch.float32),
            persistent=False,
        )
        context_strength = min(1.0, 1.0 / float(response_ratio)) if response_ratio > 0 else 1.0
        self.register_buffer(
            "context_strength",
            torch.tensor(context_strength, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("model_gate", torch.tensor(float(model_gate), dtype=torch.float32))
        self.last_global_score = None
        self.last_context_score = None
        self.last_context_ratio = None
        self.last_effective_context_ratio = None
        self.last_unit_dropout_rates = None
        self.last_period = None
        self.last_support_fraction = None

    def _ensure_noise_scorer(self, inputs: torch.Tensor) -> None:
        super()._ensure_noise_scorer(inputs)
        pending = getattr(self, "_pending_scorer_state", None)
        if pending is not None:
            self.load_state_dict(pending, strict=False)
            self._pending_scorer_state = None

    def set_model_gate(self, value: float) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("model gate must be in [0, 1].")
        self.model_gate.fill_(float(value))

    @staticmethod
    def _batch_minmax(values: torch.Tensor) -> torch.Tensor:
        minimum = values.min()
        maximum = values.max()
        span = maximum - minimum
        if bool(span > 1e-6):
            return (values - minimum) / span
        return torch.full_like(values, 0.5)

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
        context = self.temporal_reliability(details["original"], details["reconstructed"])
        context_ratio = context["unsupported_ratio"]
        strength = self.context_strength.to(context_ratio.device, context_ratio.dtype)
        effective_context_ratio = 1.0 - strength * (1.0 - context_ratio)
        context_score = relative_score * effective_context_ratio
        mapped = torch.tanh(context_score * F.softplus(self.sensitivity))
        probability_span = self.p_max - self.p_min
        unit_rates = self.p_min + probability_span * mapped
        final_rates = unit_rates * self.model_gate.to(unit_rates.device, unit_rates.dtype)

        self.last_raw_score = global_raw_score.detach()
        self.last_global_score = relative_score.detach()
        self.last_context_score = context_score.detach()
        self.last_context_ratio = context["unsupported_ratio"].detach()
        self.last_effective_context_ratio = effective_context_ratio.detach()
        self.last_unit_dropout_rates = unit_rates.detach()
        self.last_dropout_rates = final_rates.detach()
        self.last_period = context["period"].detach()
        self.last_support_fraction = context["support_fraction"].detach()
        self.last_dynamic_score = None
        self.last_stat_score = None

        if visualize:
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
