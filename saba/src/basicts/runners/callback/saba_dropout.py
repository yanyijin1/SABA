"""BasicTS callback for the finalized SABA method."""

from __future__ import annotations

from typing import Dict

import csv
import os

import torch

from basicts.modules.saba import SABA
from .dynamic_dropout import DropoutTSCallback


class SABACallback(DropoutTSCallback):
    def __init__(
        self,
        model_gate: float,
        p_min: float,
        p_max: float,
        response_ratio: float = 1.0,
        **kwargs,
    ):
        if not 0.0 <= float(p_min) < float(p_max) < 1.0:
            raise ValueError("Expected 0 <= p_min < p_max < 1.")
        kwargs["p_min"] = float(p_min)
        kwargs["p_max"] = float(p_max)
        kwargs.setdefault("init_alpha", 10.0)
        kwargs.setdefault("init_sensitivity", 1.0)
        kwargs.setdefault("scorer_type", "spectral")
        super().__init__(**kwargs)
        self.model_gate = float(model_gate)
        self.response_ratio = float(response_ratio)
        self._epoch_probability_sums = None

    def on_epoch_start(self, runner, **kwargs) -> None:
        super().on_epoch_start(runner, **kwargs)
        self._epoch_probability_sums = {
            "count": 0,
            "unit_p": 0.0,
            "final_p": 0.0,
            "context_ratio": 0.0,
            "effective_context_ratio": 0.0,
        }

    def _write_dropout_stats_from_tensors(
        self,
        runner,
        raw_score: torch.Tensor,
        dropout_rates: torch.Tensor,
        split: str = "train",
    ) -> None:
        super()._write_dropout_stats_from_tensors(runner, raw_score, dropout_rates, split)
        module = self.dropout_ts
        if split != "train" or module is None or module.last_unit_dropout_rates is None:
            return
        if self._epoch_probability_sums is None:
            self.on_epoch_start(runner)
        count = int(dropout_rates.numel())
        self._epoch_probability_sums["count"] += count
        self._epoch_probability_sums["unit_p"] += float(module.last_unit_dropout_rates.sum().item())
        self._epoch_probability_sums["final_p"] += float(dropout_rates.sum().item())
        self._epoch_probability_sums["context_ratio"] += float(module.last_context_ratio.sum().item())
        self._epoch_probability_sums["effective_context_ratio"] += float(
            module.last_effective_context_ratio.sum().item()
        )

    def on_epoch_end(self, runner, **kwargs) -> None:
        sums = self._epoch_probability_sums
        if self.stats_save_dir and sums is not None and sums["count"] > 0:
            count = sums["count"]
            epoch = kwargs.get("epoch", getattr(runner, "epoch", 0))
            row = {
                "epoch": epoch,
                "train_unit_p_mean": sums["unit_p"] / count,
                "train_final_p_mean": sums["final_p"] / count,
                "train_context_ratio_mean": sums["context_ratio"] / count,
                "train_effective_context_ratio_mean": sums["effective_context_ratio"] / count,
                "sensitivity": float(torch.nn.functional.softplus(self.dropout_ts.sensitivity).item()),
                "response_ratio": self.response_ratio,
                "context_strength": float(self.dropout_ts.context_strength.item()),
                "model_gate": self.model_gate,
            }
            os.makedirs(self.stats_save_dir, exist_ok=True)
            path = os.path.join(self.stats_save_dir, "saba_epoch_dynamics.csv")
            write_header = not os.path.exists(path)
            with open(path, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        super().on_epoch_end(runner, **kwargs)

    def on_train_start(self, runner, **kwargs) -> None:
        runner.logger.info(
            f"[SABA] Temporal Reliability + Backbone Gate; "
            f"response_ratio={self.response_ratio:.8f}, "
            f"model_gate={self.model_gate:.8f}, p_min={self.p_min:.6f}, "
            f"p_max={self.p_max:.6f}, init_sensitivity={self.init_sensitivity:.4f}."
        )
        self.dropout_ts = SABA(
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
        ).to(next(runner.model.parameters()).device)
        self._wrap_model(runner)

    def _stats_row(
        self,
        epoch: int,
        split: str,
        raw_score: torch.Tensor,
        dropout_rates: torch.Tensor,
    ) -> Dict[str, float]:
        row = super()._stats_row(epoch, split, raw_score, dropout_rates)
        row["model_gate"] = self.model_gate
        row["response_ratio"] = self.response_ratio
        row["context_strength"] = float(self.dropout_ts.context_strength.item())
        module = self.dropout_ts
        if module is not None and module.last_context_ratio is not None:
            row.update({
                "context_ratio_mean": module.last_context_ratio.float().mean().item(),
                "context_ratio_std": module.last_context_ratio.float().std(unbiased=False).item(),
                "effective_context_ratio_mean": module.last_effective_context_ratio.float().mean().item(),
                "effective_context_ratio_std": module.last_effective_context_ratio.float().std(unbiased=False).item(),
                "unit_p_mean_before_gate": module.last_unit_dropout_rates.float().mean().item(),
                "temporal_period_mean": module.last_period.float().mean().item(),
                "support_fraction_mean": module.last_support_fraction.float().mean().item(),
            })
        return row
