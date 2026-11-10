"""DropoutTS Callback: Task-Loss-Driven Sample-Adaptive Dropout

This callback integrates DropoutTS into the training pipeline, automatically
replacing standard dropout layers with sample-adaptive dropout and managing
the optimization of RRF parameters.
"""

import csv
import os
from typing import TYPE_CHECKING, List, Tuple, Any, Optional, Dict

import torch
from torch import nn

from basicts.modules import DropoutTS, SampleAdaptiveDropout, DropoutTSContext
from .callback import BasicTSCallback

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


def replace_dropout_layers(model: nn.Module) -> List[Tuple[Any, Any, nn.Dropout, SampleAdaptiveDropout]]:
    """Replace standard dropout layers with sample-adaptive dropout.

    Models without an explicit selection retain the historical behavior and
    replace every ``nn.Dropout`` layer.  A model can set
    ``saba_dropout_only = True`` and expose
    ``saba_dropout_layer_names`` to limit replacement to named SABA
    insertion points.  This keeps native stochastic components intact while
    making the intervention location explicit.
    
    Args:
        model: PyTorch model to process.
    
    Returns:
        List of tuples (parent, key, original_dropout, new_dropout) for each replacement.
    """
    replaced_layers = []
    target_names = None
    if getattr(model, "saba_dropout_only", False):
        target_names = set(getattr(model, "saba_dropout_layer_names", ()))
        if not target_names:
            raise ValueError(
                "saba_dropout_only models must declare saba_dropout_layer_names."
            )

        available = dict(model.named_modules())
        missing = sorted(name for name in target_names if name not in available)
        if missing:
            raise ValueError(
                "SABA dropout target(s) not found in model: " + ", ".join(missing)
            )
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    
    def _replace_recursive(module: nn.Module, parent_name: str = "") -> None:
        """Recursively traverse model and replace dropout layers."""
        for name, child in list(module.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            if isinstance(child, nn.Dropout) and (target_names is None or full_name in target_names):
                new_dropout = SampleAdaptiveDropout(p=child.p).to(device)
                setattr(module, name, new_dropout)
                replaced_layers.append((module, name, child, new_dropout))
            
            _replace_recursive(child, full_name)
    
    _replace_recursive(model)
    return replaced_layers


class DropoutTSCallback(BasicTSCallback):
    """Callback for sample-adaptive dropout with RRF optimization.
    
    This callback:
    1. Replaces standard dropout layers with SampleAdaptiveDropout
    2. Wraps model forward to compute adaptive dropout rates
    3. Adds RRF parameters to optimizer for end-to-end training
    4. Logs statistics and generates visualizations
    """
    
    def __init__(
        self,
        p_min: float = 0.1,
        p_max: float = 0.5,
        init_alpha: float = 10.0,
        init_sensitivity: float = 1.0,
        enable_statistics: bool = False,
        enable_visualization: bool = False,
        detrend_method: str = 'robust_ols',
        use_instance_norm: bool = True,
        use_sfm_anchor: bool = True,
        scorer_type: str = 'spectral',
        dual_k: int = 3,
        dual_top_ratio: float = 0.2,
        dual_lambda: float = 0.5,
        dual_lambda_learnable: bool = False,
        dual_lambda_eps: float = 0.3,
        dual_lambda_lr_scale: float = 0.1,
        stats_save_dir: Optional[str] = None
    ):
        """Initialize DropoutTSCallback.
        
        Args:
            p_min: Minimum dropout rate for clean samples.
            p_max: Maximum dropout rate for noisy samples.
            init_alpha: Initial value for RRF sigmoid sharpness parameter.
            init_sensitivity: Initial value for noise score sensitivity parameter.
            enable_statistics: If True, log detailed statistics each epoch.
            enable_visualization: If True, generate visualization plots each epoch.
            detrend_method: Detrending method for ablation study ('none', 'simple', 'robust_ols').
            use_instance_norm: Whether to use instance-level normalization for ablation study.
            use_sfm_anchor: Whether to use SFM as physical anchor for ablation study.
        """
        super().__init__()
        self.p_min = p_min
        self.p_max = p_max
        self.init_alpha = init_alpha
        self.init_sensitivity = init_sensitivity
        self.enable_statistics = enable_statistics
        self.enable_visualization = enable_visualization
        
        # Ablation study parameters
        self.detrend_method = detrend_method
        self.use_instance_norm = use_instance_norm
        self.use_sfm_anchor = use_sfm_anchor
        self.scorer_type = scorer_type
        self.dual_k = dual_k
        self.dual_top_ratio = dual_top_ratio
        self.dual_lambda = dual_lambda
        self.dual_lambda_learnable = dual_lambda_learnable
        self.dual_lambda_eps = dual_lambda_eps
        self.dual_lambda_lr_scale = dual_lambda_lr_scale
        self.stats_save_dir = stats_save_dir
        
        self.dropout_ts: Optional[DropoutTS] = None
        self.original_forward = None
        self.model_wrapped = False
        self.use_input_dropout_fallback = False
        self.replaced_layers: List[Tuple[Any, Any, nn.Dropout, SampleAdaptiveDropout]] = []
        self._last_batch_inputs: Optional[torch.Tensor] = None
        self._scorer_checked = False
        self._stats_written_epochs = set()
    
    def on_train_start(self, runner: "BasicTSRunner", **kwargs) -> None:
        """Initialize DropoutTS and wrap model for adaptive dropout."""
        # Log ablation study configuration
        ablation_info = (
            f"[Ablation Config] detrend_method: {self.detrend_method}, "
            f"use_instance_norm: {self.use_instance_norm}, "
            f"use_sfm_anchor: {self.use_sfm_anchor}, "
            f"scorer_type: {self.scorer_type}"
        )
        
        runner.logger.info(
            f"DropoutTS initialized - p_min: {self.p_min}, p_max: {self.p_max}, "
            f"init_alpha: {self.init_alpha}, init_sensitivity: {self.init_sensitivity}."
        )
        runner.logger.info(ablation_info)
        
        self.dropout_ts = DropoutTS(
            seq_len=None,
            p_min=self.p_min,
            p_max=self.p_max,
            init_alpha=self.init_alpha,
            init_sensitivity=self.init_sensitivity,
            detrend_method=self.detrend_method,
            use_instance_norm=self.use_instance_norm,
            use_sfm_anchor=self.use_sfm_anchor,
            scorer_type=self.scorer_type,
            dual_k=self.dual_k,
            dual_top_ratio=self.dual_top_ratio,
            dual_lambda=self.dual_lambda,
            dual_lambda_learnable=self.dual_lambda_learnable,
            dual_lambda_eps=self.dual_lambda_eps
        ).to(next(runner.model.parameters()).device)
        
        self._wrap_model(runner)
    
    def _wrap_model(self, runner: "BasicTSRunner") -> None:
        """Replace dropout layers and wrap model forward for adaptive dropout."""
        if self.model_wrapped:
            return
        
        model = runner.model.module if isinstance(runner.model, nn.parallel.DistributedDataParallel) else runner.model
        
        # Replace standard dropout layers
        self.replaced_layers = replace_dropout_layers(model)
        if self.replaced_layers:
            runner.logger.info(f"[DropoutTS] Replaced {len(self.replaced_layers)} dropout layer(s).")
        else:
            runner.logger.info("[DropoutTS] No dropout layers found.")
            self.use_input_dropout_fallback = True
            runner.logger.info("[DropoutTS] Using sample-adaptive input dropout fallback.")
        
        # Save original forward and create wrapped version
        self.original_forward = model.forward
        self._scorer_checked = False
        
        def wrapped_forward(*args, **kwargs):
            """Wrapped forward that computes and applies adaptive dropout rates."""
            inputs = args[0] if args else kwargs.get("inputs", None)
            dropout_rates = None
            
            if model.training and inputs is not None:
                # Store first batch for statistics/visualization
                if (self.enable_statistics or self.enable_visualization or self.stats_save_dir) and self._last_batch_inputs is None:
                    self._last_batch_inputs = inputs.detach().clone()
                
                # Compute adaptive dropout rates
                raw_score, dropout_rates = self.dropout_ts.compute_dropout_rates(inputs, return_score=True)
                self._write_dropout_stats_from_tensors(runner, raw_score, dropout_rates, split="train")
                DropoutTSContext.set_rates(dropout_rates)
                if self.use_input_dropout_fallback:
                    inputs = SampleAdaptiveDropout(p=self.p_max).to(inputs.device)(
                        inputs,
                        sample_dropout_rates=dropout_rates
                    )
                    if args:
                        args = (inputs,) + tuple(args[1:])
                    else:
                        kwargs["inputs"] = inputs
            
            # Check and add scorer parameters to optimizer (once per training).
            if not self._scorer_checked and dropout_rates is not None:
                self._setup_scorer_optimizer(runner, inputs)
                self._scorer_checked = True
            
            # Call original forward
            result = self.original_forward(*args, **kwargs)
            DropoutTSContext.clear()
            
            # Ensure result is in expected format
            return {"prediction": result} if isinstance(result, torch.Tensor) else result
        
        model.forward = wrapped_forward
        self.model_wrapped = True
        runner.logger.info("Model wrapped with DropoutTS.")
    
    def _setup_scorer_optimizer(self, runner: "BasicTSRunner", inputs: Optional[torch.Tensor]) -> None:
        """Add active scorer parameters to optimizer and log initial state."""
        if self.dropout_ts is None:
            return

        all_params = self.dropout_ts.scorer_parameters()
        
        # Check which parameters are not yet in optimizer
        optimizer_param_ids = {id(p) for group in runner.optimizer.param_groups for p in group['params']}
        params_to_add = [p for p in all_params if id(p) not in optimizer_param_ids]
        
        if params_to_add:
            runner.optimizer.add_param_group({
                'params': params_to_add,
                'lr': runner.optimizer.param_groups[0]['lr'] * (self.dual_lambda_lr_scale if self.dual_lambda_learnable else 1.0)
            })
            runner.logger.info(
                f"[DropoutTS] Added {len(params_to_add)} {self.scorer_type} scorer parameter tensor(s) to optimizer."
            )

        if self.scorer_type == 'dual_consistency':
            raw_score, dropout_rates = self.dropout_ts.compute_dropout_rates(inputs, return_score=True)
            dyn = self.dropout_ts.last_dynamic_score
            stat = self.dropout_ts.last_stat_score
            runner.logger.info(
                f"[DropoutTS] Initial dual-consistency: k={self.dual_k}, top_ratio={self.dual_top_ratio:.3f}, "
                f"lambda={float(self.dropout_ts.effective_dual_lambda().detach().cpu()):.3f}, p_mean={dropout_rates.mean().item():.4f}, "
                f"dyn_mean={dyn.mean().item():.4f}, stat_mean={stat.mean().item():.4f}"
            )

        if self.dropout_ts.noise_scorer is None:
            return

        rrf_filter = self.dropout_ts.noise_scorer.rrf_filter
        
        # Log initial parameter values
        alpha_val = rrf_filter.softplus(rrf_filter.alpha).item()
        sfm_scale_val = rrf_filter.sfm_scale.mean().item()
        sfm_bias_val = rrf_filter.sfm_bias.mean().item()
        sensitivity_val = torch.nn.functional.softplus(self.dropout_ts.sensitivity).item()
        
        if inputs is not None:
            # Get initial SFM score and dynamic threshold from first batch
            was_training = runner.model.training
            runner.model.eval()
            details = self.dropout_ts.noise_scorer(
                inputs, 
                return_details=True
            )
            sfm_score_val = details['sfm_score'].mean().item()
            dynamic_threshold_val = details['dynamic_threshold'].mean().item()
            if was_training:
                runner.model.train()
            
            runner.logger.info(
                f"[DropoutTS] Initial State: Alpha={alpha_val:.4f}, SFM_Scale={sfm_scale_val:.4f}, SFM_Bias={sfm_bias_val:.4f}, "
                f"SFM_Score={sfm_score_val:.4f}, Dynamic_Threshold={dynamic_threshold_val:.4f}, "
                f"Sensitivity={sensitivity_val:.4f}"
            )
        else:
            runner.logger.info(
                f"[DropoutTS] Initial State: Alpha={alpha_val:.4f}, SFM_Scale={sfm_scale_val:.4f}, SFM_Bias={sfm_bias_val:.4f}, "
                f"Sensitivity={sensitivity_val:.4f}"
            )
    
    def on_epoch_start(self, runner: "BasicTSRunner", **kwargs) -> None:
        """Reset batch inputs at start of each epoch."""
        self._last_batch_inputs = None
    
    def on_epoch_end(self, runner: "BasicTSRunner", **kwargs) -> None:
        """Log detailed statistics and generate visualizations."""
        if self.dropout_ts is None:
            return
        epoch = kwargs.get('epoch', runner.epoch if hasattr(runner, 'epoch') else 0)
        self._save_scorer_state(epoch=epoch, final=False)

        if self.scorer_type == 'dual_consistency':
            if self._last_batch_inputs is not None:
                was_training = self.dropout_ts.training
                self.dropout_ts.eval()
                raw_score, dropout_rates = self.dropout_ts.compute_dropout_rates(
                    self._last_batch_inputs,
                    return_score=True
                )
                dyn = self.dropout_ts.last_dynamic_score
                stat = self.dropout_ts.last_stat_score
                if was_training:
                    self.dropout_ts.train()
                runner.logger.info(
                    f"[DropoutTS] Epoch {epoch}: dual-consistency "
                    f"score_mean={raw_score.mean().item():.4f}, p_mean={dropout_rates.mean().item():.4f}, "
                    f"dyn_mean={dyn.mean().item():.4f}, stat_mean={stat.mean().item():.4f}"
                )
            return
        
        rrf = self.dropout_ts.noise_scorer.rrf_filter
        
        # Get current parameter values
        alpha_val = rrf.softplus(rrf.alpha).item()
        sfm_scale_val = rrf.sfm_scale.mean().item()
        sfm_bias_val = rrf.sfm_bias.mean().item()
        sensitivity_val = torch.nn.functional.softplus(self.dropout_ts.sensitivity).item()
        
        # Get SFM score and dynamic threshold from last batch
        if self._last_batch_inputs is not None:
            was_training = self.dropout_ts.training
            self.dropout_ts.eval()
            details = self.dropout_ts.noise_scorer(
                self._last_batch_inputs, 
                return_details=True
            )
            sfm_score_val = details['sfm_score'].mean().item()
            dynamic_threshold_val = details['dynamic_threshold'].mean().item()
            
            if was_training:
                self.dropout_ts.train()
            
            runner.logger.info(
                f"[DropoutTS] Epoch {epoch}: Alpha={alpha_val:.4f}, SFM_Scale={sfm_scale_val:.4f}, SFM_Bias={sfm_bias_val:.4f}, "
                f"SFM_Score={sfm_score_val:.4f}, Dynamic_Threshold={dynamic_threshold_val:.4f}, "
                f"Sensitivity={sensitivity_val:.4f}"
            )
        else:
            runner.logger.info(
                f"[DropoutTS] Epoch {epoch}: Alpha={alpha_val:.4f}, SFM_Scale={sfm_scale_val:.4f}, SFM_Bias={sfm_bias_val:.4f}, "
                f"Sensitivity={sensitivity_val:.4f}"
            )
        
        # Generate statistics and visualizations if enabled
        if self._last_batch_inputs is not None and (self.enable_statistics or self.enable_visualization):
            self._log_statistics_and_visualize(runner, epoch)
    
    def _log_statistics_and_visualize(self, runner: "BasicTSRunner", epoch: int) -> None:
        """Log detailed statistics and generate visualizations."""
        was_training = self.dropout_ts.training
        self.dropout_ts.eval()
        
        dropout_rates = self.dropout_ts.compute_dropout_rates(
            self._last_batch_inputs, 
            visualize=False
        )
        
        details = self.dropout_ts.noise_scorer(
            self._last_batch_inputs, 
            return_details=True
        )
        noise_scores = details['noise_scores']
        
        if was_training:
            self.dropout_ts.train()
        
        # Log statistics
        if self.enable_statistics:
            noise_mean = noise_scores.mean().item()
            noise_std = noise_scores.std().item()
            noise_min = noise_scores.min().item()
            noise_max = noise_scores.max().item()
            
            dropout_mean = dropout_rates.mean().item()
            dropout_std = dropout_rates.std().item()
            dropout_min = dropout_rates.min().item()
            dropout_max = dropout_rates.max().item()
            
            stat_msg = (
                f"[DropoutTS] Epoch {epoch} Statistics - "
                f"Noise (MAE): mean={noise_mean:.4f}, std={noise_std:.4f}, "
                f"min={noise_min:.4f}, max={noise_max:.4f} | "
                f"Dropout: mean={dropout_mean:.4f}, std={dropout_std:.4f}, "
                f"min={dropout_min:.4f}, max={dropout_max:.4f}"
            )
            runner.logger.info(stat_msg)
        
        # Generate visualization
        if self.enable_visualization:
            original_dir = os.getcwd()
            os.chdir(runner.ckpt_save_dir)
            self.dropout_ts.compute_dropout_rates(
                self._last_batch_inputs, 
                visualize=True, 
                max_samples=6
            )
            os.chdir(original_dir)
            runner.logger.info(
                f"[DropoutTS] Visualization saved to: "
                f"{os.path.join(runner.ckpt_save_dir, 'noise_scoring_visualization.png')}"
            )


    def on_optimizer_step(self, runner: "BasicTSRunner", **kwargs) -> None:
        if not self.dual_lambda_learnable or self.dropout_ts is None:
            return
        grad = self.dropout_ts.dual_lambda_theta.grad
        if grad is None:
            grad_value = float("nan")
        else:
            grad_value = float(grad.detach().abs().mean().cpu())
        runner.logger.info(
            f"[DropoutTS] dual-lambda step: lambda={float(self.dropout_ts.effective_dual_lambda().detach().cpu()):.6f}, "
            f"theta={float(self.dropout_ts.dual_lambda_theta.detach().cpu()):.6f}, theta_grad_abs={grad_value:.8f}"
        )

    def _write_dropout_stats(
        self,
        runner: "BasicTSRunner",
        epoch: int,
        inputs: torch.Tensor,
        split: str = "train"
    ) -> None:
        if self.dropout_ts is None:
            return
        was_training = self.dropout_ts.training
        self.dropout_ts.eval()
        with torch.no_grad():
            raw_score, dropout_rates = self.dropout_ts.compute_dropout_rates(inputs, return_score=True)
            q = torch.quantile(dropout_rates.float(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=dropout_rates.device))
            row = self._stats_row(epoch, split, raw_score, dropout_rates)
        if was_training:
            self.dropout_ts.train()

        save_dir = self.stats_save_dir or getattr(runner, "ckpt_save_dir", ".")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "dropout_rate_stats.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _save_scorer_state(self, epoch: int = None, final: bool = False) -> None:
        if self.dropout_ts is None or not self.stats_save_dir:
            return
        state_dir = os.path.join(self.stats_save_dir, "scorer_states")
        os.makedirs(state_dir, exist_ok=True)
        name = "scorer_final.pt" if final else f"scorer_epoch_{int(epoch):03d}.pt"
        torch.save({
            "epoch": epoch,
            "final": final,
            "scorer_type": self.scorer_type,
            "p_min": self.p_min,
            "p_max": self.p_max,
            "init_alpha": self.init_alpha,
            "init_sensitivity": self.init_sensitivity,
            "dual_k": self.dual_k,
            "dual_top_ratio": self.dual_top_ratio,
            "dual_lambda": float(self.dropout_ts.effective_dual_lambda().detach().cpu()) if self.dropout_ts is not None else self.dual_lambda,
            "dual_lambda_learnable": self.dual_lambda_learnable,
            "dual_lambda_eps": self.dual_lambda_eps,
            "dropout_ts_state_dict": self.dropout_ts.state_dict(),
        }, os.path.join(state_dir, name))

    def _stats_row(
        self,
        epoch: int,
        split: str,
        raw_score: torch.Tensor,
        dropout_rates: torch.Tensor
    ) -> Dict[str, float]:
        p = dropout_rates.detach().float()
        raw = raw_score.detach().float()
        q = torch.quantile(p, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=p.device))
        row = {
            "epoch": epoch,
            "split": split,
            "scorer_type": self.scorer_type,
            "p_mean": p.mean().item(),
            "p_std": p.std(unbiased=False).item(),
            "p_min": p.min().item(),
            "p_max": p.max().item(),
            "p_q05": q[0].item(),
            "p_q25": q[1].item(),
            "p_q50": q[2].item(),
            "p_q75": q[3].item(),
            "p_q95": q[4].item(),
            "raw_score_mean": raw.mean().item(),
            "raw_score_std": raw.std(unbiased=False).item(),
            "raw_score_min": raw.min().item(),
            "raw_score_max": raw.max().item(),
        }
        if self.dropout_ts is not None and self.dropout_ts.last_dynamic_score is not None:
            dyn = self.dropout_ts.last_dynamic_score.detach().float()
            stat = self.dropout_ts.last_stat_score.detach().float()
            row.update({
                "dual_k": float(self.dual_k),
                "dual_top_ratio": float(self.dual_top_ratio),
                "dual_lambda": float(self.dropout_ts.effective_dual_lambda().detach().cpu()) if self.dropout_ts is not None else float(self.dual_lambda),
                "dyn_score_mean": dyn.mean().item(),
                "dyn_score_std": dyn.std(unbiased=False).item(),
                "dyn_score_min": dyn.min().item(),
                "dyn_score_max": dyn.max().item(),
                "stat_score_mean": stat.mean().item(),
                "stat_score_std": stat.std(unbiased=False).item(),
                "stat_score_min": stat.min().item(),
                "stat_score_max": stat.max().item(),
            })
        return row

    def _write_dropout_stats_from_tensors(
        self,
        runner: "BasicTSRunner",
        raw_score: torch.Tensor,
        dropout_rates: torch.Tensor,
        split: str = "train"
    ) -> None:
        if not self.stats_save_dir:
            return
        epoch = getattr(runner, "epoch", len(self._stats_written_epochs) + 1)
        key = (epoch, split)
        if key in self._stats_written_epochs:
            return
        self._stats_written_epochs.add(key)

        with torch.no_grad():
            row = self._stats_row(epoch, split, raw_score, dropout_rates)
        os.makedirs(self.stats_save_dir, exist_ok=True)
        path = os.path.join(self.stats_save_dir, "dropout_rate_stats.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    
    def on_train_end(self, runner: "BasicTSRunner", **kwargs) -> None:
        """Restore model and log final state."""
        # Log final parameter values
        if self.dropout_ts is not None:
            self._save_scorer_state(epoch=getattr(runner, "epoch", None), final=True)
        if self.dropout_ts is not None and self.dropout_ts.noise_scorer is not None:
            rrf = self.dropout_ts.noise_scorer.rrf_filter
            alpha_val = rrf.softplus(rrf.alpha).item()
            sfm_scale_val = rrf.sfm_scale.mean().item()
            sfm_bias_val = rrf.sfm_bias.mean().item()
            sensitivity_val = torch.nn.functional.softplus(self.dropout_ts.sensitivity).item()
            
            if self._last_batch_inputs is not None:
                was_training = self.dropout_ts.training
                self.dropout_ts.eval()
                details = self.dropout_ts.noise_scorer(
                    self._last_batch_inputs, 
                    return_details=True
                )
                sfm_score_val = details['sfm_score'].mean().item()
                dynamic_threshold_val = details['dynamic_threshold'].mean().item()
                if was_training:
                    self.dropout_ts.train()
                
                runner.logger.info(
                    f"[DropoutTS] Finished. Final Params: Alpha={alpha_val:.4f}, "
                    f"SFM_Scale={sfm_scale_val:.4f}, SFM_Bias={sfm_bias_val:.4f}, SFM_Score={sfm_score_val:.4f}, "
                    f"Dynamic_Threshold={dynamic_threshold_val:.4f}, Sensitivity={sensitivity_val:.4f}"
                )
            else:
                runner.logger.info(
                    f"[DropoutTS] Finished. Final Params: Alpha={alpha_val:.4f}, "
                    f"SFM_Scale={sfm_scale_val:.4f}, SFM_Bias={sfm_bias_val:.4f}, Sensitivity={sensitivity_val:.4f}"
                )
        
        # Restore original dropout layers
        if self.replaced_layers:
            model = runner.model.module if isinstance(runner.model, nn.parallel.DistributedDataParallel) else runner.model
            for parent, key, original_dropout, _ in self.replaced_layers:
                if isinstance(key, int):
                    parent[key] = original_dropout
                else:
                    setattr(parent, key, original_dropout)
            self.replaced_layers = []
        
        # Restore original forward
        if self.model_wrapped and self.original_forward is not None:
            model = runner.model.module if isinstance(runner.model, nn.parallel.DistributedDataParallel) else runner.model
            model.forward = self.original_forward
            self.model_wrapped = False
            runner.logger.info("Restored original model forward.")
        
        DropoutTSContext.clear()
