import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from basicts import BasicTSLauncher
from basicts.modules.dropout_ts import DropoutTS
from basicts.modules.saba import SABA
from basicts.runners import BasicTSRunner
from basicts.runners.builder import Builder
from basicts.utils import BasicTSMode, RunnerStatus
from basicts.configs import BasicTSForecastingConfig
from basicts.models.Autoformer import Autoformer, AutoformerConfig
from basicts.models.Crossformer import Crossformer, CrossformerConfig
from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.models.Informer import Informer, InformerConfig
from basicts.models.HaKAN import HaKANConfig, HaKANForForecasting
from basicts.models.NLinear import NLinear, NLinearConfig
from basicts.models.PatchTST import PatchTSTConfig, PatchTSTForForecasting
from basicts.models.SOFTS import SOFTSConfig, SOFTSForForecasting
from basicts.models.SEMixer import SEMixerConfig, SEMixerForForecasting
from basicts.models.TimesNet import TimesNetConfig, TimesNetForForecasting
from basicts.models.TQNet import TQNetConfig, TQNetForForecasting
from basicts.models.iTransformer import iTransformerConfig, iTransformerForForecasting
from basicts.runners.callback import DropoutTSCallback, EarlyStopping, SABACallback


MODEL_REGISTRY = {
    "Autoformer": (Autoformer, AutoformerConfig),
    "SOFTS": (SOFTSForForecasting, SOFTSConfig),
    "Crossformer": (Crossformer, CrossformerConfig),
    "TimesNet": (TimesNetForForecasting, TimesNetConfig),
    "iTransformer": (iTransformerForForecasting, iTransformerConfig),
    "PatchTST": (PatchTSTForForecasting, PatchTSTConfig),
    "DLinear": (DLinear, DLinearConfig),
    "Informer": (Informer, InformerConfig),
    "HaKAN": (HaKANForForecasting, HaKANConfig),
    "NLinear": (NLinear, NLinearConfig),
    "SEMixer": (SEMixerForForecasting, SEMixerConfig),
    "TQNet": (TQNetForForecasting, TQNetConfig),
}


def dataset_path_for(name: str) -> Path:
    candidates = [
        ROOT / "datasets" / name,
        ROOT.parent / "datasets" / name,
    ]
    for path in candidates:
        if (path / "train_data.npy").exists():
            return path
    raise FileNotFoundError(
        f"Cannot find prepared dataset {name!r} in: "
        + ", ".join(str(p) for p in candidates)
    )


def default_cycle_for(dataset_name: str) -> int:
    return 144 if dataset_name.lower() == "weather" else 24


def model_default(args: argparse.Namespace, key: str, fallback):
    value = getattr(args, key)
    return fallback if value is None else value


def resolve_saba_gate(args: argparse.Namespace) -> float:
    if args.saba_gate_value is not None:
        gate = float(args.saba_gate_value)
        if not 0.0 <= gate <= 1.0:
            raise ValueError("--saba-gate-value must be in [0, 1].")
        return gate
    if args.saba_gate_file is None:
        return 1.0

    gate_path = Path(args.saba_gate_file)
    if not gate_path.exists():
        raise FileNotFoundError(f"SABA gate file not found: {gate_path}")

    import csv

    chosen = None
    with gate_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == args.dataset and row["model"] == args.model:
                chosen = row
                break
    if chosen is None:
        raise KeyError(f"No SABA gate for dataset={args.dataset}, model={args.model} in {gate_path}")

    gate = float(chosen["gate"])
    if not 0.0 <= gate <= 1.0:
        raise ValueError(f"Invalid gate={gate} in {gate_path}")
    return gate


def resolve_saba_response_ratio(args: argparse.Namespace) -> float:
    if args.saba_response_ratio_value is not None:
        ratio = float(args.saba_response_ratio_value)
        if ratio < 0.0:
            raise ValueError("--saba-response-ratio-value must be nonnegative.")
        return ratio
    if args.saba_gate_file is None:
        return 1.0

    gate_path = Path(args.saba_gate_file)
    if not gate_path.exists():
        raise FileNotFoundError(f"SABA gate file not found: {gate_path}")

    import csv

    with gate_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == args.dataset and row["model"] == args.model:
                ratio = float(row["response_ratio"])
                if ratio < 0.0:
                    raise ValueError(f"Invalid response_ratio={ratio} in {gate_path}")
                return ratio
    raise KeyError(f"No SABA response ratio for dataset={args.dataset}, model={args.model} in {gate_path}")



def _find_best_checkpoint(cfg: BasicTSForecastingConfig) -> Path:
    runner = BasicTSRunner(cfg)
    return Path(runner.ckpt_save_dir) / f"{runner.model_name}_best_val_{runner.target_metric.replace('/', '_')}.pt"


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _compute_per_sample_errors(pred: np.ndarray, true: np.ndarray):
    err = pred - true
    abs_err = np.abs(err)
    sq_err = err * err
    mae_per_horizon = abs_err.mean(axis=2)
    mse_per_horizon = sq_err.mean(axis=2)
    mae_per_sample = abs_err.mean(axis=(1, 2))
    mse_per_sample = sq_err.mean(axis=(1, 2))
    return mae_per_sample, mse_per_sample, mae_per_horizon, mse_per_horizon


def _load_scorer_state_if_available(
    dropout_ts: DropoutTS,
    out_dir: Path,
    example_inputs: torch.Tensor,
) -> None:
    state_dir = out_dir / "scorer_states"
    state_path = state_dir / "scorer_final.pt"
    if not state_path.exists():
        candidates = sorted(state_dir.glob("scorer_epoch_*.pt")) if state_dir.exists() else []
        state_path = candidates[-1] if candidates else None
    if state_path is None or not state_path.exists():
        return
    state = torch.load(state_path, map_location="cpu")
    if isinstance(state, dict) and "dropout_ts_state_dict" in state:
        state = state["dropout_ts_state_dict"]
    if any(key.startswith("noise_scorer.") for key in state):
        dropout_ts._ensure_noise_scorer(example_inputs)
    dropout_ts.load_state_dict(state, strict=True)


def _build_score_module(args: argparse.Namespace, out_dir: Path, device: torch.device):
    if args.method == "raw":
        return None
    if args.method == "saba":
        module = SABA(
            p_min=args.p_min,
            p_max=args.p_max,
            init_alpha=10.0,
            init_sensitivity=args.init_sensitivity,
            model_gate=resolve_saba_gate(args),
            response_ratio=resolve_saba_response_ratio(args),
        ).to(device)
    else:
        module = DropoutTS(
            p_min=args.p_min,
            p_max=args.p_max,
            init_alpha=10.0,
            init_sensitivity=args.init_sensitivity,
            scorer_type="spectral",
        ).to(device)
    example_inputs = torch.zeros(
        (1, args.input_len, args.num_features),
        dtype=torch.float32,
        device=device,
    )
    _load_scorer_state_if_available(module, out_dir, example_inputs)
    module.eval()
    return module


@torch.no_grad()
def _export_split_outputs(
    runner: BasicTSRunner,
    data_loader,
    split: str,
    export_dir: Path,
    scorer: DropoutTS = None,
    scores_only: bool = False,
) -> None:
    runner.model.eval()
    runner.status = RunnerStatus.EVALUATING

    sample_ids, inputs, preds, trues = [], [], [], []
    raw_scores, p_drops, dyn_scores, stat_scores = [], [], [], []
    context_ratios, effective_context_ratios = [], []
    unit_p_drops, temporal_periods, support_fractions = [], [], []

    for step, data in enumerate(data_loader):
        batch_sample_id = data.get("sample_id")
        data = runner.taskflow.preprocess(runner, data)
        if not scores_only:
            inputs.append(_as_numpy(data["inputs"]))

        if scorer is not None:
            scorer = scorer.to(data["inputs"].device)
            score, p_drop = scorer.compute_dropout_rates(data["inputs"], return_score=True)
            raw_scores.append(_as_numpy(score).reshape(-1))
            p_drops.append(_as_numpy(p_drop).reshape(-1))
            dyn = scorer.last_dynamic_score
            stat = scorer.last_stat_score
            n = score.numel()
            dyn_scores.append(_as_numpy(dyn).reshape(-1) if dyn is not None else np.full((n,), np.nan, dtype=np.float32))
            stat_scores.append(_as_numpy(stat).reshape(-1) if stat is not None else np.full((n,), np.nan, dtype=np.float32))
            context_ratio = getattr(scorer, "last_context_ratio", None)
            effective_context_ratio = getattr(scorer, "last_effective_context_ratio", None)
            unit_p = getattr(scorer, "last_unit_dropout_rates", None)
            period = getattr(scorer, "last_period", None)
            support = getattr(scorer, "last_support_fraction", None)
            context_ratios.append(_as_numpy(context_ratio).reshape(-1) if context_ratio is not None else np.full((n,), np.nan, dtype=np.float32))
            effective_context_ratios.append(_as_numpy(effective_context_ratio).reshape(-1) if effective_context_ratio is not None else np.full((n,), np.nan, dtype=np.float32))
            unit_p_drops.append(_as_numpy(unit_p).reshape(-1) if unit_p is not None else np.full((n,), np.nan, dtype=np.float32))
            temporal_periods.append(_as_numpy(period).reshape(-1) if period is not None else np.full((n,), np.nan, dtype=np.float32))
            support_fractions.append(_as_numpy(support).reshape(-1) if support is not None else np.full((n,), np.nan, dtype=np.float32))

        sample_ids.append(_as_numpy(batch_sample_id).reshape(-1))
        if not scores_only:
            forward_return = runner._forward(runner.model, data, step=step)
            forward_return = runner.taskflow.postprocess(runner, forward_return)
            preds.append(_as_numpy(forward_return["prediction"]))
            trues.append(_as_numpy(forward_return["targets"]))

    sample_id = np.concatenate(sample_ids, axis=0)
    export_dir.mkdir(parents=True, exist_ok=True)
    if not scores_only:
        pred = np.concatenate(preds, axis=0)
        true = np.concatenate(trues, axis=0)
        mae_sample, mse_sample, mae_horizon, mse_horizon = _compute_per_sample_errors(pred, true)
        np.savez_compressed(
            export_dir / f"{split}.npz",
            sample_id=sample_id,
            inputs=np.concatenate(inputs, axis=0),
            pred=pred,
            true=true,
            mae_per_sample=mae_sample,
            mse_per_sample=mse_sample,
            mae_per_horizon=mae_horizon,
            mse_per_horizon=mse_horizon,
        )

    if scorer is not None:
        np.savez_compressed(
            export_dir / f"{split}_scores.npz",
            sample_id=sample_id,
            raw_score=np.concatenate(raw_scores, axis=0),
            p_drop=np.concatenate(p_drops, axis=0),
            s_dyn=np.concatenate(dyn_scores, axis=0),
            s_stat=np.concatenate(stat_scores, axis=0),
            context_ratio=np.concatenate(context_ratios, axis=0),
            effective_context_ratio=np.concatenate(effective_context_ratios, axis=0),
            unit_p_before_gate=np.concatenate(unit_p_drops, axis=0),
            temporal_period=np.concatenate(temporal_periods, axis=0),
            support_fraction=np.concatenate(support_fractions, axis=0),
        )


def export_stage1_outputs(cfg: BasicTSForecastingConfig, args: argparse.Namespace, out_dir: Path) -> None:
    best_ckpt = _find_best_checkpoint(cfg)
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")

    export_batch_size = args.stage1_batch_size
    if export_batch_size is None and args.model == "Informer":
        export_batch_size = 1
    if export_batch_size is not None:
        if export_batch_size < 1:
            raise ValueError("--stage1-batch-size must be at least 1")
        cfg.train_batch_size = export_batch_size
        cfg.val_batch_size = export_batch_size
        cfg.test_batch_size = export_batch_size

    runner = BasicTSRunner(cfg)
    runner.init_logger(logger_name="BasicTS-stage1-export", log_file_name="stage1_export_log")
    runner.on_eval_start(str(best_ckpt))
    runner._init_validation()
    runner._init_test()
    runner.train_data_loader = Builder._build_data_loader(runner.cfg, BasicTSMode.TRAIN, runner.logger)

    device = next(runner.model.parameters()).device
    scorer = _build_score_module(args, out_dir, device)
    export_dir = Path(args.stage1_export_dir) if args.stage1_export_dir else out_dir / "stage1_outputs"

    split_map = {
        "train": runner.train_data_loader,
        "val": runner.val_data_loader,
        "test": runner.test_data_loader,
    }
    splits = [s.strip() for s in args.stage1_splits.split(",") if s.strip()]
    for split in splits:
        if split not in split_map:
            raise ValueError(f"Unknown split {split!r}; expected one of train,val,test")
        _export_split_outputs(
            runner,
            split_map[split],
            split,
            export_dir,
            scorer,
            scores_only=args.stage1_scores_only,
        )

    meta = {
        "model": args.model,
        "method": args.method,
        "dataset": args.dataset,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_features": args.num_features,
        "seed": args.seed,
        "init_sensitivity": args.init_sensitivity,
        "response_ratio": resolve_saba_response_ratio(args) if args.method == "saba" else None,
        "method": args.method,
        "best_checkpoint": str(best_ckpt),
        "export_batch_size": export_batch_size or args.batch_size,
        "splits": splits,
    }
    (export_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def build_model_config(args: argparse.Namespace):
    _, config_cls = MODEL_REGISTRY[args.model]
    config_kwargs = {
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_features": args.num_features,
    }

    if args.model == "NLinear":
        config_kwargs.pop("num_features")
    elif args.model == "TimesNet":
        timestamp_sizes = [int(x) for x in args.timestamp_sizes.split(",") if x]
        config_kwargs.update(use_timestamps=True, timestamp_sizes=timestamp_sizes)
    elif args.model == "Autoformer":
        timestamp_sizes = [int(x) for x in args.timestamp_sizes.split(",") if x]
        config_kwargs.update(
            label_len=max(1, args.input_len // 2),
            hidden_size=model_default(args, "hidden_size", 128),
            n_heads=args.n_heads,
            intermediate_size=args.intermediate_size,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=1,
            dropout=model_default(args, "dropout", 0.1),
            use_timestamps=True,
            timestamp_sizes=timestamp_sizes,
        )
    elif args.model == "Informer":
        config_kwargs.update(
            hidden_size=model_default(args, "hidden_size", 128),
            n_heads=args.n_heads,
            intermediate_size=args.intermediate_size,
            num_encoder_layers=args.num_layers,
            dropout=model_default(args, "dropout", 0.1),
        )
    elif args.model == "PatchTST":
        config_kwargs.update(
            patch_len=min(16, args.input_len),
            patch_stride=max(1, min(8, args.input_len // 2)),
        )
    elif args.model == "Crossformer":
        config_kwargs.update(
            patch_len=min(8, args.input_len),
            hidden_size=model_default(args, "hidden_size", 128),
            intermediate_size=args.intermediate_size,
            n_heads=args.n_heads,
            num_layers=args.num_layers,
            dropout=model_default(args, "dropout", 0.1),
        )
    elif args.model == "SOFTS":
        config_kwargs.update(
            hidden_size=model_default(args, "hidden_size", 128),
            d_core=args.d_core,
            intermediate_size=args.intermediate_size,
            num_layers=args.num_layers,
            dropout=model_default(args, "dropout", 0.1),
        )
    elif args.model == "TQNet":
        config_kwargs.update(
            cycle=args.cycle if args.cycle is not None else default_cycle_for(args.dataset),
            hidden_size=model_default(args, "hidden_size", 512),
            dropout=model_default(args, "dropout", 0.5),
            use_revin=model_default(args, "use_revin", 1),
            use_tq=not args.no_tq,
            channel_aggre=not args.no_channel_aggre,
            channel_aggre_heads=args.channel_aggre_heads,
            channel_aggre_dropout=args.channel_aggre_dropout,
        )
    elif args.model == "HaKAN":
        config_kwargs.update(
            patch_len=min(args.hakan_patch_len, args.input_len),
            stride=max(1, min(args.hakan_stride, args.input_len)),
            hidden_size=model_default(args, "hidden_size", 64),
            num_blocks=args.hakan_num_blocks,
            padding_patch="end",
            use_revin=True,
            affine=True,
            subtract_last=False,
            # The callback replaces this explicit target with sample-adaptive
            # dropout.  Raw training keeps the official model deterministic.
            dropout=0.0,
        )
    elif args.model == "SEMixer":
        config_kwargs.update(
            hidden_size=model_default(args, "hidden_size", 128),
            patch_len=min(args.semixer_patch_len or max(1, args.input_len // 16), args.input_len),
            stride=max(1, min(args.semixer_stride or max(1, args.input_len // 16), args.input_len)),
            reduce_dim=args.semixer_reduce_dim,
            eib_num=args.semixer_eib_num,
            eib_num_1scale=args.semixer_base_eib_num,
            head_dropout=0.0,
            connection_probability=args.semixer_connection_probability,
            random_attention=not args.no_semixer_random_attention,
            # Native mixer and RAM randomness are preserved; only this target
            # is replaced by SABA through saba_dropout_only.
            saba_dropout=0.0,
        )
    return config_cls(**config_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Illness")
    parser.add_argument("--data-file-path", default=None)
    parser.add_argument("--cycle", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--d-core", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--use-revin", type=int, default=None)
    parser.add_argument("--no-tq", action="store_true")
    parser.add_argument("--no-channel-aggre", action="store_true")
    parser.add_argument("--channel-aggre-heads", type=int, default=4)
    parser.add_argument("--channel-aggre-dropout", type=float, default=0.5)
    parser.add_argument("--hakan-patch-len", type=int, default=16)
    parser.add_argument("--hakan-stride", type=int, default=8)
    parser.add_argument("--hakan-num-blocks", type=int, default=5)
    parser.add_argument("--semixer-patch-len", type=int, default=None)
    parser.add_argument("--semixer-stride", type=int, default=None)
    parser.add_argument("--semixer-reduce-dim", type=int, default=64)
    parser.add_argument("--semixer-eib-num", type=int, default=1)
    parser.add_argument("--semixer-base-eib-num", type=int, default=1)
    parser.add_argument("--semixer-connection-probability", type=float, default=0.85)
    parser.add_argument("--no-semixer-random-attention", action="store_true")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="TimesNet")
    parser.add_argument("--method", choices=["raw", "spectral", "saba"], default="saba")
    parser.add_argument("--input-len", type=int, default=24)
    parser.add_argument("--output-len", type=int, default=24)
    parser.add_argument("--num-features", type=int, default=7)
    parser.add_argument("--timestamp-sizes", default="60,7,31,366")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--init-sensitivity", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default=str(ROOT.parent / "outputs" / "latent"))
    parser.add_argument("--p-min", type=float, default=0.05)
    parser.add_argument("--p-max", type=float, default=0.5)
    parser.add_argument("--saba-gate-file", default=None)
    parser.add_argument("--saba-gate-value", type=float, default=None)
    parser.add_argument("--saba-response-ratio-value", type=float, default=None)
    parser.add_argument("--skip-epoch-test", action="store_true")
    parser.add_argument("--save-stage1-outputs", action="store_true")
    parser.add_argument("--export-stage1-only", action="store_true")
    parser.add_argument("--stage1-scores-only", action="store_true")
    parser.add_argument("--stage1-splits", default="train,val")
    parser.add_argument("--stage1-export-dir", default=None)
    parser.add_argument("--stage1-batch-size", type=int, default=None)
    args = parser.parse_args()
    if not 0.0 <= args.p_min < args.p_max < 1.0:
        raise ValueError("Expected 0 <= --p-min < --p-max < 1.")

    data_path = Path(args.data_file_path) if args.data_file_path else dataset_path_for(args.dataset)
    run_name = args.run_name or f"h{args.output_len}_e{args.epochs}"
    out_dir = Path(args.output_root) / args.dataset / args.model / args.method / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    command_name = "export_command.txt" if args.export_stage1_only else "command.txt"
    (out_dir / command_name).write_text(" ".join(sys.argv) + "\n")

    callbacks = [EarlyStopping(patience=args.patience)]
    if args.method == "saba":
        callbacks.insert(0, SABACallback(
            model_gate=resolve_saba_gate(args),
            response_ratio=resolve_saba_response_ratio(args),
            p_min=args.p_min,
            p_max=args.p_max,
            init_alpha=10.0,
            init_sensitivity=args.init_sensitivity,
            enable_statistics=False,
            enable_visualization=False,
            stats_save_dir=str(out_dir),
        ))
    elif args.method == "spectral":
        callbacks.insert(0, DropoutTSCallback(
            p_min=args.p_min,
            p_max=args.p_max,
            init_alpha=10.0,
            init_sensitivity=args.init_sensitivity,
            enable_statistics=False,
            enable_visualization=False,
            scorer_type="spectral",
            stats_save_dir=str(out_dir),
        ))

    model_cls, _ = MODEL_REGISTRY[args.model]
    use_timestamps = args.model in {"Autoformer", "TimesNet", "TQNet"}

    cfg = BasicTSForecastingConfig(
        model=model_cls,
        model_config=build_model_config(args),
        dataset_name=args.dataset,
        dataset_params={"data_file_path": str(data_path)},
        input_len=args.input_len,
        output_len=args.output_len,
        use_timestamps=use_timestamps,
        gpus=args.gpu,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        loss="MAE",
        metrics=["MAE", "MSE", "RMSE"],
        target_metric="MAE",
        lr=args.learning_rate,
        optimizer_params={"lr": args.learning_rate, "weight_decay": args.weight_decay},
        callbacks=callbacks,
        seed=args.seed,
        ckpt_save_dir=str(out_dir / "checkpoints"),
        train_data_num_workers=0,
        val_data_num_workers=0,
        test_data_num_workers=0,
        test_interval=args.epochs + 1 if args.skip_epoch_test else 1,
        train_data_pin_memory=False,
        val_data_pin_memory=False,
        test_data_pin_memory=False,
    )
    if args.export_stage1_only:
        export_stage1_outputs(cfg, args, out_dir)
        return
    BasicTSLauncher.launch_training(cfg)
    if args.save_stage1_outputs:
        export_stage1_outputs(cfg, args, out_dir)


if __name__ == "__main__":
    main()
