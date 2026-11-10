"""Run an isolated SABA probability-path ablation.

This wrapper reuses the clean package runner and changes only the callback
class. It intentionally keeps the original command-line surface unchanged
except for `--ablation-variant`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
SABA_ROOT = PACKAGE_ROOT / "saba"
SABA_SRC = SABA_ROOT / "src"
RUNNER_PATH = SABA_ROOT / "scripts" / "latent" / "run.py"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SABA_SRC))

from abl import ABLATION_VARIANTS
from cb import AblationSABACallback


def load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "saba_clean_runner_for_ablation", RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load clean SABA runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--ablation-variant",
        choices=ABLATION_VARIANTS,
        default="full",
    )
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    runner = load_runner_module()
    variant = known.ablation_variant

    def callback_factory(*args, **kwargs):
        return AblationSABACallback(
            *args,
            ablation_variant=variant,
            **kwargs,
        )

    runner.SABACallback = callback_factory
    runner.main()

    # Keep the ablation identity next to the run because the clean runner
    # intentionally receives the original argument surface unchanged.
    def value_of(name: str, default=None):
        try:
            index = remaining.index(name)
        except ValueError:
            return default
        return remaining[index + 1] if index + 1 < len(remaining) else default

    dataset = value_of("--dataset", "Illness")
    model = value_of("--model", "TimesNet")
    method = value_of("--method", "saba")
    output_root = Path(value_of("--output-root", str(PACKAGE_ROOT / "outputs" / "latent")))
    output_len = int(value_of("--output-len", "24"))
    epochs = int(value_of("--epochs", "100"))
    run_name = value_of("--run-name", None) or f"h{output_len}_e{epochs}"
    run_dir = output_root / dataset / model / method / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ablation_variant.json").write_text(
        json.dumps({"variant": variant}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
