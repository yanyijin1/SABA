# SABA

**Sample-Adaptive and Backbone-Aware Dropout for Time Series Forecasting**

SABA allocates dropout probabilities using the temporal reliability of each
input sample and the sensitivity of the forecasting backbone. This repository
provides the method implementation and training code for 12 backbones on four
datasets.

## Overview

The implementation combines three components:

1. **Global deviation:** a learnable spectral soft mask reconstructs the input,
   and the reconstruction residual measures sample deviation.
2. **Recurring support:** residuals at corresponding cycle positions distinguish
   recurring variations from unsupported deviations.
3. **Backbone modulation:** a calibrated response ratio conditions the recurring
   signal, and a backbone gate scales the final dropout probability.

The spectral scorer and dropout sensitivity are optimized through forecasting
training. Backbone response ratios and gates are supplied as calibration inputs.

## Installation

The code has been checked with **Python 3.10** and **PyTorch 2.2.0** on Linux.
Install a PyTorch build appropriate for your CUDA environment first. From the
repository root, install the remaining dependencies and the local package:

```bash
python -m pip install -r saba/requirements.txt
python -m pip install -e ./saba --no-deps
```

Check the available options without starting training:

```bash
python prep.py --help
python saba/scripts/latent/run.py --help
python ablation/run.py --help
```

All commands below assume that the current directory is the repository root.
For an extracted supplementary archive, use its `SABA/` directory.

## Datasets

| Dataset | Input length L | Prediction horizons H | Channels | Timestamp sizes |
| :--- | ---: | :--- | ---: | :--- |
| Illness | 24 | 24, 36, 48 | 7 | `60,7,31,366` |
| TTCAEnergy | 36 | 24, 36, 48 | 9 | `53,12,31` |
| CarbonMonitor | 96 | 24, 36, 48 | 6 | `7,31,366` |
| ExchangeRate | 96 | 96, 192, 336 | 8 | `60,7,31,366` |

The full configuration matrix is in
[`configs/experiment_matrix.json`](configs/experiment_matrix.json).
It records configurations; the training entry point receives them as command-line arguments.

### Prepare the original files

Obtain the datasets separately and use this directory structure:

```text
data/
├── Illness/Illness.csv
├── TTCAEnergy/Energy.csv
├── CarbonMonitor/carbonmonitor_global.csv
└── ExchangeRate/ExchangeRate.csv
```

The loaders in [`prep.py`](prep.py) define the required schemas:

- **Illness:** a `date` column (`%Y-%m-%d %H:%M:%S`) followed by seven numeric channels.
- **TTCAEnergy:** a `date` column, `OT`, and columns beginning with `Weekly `;
  the channel order is `OT` followed by the weekly columns in file order.
- **CarbonMonitor:** `date`, `sector`, and `value` columns. Preparation aggregates
  by date and sector, sorts sector columns, and fills gaps on the daily grid
  using the existing interpolation procedure.
- **ExchangeRate:** a `date` column (`%Y/%m/%d %H:%M`) followed by eight numeric channels.

Upstream sources recorded by the preparation code:
[Illness](https://raw.githubusercontent.com/scalation/data/master/Influenza/national_illness.csv),
[TTCAEnergy](https://raw.githubusercontent.com/synapsespectrum/TTCA/main/dataset/Energy.csv),
and [CarbonMonitor](https://datas.carbonmonitor.org/API/downloadFullDataset.php?source=carbon_global).
These links identify sources; files must match the schemas above before use.

Prepare one dataset:

```bash
python prep.py --dataset TTCAEnergy
```

Or prepare all four with `python prep.py`. Prepared arrays and metadata are
written to `datasets/<dataset>/`. Custom locations are supported through
`--raw-root` and `--out-root`.

Preparation uses chronological training, validation, and test partitions with
approximately 70% / 10% / 20% of the observations, following the integer split
boundaries in `prep.py`. TTCAEnergy receives a per-channel z-score transform
fit on the training partition during preparation. The forecasting framework
also retains its configured training-fitted, per-channel scaler. Metrics use
the configured evaluation scale; they should not automatically be interpreted
as values in the original physical units.

## Training and evaluation

Supported backbones: **Autoformer, Crossformer, DLinear, HaKAN, Informer,
iTransformer, NLinear, PatchTST, SEMixer, SOFTS, TimesNet, and TQNet**.

### SABA

Run TTCAEnergy / Autoformer with L=36, H=24, and seed 42:

```bash
python saba/scripts/latent/run.py \
  --method saba --dataset TTCAEnergy --model Autoformer \
  --input-len 36 --output-len 24 --num-features 9 \
  --timestamp-sizes 53,12,31 --cycle 53 \
  --seed 42 --gpu 0 --run-name h24_seed42 \
  --saba-gate-file /path/to/calibrated_gates.csv
```

For CarbonMonitor, set `--input-len 96`, `--num-features 6`,
`--timestamp-sizes 7,31,366`, and `--cycle 7`.
The `--cycle` argument configures the TQNet backbone; SABA estimates its own
sample-wise dominant period from the input.

The calibration CSV requires the columns `dataset`, `model`, `gate`, and
`response_ratio`, with a matching row for each requested dataset/backbone.
Use a separate calibration file for each horizon and seed, since the loader
matches rows by dataset and model. Alternatively, supply the calibrated values
through `--saba-gate-value` and `--saba-response-ratio-value`; each explicit
value takes precedence over its CSV field.

**Calibration scope.** This source release does not bundle calibration results
or an offline response-calibration pipeline. The runner expects externally
computed response ratios and gates. When omitted, both default to 1.0, giving
neutral backbone modulation. Such a run checks execution but does not reproduce
the paper's calibrated SABA setting.

### Raw backbone baseline

```bash
python saba/scripts/latent/run.py \
  --method raw --dataset TTCAEnergy --model Autoformer \
  --input-len 36 --output-len 24 --num-features 9 \
  --timestamp-sizes 53,12,31 --cycle 53 \
  --seed 42 --gpu 0 --run-name h24_raw_seed42
```

The additional `--method spectral` option runs the shared spectral dropout baseline.

### Training settings and outputs

The runner defaults to 100 epochs, early stopping with patience 20,
batch size 16, learning rate `1e-4`, weight decay `5e-4`, and seed 42.
MAE is the training loss and validation model-selection metric; evaluation
reports MAE, MSE, and RMSE. SABA defaults to `p_min=0.05` and `p_max=0.5`
before final gate scaling.

Generated files are written under:

```text
outputs/latent/<dataset>/<backbone>/<method>/<run-name>/
```

Use distinct `--run-name` values or `--output-root` directories for separate
experiments. The Bash convenience entry point `run.sh` forwards dataset-specific
timestamp and cycle settings and accepts calibrated values through the
`SABA_GATE` and `SABA_RATIO` environment variables.

## Ablation studies

[`ablation/run.py`](ablation/run.py) reuses the training entry point and changes
only the probability allocation path. It accepts the same training arguments
plus `--ablation-variant`.

Let `s` denote the batch-normalized global deviation, `eta` the unsupported
deviation ratio, `eta_eff` the backbone-conditioned ratio, and `w` the gate.
The variants change the signal passed to the shared Softplus/tanh probability
mapping as follows:

| Variant | Signal to probability mapping | Final gate | Interpretation |
| :--- | :--- | :--- | :--- |
| `full` | `s * eta_eff` | `w` | Complete SABA |
| `no_backbone` | `s * eta` | `1` | Removes both gate and backbone conditioning |
| `global_only` | `s` | `w` | Removes recurring-support modulation |
| `recurring_only` | `eta_eff` | `w` | Removes global-deviation magnitude |

For example, replace the main training command's entry point with
`python ablation/run.py --ablation-variant global_only` and keep the dataset,
model, and calibration arguments. Give each variant a distinct run name.
The exact computations are in [`ablation/abl.py`](ablation/abl.py).

## Code organization

```text
SABA/
├── prep.py                         # Dataset preparation
├── run.sh                          # Bash training helper
├── configs/experiment_matrix.json # Dataset/backbone configurations
├── ablation/                       # Ablation implementations and runner
└── saba/
    ├── pyproject.toml
    ├── requirements.txt
    ├── scripts/latent/run.py        # Training/evaluation entry point
    └── src/basicts/
        ├── models/                 # 12 forecasting backbones
        ├── modules/saba.py          # Reliability and backbone modulation
        ├── modules/dropout_ts.py    # Shared spectral/dropout primitives
        ├── data/                   # Sliding-window datasets
        ├── scaler/                 # Framework normalization
        └── runners/callback/
            └── saba_dropout.py     # SABA training integration
```

## Release contents

The repository and supplementary ZIP contain source code, configuration,
dependency declarations, documentation, and license files. Datasets, weights,
logs, figures, analysis outputs, and previous experimental entry points are
excluded. Training creates its own artifacts locally.

The public source repository is [github.com/yanyijin1/SABA](https://github.com/yanyijin1/SABA).

## Acknowledgments and license

The training framework and shared dropout infrastructure build on BasicTS and
DropoutTS. Backbone implementations retain their source comments and
attribution. The included license is [Apache License 2.0](LICENSE).
