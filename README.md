# SABA

Sample-Adaptive and Backbone-Aware Dropout for time series forecasting.

This repository contains the final method implementation, forecasting backbones,
data preparation, training entry points, and ablation code. Datasets, trained
weights, experiment outputs, and analysis artifacts are not distributed here.

## Installation

Use Python 3.10. Install PyTorch for your CUDA version first; the implementation
has been checked with PyTorch 2.2.0. Then run from the repository root:

```bash
python -m pip install -r saba/requirements.txt
python -m pip install -e ./saba --no-deps
```

## Source layout

- `saba/src/basicts/modules/saba.py`: temporal reliability and backbone modulation.
- `saba/src/basicts/modules/dropout_ts.py`: shared spectral scorer and dropout primitives.
- `saba/src/basicts/runners/callback/saba_dropout.py`: SABA training integration.
- `saba/src/basicts/models/`: the 12 forecasting backbones.
- `saba/scripts/latent/run.py`: training and evaluation entry point.
- `prep.py`: chronological data preparation.
- `configs/experiment_matrix.json`: dataset and backbone settings.
- `ablation/`: probability-path ablations.

Supported backbones: Autoformer, Crossformer, DLinear, HaKAN, Informer, NLinear,
PatchTST, SEMixer, SOFTS, TimesNet, TQNet, and iTransformer.

## Data preparation

Obtain the original datasets separately and place their CSV files at:

```text
data/Illness/Illness.csv
data/TTCAEnergy/Energy.csv
data/CarbonMonitor/carbonmonitor_global.csv
data/ExchangeRate/ExchangeRate.csv
```

The CSV schemas and available upstream source URLs are specified in `prep.py`.
Run preparation for one dataset or omit `--dataset` to prepare all four:

```bash
python prep.py --dataset TTCAEnergy
```

The script writes the prepared chronological splits into `datasets/`.
TTCAEnergy uses training-only z-score normalization during preparation; the
other datasets retain their original preparation and framework scaling logic.
Input lengths, feature counts, horizons, and timestamp settings are recorded
in `configs/experiment_matrix.json`.

## Training

Example for TTCAEnergy with Autoformer, L=36 and H=24:

```bash
python saba/scripts/latent/run.py \
  --method saba --dataset TTCAEnergy --model Autoformer \
  --input-len 36 --output-len 24 --num-features 9 \
  --timestamp-sizes 53,12,31 --cycle 53 --seed 42 \
  --saba-gate-file /path/to/calibrated_gates.csv
```

The calibration CSV must contain `dataset`, `model`, `gate`, and
`response_ratio` columns for the desired dataset/backbone. Use a separate file
for each horizon/seed calibration. These inputs are the actual calibrated
backbone gate and response ratio; they are not learned by the forecasting
runner. Alternatively, pass `--saba-gate-value` and
`--saba-response-ratio-value` explicitly. Explicit values override the CSV.
Without calibration inputs, both values default to 1.0 (neutral modulation);
this default is suitable for checking execution, and does not reproduce the
paper's calibrated SABA configurations.

The portable Bash helper forwards dataset-specific timestamp and cycle settings:

```bash
SABA_GATE=<calibrated_gate> SABA_RATIO=<calibrated_response_ratio> \
  bash run.sh TTCAEnergy Autoformer 36 24 9
```

Use `--method raw` for the unchanged backbone or `--method spectral` for the
shared spectral dropout baseline. Run `--help` for all training options.
Training generates new checkpoints and logs under `outputs/`; these files are
excluded from version control.

## Ablations

`ablation/run.py` accepts the same arguments as the main runner plus
`--ablation-variant`:

- `full`: complete SABA allocation.
- `no_backbone`: removes the backbone gate and response-conditioned context.
- `global_only`: retains global deviation and the gate, omitting recurring support.
- `recurring_only`: retains the conditioned recurring signal and gate, omitting global deviation magnitude.

Use a distinct `--run-name` or `--output-root` for every ablation.

## Acknowledgments and license

The training framework and shared dropout infrastructure build on BasicTS and
DropoutTS. The backbone implementations retain their source comments and
attribution. See [LICENSE](LICENSE) for the included Apache License 2.0.
