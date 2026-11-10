#!/usr/bin/env python
"""Prepare the four SABA datasets from raw files.

The script keeps one simple chronological 7:1:2 split for every dataset.
It writes the BasicTS files used by SABA without touching model code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("Illness", "ExchangeRate", "CarbonMonitor", "TTCAEnergy")


def split_bounds(n: int) -> tuple[int, int]:
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError(f"invalid split for length={n}")
    return train_end, val_end


def save_split(out_dir: Path, values: np.ndarray, stamps: np.ndarray, meta: dict, val_mode: str = "ratio") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_end, val_end = split_bounds(len(values))
    if val_mode == "length":
        val_end = train_end + int(len(values) * 0.1)
    parts = {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, len(values)),
    }
    for name, slc in parts.items():
        np.save(out_dir / f"{name}_data.npy", values[slc].astype(np.float32))
        np.save(out_dir / f"{name}_timestamps.npy", stamps[slc].astype(np.float32))
    meta.update(
        {
            "shape": [int(values.shape[0]), int(values.shape[1])],
            "timestamps_shape": [int(stamps.shape[0]), int(stamps.shape[1])],
            "train_val_test_ratio": [0.7, 0.1, 0.2],
            "split_lengths": {k: int(s.stop - s.start) for k, s in parts.items()},
        }
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def calendar_stamps(dates: pd.DatetimeIndex, time_of_day: np.ndarray | None = None) -> np.ndarray:
    cols = []
    if time_of_day is not None:
        cols.append(time_of_day.astype(np.float32))
    cols.extend(
        [
            (dates.dayofweek.to_numpy(dtype=np.float32) / 7.0),
            ((dates.day.to_numpy(dtype=np.float32) - 1.0) / 31.0),
            ((dates.dayofyear.to_numpy(dtype=np.float32) - 1.0) / 366.0),
        ]
    )
    return np.stack(cols, axis=-1).astype(np.float32)


def load_illness(raw_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    path = raw_dir / "Illness" / "Illness.csv"
    df = pd.read_csv(path)
    dates = pd.DatetimeIndex(pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S"))
    values = df.drop(columns=["date"]).to_numpy(dtype=np.float32)
    stamps = calendar_stamps(dates, np.zeros(len(df), dtype=np.float32))
    meta = {
        "name": "Illness",
        "source": "https://raw.githubusercontent.com/scalation/data/master/Influenza/national_illness.csv",
        "raw_path": str(path),
        "columns": list(df.columns[1:]),
        "timestamps_description": ["time of day", "day of week", "day of month", "day of year"],
    }
    return values, stamps, meta


def load_exchange(raw_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    path = raw_dir / "ExchangeRate" / "ExchangeRate.csv"
    df = pd.read_csv(path)
    dates = pd.DatetimeIndex(pd.to_datetime(df["date"], format="%Y/%m/%d %H:%M"))
    values = df.drop(columns=["date"]).to_numpy(dtype=np.float32)
    time_of_day = np.zeros(len(df), dtype=np.float32)
    stamps = calendar_stamps(dates, time_of_day)
    meta = {
        "name": "ExchangeRate",
        "domain": "exchange rate",
        "raw_path": str(path),
        "frequency (minutes)": 1440,
        "num_time_steps": int(len(values)),
        "num_vars": int(values.shape[1]),
        "has_graph": False,
        "timestamps_description": ["time of day", "day of week", "day of month", "day of year"],
    }
    return values, stamps, meta


def load_carbon(raw_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    path = raw_dir / "CarbonMonitor" / "carbonmonitor_global.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="raise")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    wide = df.groupby(["date", "sector"], observed=True)["value"].sum(min_count=1).unstack("sector")
    wide = wide.sort_index()
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"))
    wide = wide.interpolate(method="time", limit_direction="both").ffill().bfill().fillna(0.0)
    wide = wide.reindex(sorted(wide.columns), axis=1)
    dates = pd.DatetimeIndex(wide.index)
    stamps = np.stack(
        [
            dates.dayofweek.to_numpy(dtype=np.float32) / 7.0,
            (dates.day.to_numpy(dtype=np.float32) - 1.0) / 31.0,
            (dates.dayofyear.to_numpy(dtype=np.float32) - 1.0) / 366.0,
        ],
        axis=-1,
    ).astype(np.float32)
    meta = {
        "name": "CarbonMonitor",
        "source": "https://datas.carbonmonitor.org/API/downloadFullDataset.php?source=carbon_global",
        "raw_path": str(path),
        "variant": "global_sector",
        "country": None,
        "columns": list(wide.columns),
        "date_start": str(dates.min().date()),
        "date_end": str(dates.max().date()),
        "timestamp_columns": ["dayofweek", "day", "dayofyear"],
        "recommended_timestamp_sizes": [7, 31, 366],
        "recommended_cycle": 7,
    }
    return wide.to_numpy(dtype=np.float32), stamps, meta


def load_ttca(raw_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    path = raw_dir / "TTCAEnergy" / "Energy.csv"
    table = pd.read_csv(path)
    dates = pd.DatetimeIndex(pd.to_datetime(table["date"], errors="raise"))
    feature_cols = ["OT"] + [c for c in table.columns if c.startswith("Weekly ")]
    values = table[feature_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    train_end, _ = split_bounds(len(values))
    mean = values[:train_end].mean(axis=0)
    std = values[:train_end].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    scaled = ((values - mean) / std).astype(np.float32)
    iso_week = dates.isocalendar().week.to_numpy(dtype=np.float32)
    stamps = np.stack(
        [
            (iso_week - 1.0) / 53.0,
            (dates.month.to_numpy(dtype=np.float32) - 1.0) / 12.0,
            (dates.day.to_numpy(dtype=np.float32) - 1.0) / 31.0,
        ],
        axis=-1,
    ).astype(np.float32)
    meta = {
        "name": "TTCAEnergy",
        "source": "https://raw.githubusercontent.com/synapsespectrum/TTCA/main/dataset/Energy.csv",
        "raw_path": str(path),
        "columns": feature_cols,
        "date_start": str(dates.min().date()),
        "date_end": str(dates.max().date()),
        "timestamp_columns": ["iso_week", "month", "day"],
        "recommended_timestamp_sizes": [53, 12, 31],
        "recommended_cycle": 53,
        "normalization": "z-score fit on chronological training split only",
    }
    return scaled, stamps, meta


LOADERS = {
    "Illness": load_illness,
    "ExchangeRate": load_exchange,
    "CarbonMonitor": load_carbon,
    "TTCAEnergy": load_ttca,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--raw-root", default="data")
    parser.add_argument("--out-root", default="datasets")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    if not raw_root.is_absolute():
        raw_root = root / raw_root
    if not out_root.is_absolute():
        out_root = root / out_root

    names = [args.dataset] if args.dataset else list(DATASETS)
    for name in names:
        values, stamps, meta = LOADERS[name](raw_root)
        val_mode = "bounds" if name == "TTCAEnergy" else "length"
        save_split(out_root / name, values, stamps, meta, val_mode=val_mode)
        print(name, values.shape, stamps.shape, "->", out_root / name)


if __name__ == "__main__":
    main()
