#!/usr/bin/env python3
# === stream_infer.py ===
# Inference-only pipeline: prints ONE prediction every N seconds and (optionally) logs to CSV.
# Default window: 2007-01-01 → 2017-12-31 (configurable via CLI).
# No training: loads a pre-trained scikit-learn model (joblib).

import os
import csv
import time
import argparse
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import xarray as xr
from joblib import load


# -------------------
# ENSO I/O
# -------------------
def load_enso_indices(path: str = "nino34.long.anom.data.txt") -> pd.Series:
    """
    Reads the txt data file and returns a monthly ENSO Series starting 1870-01-01.
    Each line is assumed like: YEAR v1 v2 ... v12
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ENSO file not found: {path}")
    vals = []
    with open(path) as f:
        for line in f:
            toks = line.split()
            if len(toks) > 1:
                vals.extend(map(float, toks[1:]))
    # raw monthly index
    s = pd.Series(vals, index=pd.date_range("1870-01-01", freq="MS", periods=len(vals)))
    return s


def clean_enso(series: pd.Series, missing_sentinel: float = -99.99) -> pd.Series:
    """
    Replace sentinel with NaN, drop missing, and normalize index to month-start.
    Uses period → timestamp with how='S' (start of month) to avoid 'MS' issue.
    """
    s = series.replace(missing_sentinel, np.nan).dropna()
    s.index = pd.to_datetime(s.index)
    s.index = s.index.to_period("M").to_timestamp(how="S")
    s = s.sort_index()
    return s


# -------------------
# Feature builder (inference-only)
# -------------------
def build_features_for_inference(
    start_date: str,
    end_date: str,
    lead_time: int,
    max_lag: int = 15,
    sst_path: str = "sst.mon.mean.trefadj.anom.1880to2018.nc",
    enso_series: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns a DataFrame with index=monthly dates (MS) and columns:
      ['mean', 'std', 'ENSO_lag1', ..., f'ENSO_lag{max_lag}'].
    Pure inference (no labels).

    Convention:
      For month t, lag k = ENSO[t + lead_time - k].
      Implemented via shift(-(lead_time - k)) and reindexing to SST dates.
    """
    if not os.path.exists(sst_path):
        raise FileNotFoundError(f"SST file not found: {sst_path}")

    # Load SST and aggregate
    ds = xr.open_dataset(sst_path)
    try:
        sst = ds["sst"].sel(time=slice(start_date, end_date))

        # Use dataset's own time coordinate, snapped to month START
        idx = pd.to_datetime(sst["time"].values)
        idx = pd.DatetimeIndex(idx).to_period("M").to_timestamp(how="S")

        nT = sst.shape[0]
        sst_vals = np.asarray(sst.values).reshape(nT, -1)
        sst_vals[np.isnan(sst_vals)] = 0

        df_monthly = pd.DataFrame(sst_vals, index=idx)

        # Drop all-zero columns (e.g., land masks)
        nonzero = ~(df_monthly == 0).all(axis=0)
        df_monthly = df_monthly.loc[:, nonzero]

        # Aggregate features
        df_feat = pd.DataFrame({
            "mean": df_monthly.mean(axis=1),
            "std":  df_monthly.std(axis=1, ddof=0),
        })
    finally:
        ds.close()

    # ENSO lags
    if enso_series is None:
        enso_series = clean_enso(load_enso_indices())
    else:
        enso_series = clean_enso(enso_series)

    # Force monthly frequency; no fill so we avoid leakage
    enso_ms = enso_series.asfreq("MS")

    # Build lag features per convention: lag k at t = ENSO[t + lead_time - k]
    lag_df = pd.DataFrame(index=df_feat.index)
    for k in range(1, max_lag + 1):
        offset = lead_time - k
        lag_k = enso_ms.shift(-offset)  # negative brings future into current row
        lag_df[f"ENSO_lag{k}"] = lag_k.reindex(df_feat.index)

    feats = pd.concat([df_feat, lag_df], axis=1).dropna()
    feature_names = ["mean", "std"] + [f"ENSO_lag{k}" for k in range(1, max_lag + 1)]
    feats = feats[feature_names]
    return feats, feature_names


# -------------------
# Model loading
# -------------------
def load_model(model_path: str = "linear_lag.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load(model_path)


# -------------------
# CSV logging helpers
# -------------------
def init_csv(out_csv: str, feature_names: List[str]) -> None:
    """
    Create CSV with header if it doesn't exist or is empty.
    Columns: timestamp, pred, then all feature columns (for audit).
    """
    needs_header = True
    if os.path.exists(out_csv):
        try:
            needs_header = (os.path.getsize(out_csv) == 0)
        except OSError:
            needs_header = True

    if needs_header:
        header = ["timestamp", "pred"] + feature_names
        with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)


def append_csv_row(out_csv: str, ts: pd.Timestamp, pred: float, row: pd.Series) -> None:
    """
    Append a single prediction row. Keeps file open only briefly to avoid locks.
    """
    try:
        with open(out_csv, mode="a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ts.strftime("%Y-%m-%d"), f"{pred:.10f}"] + [f"{row[k]:.10f}" for k in row.index])
            try:
                f.flush()
                os.fsync(f.fileno())  # ensure data hits disk (if supported)
            except Exception:
                pass
    except PermissionError:
        # If the file is open in Excel, append will fail—just warn and continue.
        print(f"[WARN] Could not write to '{out_csv}' (maybe open in another program). Will keep streaming.")


# -------------------
# Streaming inference
# -------------------
def stream_predictions(
    start_date: str = "2007-01-01",
    end_date: str = "2017-12-31",
    lead_time: int = 1,
    max_lag: int = 15,
    sst_path: str = "sst.mon.mean.trefadj.anom.1880to2018.nc",
    model_path: str = "linear_lag.joblib",
    enso_path: str = "nino34.long.anom.data.txt",
    interval_sec: float = 10.0,
    show_features: bool = False,
    out_csv: Optional[str] = None,
):
    """
    Prints a single prediction every `interval_sec` seconds to the terminal.
    If out_csv is provided, appends each prediction to CSV in real time.
    """
    enso_clean = clean_enso(load_enso_indices(enso_path))
    feats_df, feature_names = build_features_for_inference(
        start_date=start_date,
        end_date=end_date,
        lead_time=lead_time,
        max_lag=max_lag,
        sst_path=sst_path,
        enso_series=enso_clean,
    )
    model = load_model(model_path)

    if feats_df.empty:
        print("[WARN] No rows to score after feature alignment. "
              "Check date window, lead_time/max_lag, and that input files cover the window.")
        return

    # Prepare CSV if requested
    if out_csv is not None:
        init_csv(out_csv, feature_names)
        print(f"[INFO] Live CSV logging to: {os.path.abspath(out_csv)}")

    print(f"\n[INFO] Streaming predictions from {feats_df.index.min().date()} to {feats_df.index.max().date()}")
    print(f"[INFO] Interval: {interval_sec} sec — Model: {os.path.basename(model_path)} — Rows: {len(feats_df)}")
    print("[INFO] Press Ctrl+C to stop.\n")

    try:
        for ts, row in feats_df.iterrows():
            x = row.values.reshape(1, -1)
            y_pred = model.predict(x)[0]
            y_out = float(np.ravel([y_pred])[0])

            if show_features:
                preview_keys = list(row.index)[:6]  # mean, std, a few lags
                feat_str = ", ".join(f"{k}={row[k]:.4f}" for k in preview_keys)
                print(f"{ts.strftime('%Y-%m-%d')}  pred={y_out:.6f}  | {feat_str} ...")
            else:
                print(f"{ts.strftime('%Y-%m-%d')}  pred={y_out:.6f}")

            if out_csv is not None:
                append_csv_row(out_csv, ts, y_out, row)

            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\n[INFO] Streaming interrupted by user.")


# -------------------
# CLI entrypoint
# -------------------
def parse_args():
    p = argparse.ArgumentParser(description="Stream monthly ENSO predictions every N seconds (inference only).")
    p.add_argument("--start", type=str, default="2007-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default="2017-12-31", help="End date (YYYY-MM-DD)")
    p.add_argument("--lead", type=int, default=3, help="Lead time in months used at training")
    p.add_argument("--max_lag", type=int, default=15, help="Number of ENSO lag features")
    p.add_argument("--sst_path", type=str, default="sst.mon.mean.trefadj.anom.1880to2018.nc", help="SST anomalies NetCDF file")
    p.add_argument("--enso_path", type=str, default="nino34.long.anom.data.txt", help="ENSO text file (nino3.4 anomalies)")
    p.add_argument("--model", type=str, default="linear_lag.joblib", help="Path to trained model (joblib)")
    p.add_argument("--interval", type=float, default=10.0, help="Seconds between successive predictions")
    p.add_argument("--show_features", action="store_true", help="Print a subset of features alongside predictions")
    p.add_argument("--out_csv", type=str, default=None, help="Path to a CSV file for live predictions logging")
    return p.parse_args()


def main():
    args = parse_args()
    stream_predictions(
        start_date=args.start,
        end_date=args.end,
        lead_time=args.lead,
        max_lag=args.max_lag,
        sst_path=args.sst_path,
        model_path=args.model,
        enso_path=args.enso_path,
        interval_sec=args.interval,
        show_features=args.show_features,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()


# python .\stream_infer.py `
#   --model .\linear_lag.joblib `
#   --sst_path .\sst.mon.mean.trefadj.anom.1880to2018.nc `
#   --enso_path .\nino34.long.anom.data.txt `
#   --start 2007-01-01 `
#   --end 2017-12-31 `
#   --lead 1 `
#   --max_lag 15 `
#   --interval 10 `
#   --out_csv .\live_predictions.csv `
#   --show_features
