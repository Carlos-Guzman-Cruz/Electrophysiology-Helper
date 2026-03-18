"""
Filename: patch_clamp_analysis_helper.py

Author: Carlos A. Guzman-Cruz
Date: Jan 2026
Version: 2.0.1
Description:
This file focuses on the math, DataFrame creation, column manipulation,
and returning processed DataFrames to patch_clamp_hub.py.
All functions here are pure-Python (no widgets / display code) so they
can be unit-tested independently of the Jupyter environment.
"""

__author__ = "Carlos A. Guzman-Cruz"
__email__  = "carguz2002@gmail.com"
__version__ = "2.0.1"

import pyabf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks


# ─────────────────────────────────────────────────────────────────────────────
# 1.  RAW → DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def abf_to_dataframe(path: str, sweep: int = 0, channel: int = 0) -> pd.DataFrame:
    """
    Load an .abf file and return its raw data as a DataFrame.

    Columns
    -------
    time              : float  (seconds)
    signal            : float  (primary channel, e.g. pA or mV)
    command           : float  (command waveform, e.g. voltage holding or injected current)
    sweep             : int
    channel           : int

    Parameters
    ----------
    path    : str   – absolute path to the .abf file
    sweep   : int   – which sweep to load (default 0)
    channel : int   – which channel to load (default 0)

    Returns
    -------
    pd.DataFrame
    """
    abf = pyabf.ABF(path)
    abf.setSweep(sweep, channel=channel)

    df = pd.DataFrame({
        "time":    abf.sweepX.copy(),
        "signal":  abf.sweepY.copy(),
        "command": abf.sweepC.copy(),
        "sweep":   sweep,
        "channel": channel,
    })
    df.attrs["abfID"]       = abf.abfID
    df.attrs["protocol"]    = abf.protocol
    df.attrs["sampleRate"]  = abf.sampleRate
    df.attrs["signalUnits"] = abf.sweepUnitsY
    return df


def abf_all_sweeps_to_dataframe(path: str, channel: int = 0) -> pd.DataFrame:
    """
    Load ALL sweeps from an .abf file into a single long-format DataFrame.

    Columns
    -------
    time, signal, command, sweep, channel   (same as abf_to_dataframe)

    Notes
    -----
    For multi-sweep protocols (IH, step protocols) each sweep occupies its
    own block of rows; the 'sweep' column distinguishes them.
    """
    abf = pyabf.ABF(path)
    frames = []
    for sw in abf.sweepList:
        abf.setSweep(sw, channel=channel)
        frames.append(pd.DataFrame({
            "time":    abf.sweepX.copy(),
            "signal":  abf.sweepY.copy(),
            "command": abf.sweepC.copy(),
            "sweep":   sw,
            "channel": channel,
        }))
    df = pd.concat(frames, ignore_index=True)
    df.attrs["abfID"]       = abf.abfID
    df.attrs["protocol"]    = abf.protocol
    df.attrs["sampleRate"]  = abf.sampleRate
    df.attrs["signalUnits"] = abf.sweepUnitsY
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CLEANING  (time window + amplitude thresholds)
# ─────────────────────────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame,
                    t_start:      float | None = None,
                    t_end:        float | None = None,
                    upper_thresh: float | None = None,
                    lower_thresh: float | None = None) -> pd.DataFrame:
    """
    Return a copy of *df* with rows outside the requested window removed.

    Parameters
    ----------
    df            : DataFrame with at least 'time' and 'signal' columns
    t_start       : keep rows where time >= t_start  (None → no lower clip)
    t_end         : keep rows where time <= t_end    (None → no upper clip)
    upper_thresh  : keep rows where signal < upper_thresh  (None → no clip)
    lower_thresh  : keep rows where signal > lower_thresh  (None → no clip)

    Returns
    -------
    Cleaned pd.DataFrame (index reset).
    """
    mask = pd.Series([True] * len(df), index=df.index)

    if t_start is not None:
        mask &= df["time"] >= t_start
    if t_end is not None:
        mask &= df["time"] <= t_end
    if upper_thresh is not None:
        mask &= df["signal"] < upper_thresh
    if lower_thresh is not None:
        mask &= df["signal"] > lower_thresh

    return df.loc[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PEAK / SPIKE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_peaks(df: pd.DataFrame,
                 threshold:       float = 0.0,
                 negative_spikes: bool  = True,
                 min_depth:       float = 0.0,
                 min_isi_ms:      float = 0.0,
                 sample_rate:     float | None = None) -> pd.DataFrame:
    """
    Detect action-potential-like peaks in a cleaned signal DataFrame.

    Parameters
    ----------
    df              : cleaned DataFrame (must have 'time' and 'signal').
    threshold       : amplitude threshold (pA or mV) to consider a crossing.
    negative_spikes : True  → look for downward (negative) spikes.
                      False → look for upward (positive) spikes.
    min_depth       : minimum absolute amplitude above threshold to count.
    min_isi_ms      : minimum inter-spike interval (ms) to suppress doubles.
    sample_rate     : Hz; inferred from df['time'] diffs if None.

    Returns
    -------
    peaks_df : DataFrame with columns
        ['peak_idx', 'time', 'signal', 'sweep']
    """
    if df.empty:
        return pd.DataFrame(columns=["peak_idx", "time", "signal", "sweep"])

    y = df["signal"].values

    if sample_rate is None:
        dt = float(np.median(np.diff(df["time"].values[:1000])))
        sample_rate = 1.0 / dt if dt > 0 else 10_000.0

    min_dist_samples = max(1, int((min_isi_ms / 1000.0) * sample_rate)) if min_isi_ms > 0 else 1

    if negative_spikes:
        inv_y = -y
        height_val = -threshold + min_depth if min_depth > 0 else -threshold
        peaks_idx, props = find_peaks(inv_y, height=height_val, distance=min_dist_samples)
    else:
        height_val = threshold + min_depth
        peaks_idx, props = find_peaks(y, height=height_val, distance=min_dist_samples)

    if len(peaks_idx) == 0:
        return pd.DataFrame(columns=["peak_idx", "time", "signal", "sweep"])

    peaks_df = df.iloc[peaks_idx][["time", "signal"]].copy()
    peaks_df.insert(0, "peak_idx", peaks_idx)
    if "sweep" in df.columns:
        peaks_df["sweep"] = df.iloc[peaks_idx]["sweep"].values
    peaks_df = peaks_df.reset_index(drop=True)
    return peaks_df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  INTER-SPIKE INTERVAL (ISI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_isi(peaks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the inter-spike interval (ISI) from a peaks DataFrame.

    Parameters
    ----------
    peaks_df : output of detect_peaks(); must contain 'time'.

    Returns
    -------
    isi_df : DataFrame with columns
        ['spike_i', 'spike_j', 't_i', 't_j', 'isi_s', 'isi_ms', 'inst_freq_hz']
    """
    if peaks_df is None or len(peaks_df) < 2:
        return pd.DataFrame(columns=["spike_i", "spike_j", "t_i", "t_j",
                                     "isi_s", "isi_ms", "inst_freq_hz"])

    times = peaks_df["time"].values
    records = []
    for k in range(len(times) - 1):
        isi_s  = times[k + 1] - times[k]
        isi_ms = isi_s * 1000.0
        inst_f = 1.0 / isi_s if isi_s > 0 else np.nan
        records.append({
            "spike_i":     k,
            "spike_j":     k + 1,
            "t_i":         times[k],
            "t_j":         times[k + 1],
            "isi_s":       isi_s,
            "isi_ms":      isi_ms,
            "inst_freq_hz": inst_f,
        })
    return pd.DataFrame(records)


def isi_summary(isi_df: pd.DataFrame) -> dict:
    """
    Return summary statistics for an ISI DataFrame.

    Returns
    -------
    dict with keys:
        'isi_mean_ms', 'isi_std_ms', 'isi_cv',
        'inst_freq_mean_hz', 'inst_freq_std_hz',
        'n_intervals'
    """
    if isi_df is None or isi_df.empty:
        return {}

    return {
        "n_intervals":        len(isi_df),
        "isi_mean_ms":        float(isi_df["isi_ms"].mean()),
        "isi_std_ms":         float(isi_df["isi_ms"].std()),
        "isi_cv":             float(isi_df["isi_ms"].std() / isi_df["isi_ms"].mean())
                              if isi_df["isi_ms"].mean() != 0 else np.nan,
        "inst_freq_mean_hz":  float(isi_df["inst_freq_hz"].mean()),
        "inst_freq_std_hz":   float(isi_df["inst_freq_hz"].std()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COLUMN UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def add_column(df: pd.DataFrame, col_name: str, values) -> pd.DataFrame:
    """
    Add (or overwrite) a column in a DataFrame.

    Parameters
    ----------
    df       : source DataFrame
    col_name : name for the new column
    values   : scalar, list, np.ndarray, or pd.Series aligned to df

    Returns
    -------
    New DataFrame with the column added.
    """
    df = df.copy()
    df[col_name] = values
    return df


def remove_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Drop a column (no-op if missing). Returns new DataFrame."""
    df = df.copy()
    if col_name in df.columns:
        df = df.drop(columns=[col_name])
    return df


def rename_column(df: pd.DataFrame, old_name: str, new_name: str) -> pd.DataFrame:
    """Rename a single column. Returns new DataFrame."""
    return df.rename(columns={old_name: new_name})


def add_rolling_mean(df: pd.DataFrame,
                     signal_col: str = "signal",
                     window_ms: float = 10.0,
                     sample_rate: float = 10_000.0,
                     out_col: str = "signal_rolling_mean") -> pd.DataFrame:
    """
    Append a rolling-mean smoothed version of a signal column.

    Parameters
    ----------
    window_ms   : rolling window in milliseconds
    sample_rate : sampling rate in Hz (used to convert ms → samples)

    Returns
    -------
    DataFrame with new column appended.
    """
    window_samples = max(1, int((window_ms / 1000.0) * sample_rate))
    df = df.copy()
    df[out_col] = df[signal_col].rolling(window=window_samples, center=True, min_periods=1).mean()
    return df


def add_zscore(df: pd.DataFrame,
               signal_col: str = "signal",
               out_col:    str = "signal_zscore") -> pd.DataFrame:
    """Append a z-score normalised version of a signal column."""
    df = df.copy()
    mu = df[signal_col].mean()
    sd = df[signal_col].std()
    df[out_col] = (df[signal_col] - mu) / sd if sd != 0 else 0.0
    return df


def add_delta_signal(df: pd.DataFrame,
                     signal_col: str = "signal",
                     out_col:    str = "delta_signal") -> pd.DataFrame:
    """Append a first-difference (dV/dt or dI/dt proxy) column."""
    df = df.copy()
    df[out_col] = df[signal_col].diff().fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6.  HIGH-LEVEL PIPELINE  (called by hub)
# ─────────────────────────────────────────────────────────────────────────────

def run_vc_pipeline(path:          str,
                    t_start:       float = 0.0,
                    t_end:         float | None = None,
                    upper_thresh:  float = 200.0,
                    lower_thresh:  float = -200.0,
                    spike_thresh:  float = 0.0,
                    negative_spikes: bool = True,
                    min_depth:     float = 0.0,
                    min_isi_ms:    float = 0.0) -> dict:
    """
    End-to-end pipeline for a Voltage-Clamp GapFree recording.

    Steps
    -----
    1. Load ABF → raw DataFrame
    2. Clean (time window + amplitude thresholds)
    3. Detect peaks/spikes
    4. Compute ISI
    5. Return all DataFrames + summary statistics

    Returns
    -------
    dict matching the isi_key_data entry schema:
    {
        'path':            str,
        'tmStrT':          float,
        'tmEND':           float,
        'upTHR':           float,
        'lwTHR':           float,
        'cleaned_df':      pd.DataFrame,
        'peaks_df':        pd.DataFrame,
        'isi_df':          pd.DataFrame,
        'isi_mean':        float,
        'inst_freq_mean':  float,
        'summary':         dict,
    }
    """
    # ── 1. Load ──────────────────────────────────────────────────────────────
    raw_df = abf_to_dataframe(path)
    sr     = raw_df.attrs.get("sampleRate", 10_000.0)

    # Set defaults for t_end
    if t_end is None:
        t_end = float(raw_df["time"].max())

    # ── 2. Clean ─────────────────────────────────────────────────────────────
    cleaned_df = clean_dataframe(raw_df,
                                 t_start=t_start, t_end=t_end,
                                 upper_thresh=upper_thresh,
                                 lower_thresh=lower_thresh)

    # ── 3. Peaks ─────────────────────────────────────────────────────────────
    peaks_df = detect_peaks(cleaned_df,
                            threshold=spike_thresh,
                            negative_spikes=negative_spikes,
                            min_depth=min_depth,
                            min_isi_ms=min_isi_ms,
                            sample_rate=sr)

    # ── 4. ISI ───────────────────────────────────────────────────────────────
    isi_df  = compute_isi(peaks_df)
    summary = isi_summary(isi_df)

    return {
        "path":           path,
        "tmStrT":         t_start,
        "tmEND":          t_end,
        "upTHR":          upper_thresh,
        "lwTHR":          lower_thresh,
        "cleaned_df":     cleaned_df,
        "peaks_df":       peaks_df,
        "isi_df":         isi_df,
        "isi_mean":       summary.get("isi_mean_ms", np.nan),
        "inst_freq_mean": summary.get("inst_freq_mean_hz", np.nan),
        "summary":        summary,
    }
