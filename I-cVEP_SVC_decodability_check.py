#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decode_multi_file_best_ivep.py

Author:
    Milán András Fodor (0009-0009-9769-0821)
    Ivan Volosyak     (0000-0001-6555-7617)

Description:
    Performs bit‑wise decoding of I‑VEP trials using a Support Vector
    Classifier (SVC) with leave‑one‑trial‑out (LOTO) cross‑validation.
    - Automatically detects comma- vs. tab-separated CSVs.
    - Segments data by ‘perfect_mseq_label’ into trials.
    - Extracts band‑power, harmonic, low‑frequency, and flip‑potential
      features (causal, Hilbert-based, t‑norm calibration).
    - Imputes missing features, standardizes, and trains an RBF‑SVM.
    - Reports per‑fold and overall bit‑level accuracy per file, plus
      aggregate metrics across all files.

Usage:
    python decode_multi_file_best_ivep.py \
        --data-dir /path/to/csvs \
        [--min-trials 22] \
        [--verbose]

Requirements:
    Python 3.8+
    numpy, pandas, scipy, scikit-learn
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import (
    butter,
    hilbert,
    iirnotch,
    lfilter,
    resample_poly,
    sosfilt,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ──────────────── Configuration ────────────────────────────

DEFAULT_MIN_TRIALS = 22
DEFAULT_PHASE_VAL = 4

# EEG / timing
FS_RAW = 1200
REFRESH_RATE = 360
FRAMES_PER_BIT = 15
BITS_PER_TRIAL = 31
SAMPLES_PER_BIT = int(FRAMES_PER_BIT * FS_RAW / REFRESH_RATE)
SAMPLES_PER_TRIAL = SAMPLES_PER_BIT * BITS_PER_TRIAL

# Columns and channels
PHASE_COL = "HHCVEP_phase_label"
MSEQ_COL = "perfect_mseq_label"
CHANNELS = ["Gtec_EEG_Pz", "Gtec_EEG_POz", "Gtec_EEG_Oz", "Gtec_EEG_Iz"]

# SVM hyperparameters
SVM_C = 200
SVM_KERNEL = "rbf"
SVM_GAMMA = 1

# Feature selection
FEATURE_KEYS = [
    "H60p", "H120p", "B0", "B1", "LF_B0", "LF_B1",
    "flip_pot_B0", "flip_pot_LF_B0", "flip_pot_H60p",
    "time_in_trial", "H60p_tnorm", "LF_B0_tnorm"
]

# Band definitions
NOTCH_FREQ = 50  # Hz
NOTCH_Q = 30
ENV_BANDS = [(58, 62), (50, 70), (40, 80)]
HARMONICS = [60, 120, 180]
LOW_FREQ_BANDS = [(10, 14), (20, 28)]

# ─────────────── Utility Functions ────────────────────────

def make_blocks(trial_indices, bits_per_trial=BITS_PER_TRIAL):
    """Return list of (trial_idx, bit_idx) pairs for given trials."""
    return [(ti, bi) for ti in trial_indices for bi in range(bits_per_trial)]


def extract_features_causal_tnorm(
    block_list,
    fs_ds,
    bit_interval,
    window_delay,
    analysis_window,
    trial_envelopes,
    env_bands,
    harm_bands,
    lowf_bands,
    enable_lowf,
    enable_flip,
    bits_per_trial,
    raw_trials,
    fb_env,
    fb_harm,
    fb_lowf,
    calib_H60=None,
    calib_LF=None
):
    """Extract power, harmonic, low-freq, flip-potential, and t‑norm features."""
    N = len(block_list)
    feat = {k: np.full(N, np.nan) for k in []}
    # Initialize feature arrays
    mf = np.full(N, np.nan)
    hm = np.full(N, np.nan)
    env_feats = {f"B{i}": np.full(N, np.nan) for i in range(len(env_bands))}
    harm_feats = {h: np.full(N, np.nan) for h in harm_bands}
    lf_feats = {f"LF_B{i}": np.full(N, np.nan) for i in range(len(lowf_bands))}
    flip_B0 = np.full(N, np.nan)
    flip_LF = np.full(N, np.nan)
    flip_H60 = np.full(N, np.nan)
    prev_B0 = defaultdict(lambda: np.nan)
    prev_LF = defaultdict(lambda: np.nan)
    prev_H60 = defaultdict(lambda: np.nan)
    t_in_trial = np.array([
        bit_idx / (bits_per_trial - 1)
        if bits_per_trial > 1 else 0
        for _, bit_idx in block_list
    ])
    H60_norm = np.full(N, np.nan)
    LF_norm = np.full(N, np.nan)
    raw_H60_vals = []
    raw_LF_vals = []

    for idx, (t_idx, b_idx) in enumerate(block_list):
        start = b_idx * bit_interval + window_delay
        end = start + analysis_window
        env = trial_envelopes[t_idx]
        max_samples = raw_trials[t_idx].shape[0]

        if start < 0 or end > max_samples:
            continue

        # Fundamental (60Hz)
        if 60 in harm_bands and "H60" in env:
            seg = env["H60"][start:end]
            val = seg.mean() if seg.size else np.nan
            mf[idx] = val
            if calib_H60 is None and not np.isnan(val):
                raw_H60_vals.append(val)
            elif calib_H60:
                H60_norm[idx] = val / calib_H60 if calib_H60 else np.nan

        # Harmonics
        sum_h = 0
        has_h = False
        for h in harm_bands:
            key = f"H{h}"
            if key in env:
                seg = env[key][start:end]
                mval = seg.mean() if seg.size else np.nan
                harm_feats[h][idx] = mval
                if not np.isnan(mval):
                    sum_h += mval
                    has_h = True
        if has_h:
            hm[idx] = sum_h

        # Env bands
        for i, key in enumerate(env_feats):
            seg = env[key][start:end] if key in env else np.array([])
            env_feats[key][idx] = seg.mean() if seg.size else np.nan

        # Low‑freq
        if enable_lowf:
            for i, key in enumerate(lf_feats):
                seg = env[key][start:end] if key in env else np.array([])
                val = seg.mean() if seg.size else np.nan
                lf_feats[key][idx] = val
                if key == "LF_B0":
                    if calib_LF is None and not np.isnan(val):
                        raw_LF_vals.append(val)
                    elif calib_LF:
                        LF_norm[idx] = val / calib_LF if calib_LF else np.nan

        # Flip potentials
        if enable_flip:
            cur_B0 = env_feats.get("B0", np.full(N, np.nan))[idx]
            if not np.isnan(cur_B0) and not np.isnan(prev_B0[t_idx]):
                flip_B0[idx] = cur_B0 - prev_B0[t_idx]
            prev_B0[t_idx] = cur_B0

            if enable_lowf:
                cur_LF = lf_feats.get("LF_B0", np.full(N, np.nan))[idx]
                if not np.isnan(cur_LF) and not np.isnan(prev_LF[t_idx]):
                    flip_LF[idx] = cur_LF - prev_LF[t_idx]
                prev_LF[t_idx] = cur_LF

            if not np.isnan(mf[idx]) and not np.isnan(prev_H60[t_idx]):
                flip_H60[idx] = mf[idx] - prev_H60[t_idx]
            prev_H60[t_idx] = mf[idx]

    # Calculate calibration means if needed
    calc_H60 = np.nanmean(raw_H60_vals) if raw_H60_vals else np.nan
    calc_LF = np.nanmean(raw_LF_vals) if raw_LF_vals else np.nan

    if calib_H60 is None and not np.isnan(calc_H60):
        H60_norm = np.where(~np.isnan(mf), mf / calc_H60, np.nan)
    if calib_LF is None and not np.isnan(calc_LF):
        LF_norm = np.where(~np.isnan(lf_feats.get("LF_B0", [])), lf_feats["LF_B0"] / calc_LF, np.nan)

    # Assemble feature dictionary
    features = {
        "mf": mf, "hm": hm,
        **env_feats,
        **{f"H{h}p": harm_feats[h] for h in harm_feats},
        **(lf_feats if enable_lowf else {}),
        **({"flip_pot_B0": flip_B0,
            "flip_pot_LF_B0": flip_LF,
            "flip_pot_H60p": flip_H60} if enable_flip else {}),
        "time_in_trial": t_in_trial,
        "H60p_tnorm": H60_norm,
        "LF_B0_tnorm": LF_norm,
    }
    # Include calibration means for next call
    if calib_H60 is None:
        features["_calib_H60"] = calc_H60
    if calib_LF is None:
        features["_calib_LF"] = calc_LF

    return features


def process_file(csv_path: Path, args) -> dict:
    """Load CSV, segment trials, extract features, run LOTO SVC, return metrics."""
    # 1) Load CSV robustly
    try:
        df = pd.read_csv(csv_path, sep=",", low_memory=False, encoding="latin1")
        if df.shape[1] <= 1:
            df = pd.read_csv(csv_path, sep="\t", low_memory=False, encoding="latin1")
    except Exception as err:
        print(f"  [ERROR] reading {csv_path.name}: {err}")
        return {}

    df.columns = df.columns.str.strip()
    required = [PHASE_COL, MSEQ_COL] + CHANNELS
    if any(col not in df for col in required):
        print(f"  [ERROR] missing required columns in {csv_path.name}")
        return {}

    # 2) Filter to target phase
    df_phase = df[df[PHASE_COL] == args.phase].reset_index(drop=True)

    # 3) Detect trial starts
    prev = df_phase[MSEQ_COL].shift(1) == -1
    curr = df_phase[MSEQ_COL] != -1
    starts = df_phase.index[(prev & curr) | (df_phase.index == 0 & curr)].tolist()

    trials_raw = []
    bits_raw = []
    for st in starts:
        en = st + SAMPLES_PER_TRIAL
        if en > len(df_phase):
            continue
        seg = df_phase.iloc[st:en]
        if seg[CHANNELS].isnull().any().any():
            continue
        trials_raw.append(seg[CHANNELS].values)
        bits = seg[MSEQ_COL].values.astype(int)[::SAMPLES_PER_BIT]
        bits_raw.append(bits)

    if len(trials_raw) < args.min_trials:
        print(f"  [WARN] only {len(trials_raw)} trials; skip {csv_path.name}")
        return {}

    # 4) Prepare filters
    fs_ds = FS_RAW  # downsample factor is 1 here
    nyq = fs_ds / 2.0
    # Notch
    if NOTCH_FREQ < nyq:
        b_notch, a_notch = iirnotch(NOTCH_FREQ / nyq, NOTCH_Q)
    else:
        b_notch, a_notch = None, None
    # Enveloping filters
    fb_env = {
        f"B{i}": butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
        for i, (low, high) in enumerate(ENV_BANDS)
        if low < nyq
    }
    fb_harm = {
        h: butter(4, [(h - 2) / nyq, (h + 2) / nyq], btype="bandpass", output="sos")
        for h in HARMONICS
        if h + 2 < nyq
    }
    fb_lowf = {
        f"LF_B{i}": butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
        for i, (low, high) in enumerate(LOW_FREQ_BANDS)
        if low < nyq
    } if args.enable_lowf else {}

    # 5) Envelope extraction
    trial_envs = []
    for raw in trials_raw:
        ds = resample_poly(raw, up=fs_ds, down=FS_RAW, axis=0)
        if b_notch is not None:
            for ch in range(ds.shape[1]):
                ds[:, ch] = lfilter(b_notch, a_notch, ds[:, ch])
        env = {}
        for k, sos in fb_env.items():
            env[k] = np.abs(hilbert(sosfilt(sos, ds, axis=0), axis=0)).mean(axis=1)
        for h, sos in fb_harm.items():
            env[f"H{h}"] = np.abs(hilbert(sosfilt(sos, ds, axis=0), axis=0)).mean(axis=1)
        if args.enable_lowf:
            for k, sos in fb_lowf.items():
                env[k] = np.abs(hilbert(sosfilt(sos, ds, axis=0), axis=0)).mean(axis=1)
        trial_envs.append(env)

    # 6) LOTO cross‑validation
    n_trials = len(trial_envs)
    per_fold_acc = []
    all_preds, all_truth = [], []

    print(f"  LOTO on {n_trials} trials...")
    for left in range(n_trials):
        train_idx = [i for i in range(n_trials) if i != left]
        test_idx = [left]

        # Build blocks
        train_blocks = make_blocks(train_idx)
        test_blocks = make_blocks(test_idx)

        # Extract features & calibrate
        feats_train = extract_features_causal_tnorm(
            train_blocks, fs_ds, SAMPLES_PER_BIT, 0, 2,
            trial_envs, ENV_BANDS, HARMONICS, LOW_FREQ_BANDS,
            args.enable_lowf, args.enable_flip, BITS_PER_TRIAL,
            trials_raw, fb_env, fb_harm, fb_lowf
        )
        calib_H60 = feats_train.pop("_calib_H60", np.nan)
        calib_LF = feats_train.pop("_calib_LF", np.nan)
        feats_test = extract_features_causal_tnorm(
            test_blocks, fs_ds, SAMPLES_PER_BIT, 0, 2,
            trial_envs, ENV_BANDS, HARMONICS, LOW_FREQ_BANDS,
            args.enable_lowf, args.enable_flip, BITS_PER_TRIAL,
            trials_raw, fb_env, fb_harm, fb_lowf,
            calib_H60, calib_LF
        )

        # Assemble matrices
        y_train = np.concatenate([bits_raw[i] for i in train_idx])
        y_test = bits_raw[test_idx[0]]
        X_train = np.vstack([feats_train[k] for k in FEATURE_KEYS]).T
        X_test  = np.vstack([feats_test[k]  for k in FEATURE_KEYS]).T

        # Impute & scale
        col_means = np.nanmean(X_train, axis=0)
        X_train = np.where(np.isnan(X_train), col_means, X_train)
        X_test  = np.where(np.isnan(X_test),  col_means, X_test)

        # Skip unsuitable folds
        if X_train.shape[0] < 2 or len(np.unique(y_train)) < 2:
            preds = np.full_like(y_test, np.bincount(y_train).argmax())
        else:
            scaler = StandardScaler().fit(X_train)
            clf = SVC(
                C=SVM_C, kernel=SVM_KERNEL, gamma=SVM_GAMMA,
                class_weight="balanced", random_state=42
            )
            clf.fit(scaler.transform(X_train), y_train)
            preds = clf.predict(scaler.transform(X_test))

        acc = (preds == y_test).mean()
        per_fold_acc.append(acc)
        all_preds.extend(preds)
        all_truth.extend(y_test)
        if args.verbose:
            print(f"    Fold {left+1}/{n_trials} acc: {acc*100:.2f}%")

    # 7) Summarize
    mean_loto = np.nanmean(per_fold_acc) if per_fold_acc else np.nan
    overall_acc = np.nanmean(np.array(all_preds) == np.array(all_truth)) \
        if all_preds else np.nan

    return {
        "file": csv_path.name,
        "n_trials": n_trials,
        "mean_loto": mean_loto,
        "overall_bit_acc": overall_acc
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode I-VEP CSVs with LOTO SVC pipeline"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        required=True,
        help="Directory containing CSV files"
    )
    parser.add_argument(
        "--min-trials", "-m",
        type=int,
        default=DEFAULT_MIN_TRIALS,
        help="Minimum trials required to process a file"
    )
    parser.add_argument(
        "--phase", "-p",
        type=int,
        default=DEFAULT_PHASE_VAL,
        help="Phase value to filter (default: 4)"
    )
    parser.add_argument(
        "--enable-lowf", action="store_true",
        help="Enable low-frequency band features"
    )
    parser.add_argument(
        "--enable-flip", action="store_true",
        help="Enable flip-potential features"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-fold accuracies"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f"[ERROR] Data directory not found: {data_dir}")
        sys.exit(1)

    files = sorted(data_dir.glob("*.csv"))
    if not files:
        print(f"[ERROR] No CSV files in {data_dir}")
        sys.exit(1)

    summaries = []
    for idx, path in enumerate(files, 1):
        print(f"\nProcessing file {idx}/{len(files)}: {path.name}")
        result = process_file(path, args)
        if result:
            print(f"  → Trials: {result['n_trials']}, "
                  f"LOTO acc: {result['mean_loto']*100:.2f}%, "
                  f"Overall bit acc: {result['overall_bit_acc']*100:.2f}%")
            summaries.append(result)

    # Aggregate
    if summaries:
        mean_loto = np.nanmean([r["mean_loto"] for r in summaries])
        std_loto  = np.nanstd( [r["mean_loto"] for r in summaries])
        mean_bit  = np.nanmean([r["overall_bit_acc"] for r in summaries])
        std_bit   = np.nanstd( [r["overall_bit_acc"] for r in summaries])

        print("\n=== Aggregate Summary ===")
        print(f"Files processed: {len(summaries)}")
        print(f"Mean LOTO accuracy: {mean_loto*100:.2f}% ± {std_loto*100:.2f}%")
        print(f"Mean overall bit accuracy: {mean_bit*100:.2f}% ± {std_bit*100:.2f}%")

    total = (time.time() - start) / 60
    print(f"\nTotal processing time: {total:.2f} minutes")


if __name__ == "__main__":
    main()
