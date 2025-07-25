#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nested_cv_optuna.py

Author:
    Milán András Fodor (0009-0009-9769-0821)
    Ivan Volosyak (0000-0001-6555-7617)

Description:
    Performs nested cross-validation on EEG data for the
    Imperceptible Visual Evoked Potentials (I-VEP) paradigm
    using Optuna for hyperparameter optimization and
    TensorFlow/Keras for model training.

    - Loads and preprocesses CSV data from a specified directory.
    - Applies causal bandpass filtering and Hilbert transform
      to extract amplitude and phase features.
    - Trains a ResNet-style 1D-CNN with AdamW optimizer.
    - Uses Optuna to optimize filtering, windowing, and model
      hyperparameters via a 4-fold outer CV and inner Optuna loop.
    - Saves best model and hyperparameters per fold.
    - Outputs a summary CSV with accuracy and confusion-matrix stats.

Usage:
    python nested_cv_optuna.py --data-dir /path/to/csvs --results results.csv

Requirements:
    - Python 3.8+
    - numpy, pandas, scipy
    - scikit-learn, optuna
    - tensorflow (2.x), tensorflow-addons
"""

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from scipy.signal import butter, decimate, hilbert, lfilter
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.exceptions import TrialPruned

# ──────────────── Global Constants ─────────────────
SEED = 42
FS_RAW = 1200.0
BIT_SAMPS_RAW = 100
NUM_CLS = 2
PHASE_COL = "HHCVEP_phase_label"
BOX_COL = "HHCVEP_box_label"
MSEQ_COL = "perfect_mseq_label"
BEST_8 = [
    "Gtec_EEG_PO3", "Gtec_EEG_Pz", "Gtec_EEG_POz", "Gtec_EEG_PO4",
    "Gtec_EEG_O1", "Gtec_EEG_Oz", "Gtec_EEG_O2", "Gtec_EEG_Iz"
]
ALL_16 = [
    "Gtec_EEG_P7", "Gtec_EEG_P3", "Gtec_EEG_Pz", "Gtec_EEG_P4",
    "Gtec_EEG_P8", "Gtec_EEG_PO7", "Gtec_EEG_PO3", "Gtec_EEG_POz",
    "Gtec_EEG_PO4", "Gtec_EEG_PO8", "Gtec_EEG_O1", "Gtec_EEG_Oz",
    "Gtec_EEG_O2", "Gtec_EEG_O9", "Gtec_EEG_Iz", "Gtec_EEG_O10"
]

# ───────────── Search Space ───────────────────────
SEARCH_SPACE = dict(
    decim=[5],
    stride_divisor=[1],
    filter_mode=["combined"],
    filter_center_fundamental=[59.8, 60.06, 60.3],
    filter_center_harmonic=[119.6, 121.26, 122.0],
    bp_half_width=[1.5, 2.0, 3.0],
    bp_order_simple=[4, 5],
    label_shift_ms=[40, 60, 80, 89, 100, 110],
    window_samps=[40, 60],
    cnn_blocks=[1],
    num_filters_cnn=[32],
    kernel_time_cnn=[11, 15],
    dense_units=[64],
    dropout=[0.20, 0.25, 0.30],
    lr=[3e-4, 5e-4],
    batch=[64],
    patience=[10, 15],
    use_car=[False, True],
    norm=["none", "zscore"],
    act=["relu"],
    l2=[3e-6, 1e-5],
    channels=["best8", "all16"],
)


def causal_bp_filter(x: np.ndarray, fs: float, lo: float, hi: float, order: int) -> np.ndarray:
    """Apply a causal Butterworth bandpass filter."""
    nyq = 0.5 * fs
    lo, hi = max(lo, 0.1), min(hi, nyq - 0.1)
    if lo >= hi:
        return np.zeros_like(x)
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    y = lfilter(b.astype(np.float32), a.astype(np.float32), x, axis=0)
    return y if np.all(np.isfinite(y)) else np.zeros_like(x)


def make_windows(raw, lbl, hp, fs):
    """Slice the continuous data into (window, label) pairs."""
    window_size = hp["window_samps"]
    bit = BIT_SAMPS_RAW // hp["decim"]
    stride = max(1, bit // hp["stride_divisor"])
    shift = int(hp["label_shift_ms"] / 1000 * fs)

    X, Y = [], []
    for start in range(0, raw.shape[0] - window_size + 1, stride):
        label_idx = start - shift
        if 0 <= label_idx < len(lbl):
            w = raw[start : start + window_size].astype(np.float32)
            if hp["norm"] == "zscore":
                m, s = w.mean(0, keepdims=True), w.std(0, keepdims=True)
                w = (w - m) / np.maximum(s, 1e-7)
            X.append(w)
            Y.append(int(lbl[label_idx]))

    if not X:
        return np.empty((0, window_size, raw.shape[1])), np.empty((0,), dtype=int)
    return np.stack(X), np.array(Y, dtype=int)


def load_files(file_list, hp):
    """Load, filter, preprocess, and window all CSV files."""
    channels = BEST_8 if hp["channels"] == "best8" else ALL_16
    all_X, all_Y = [], []

    for fp in file_list:
        df = pd.read_csv(fp).dropna(subset=channels + [BOX_COL, PHASE_COL, MSEQ_COL])
        df = df[(df[PHASE_COL] == 3) & df[BOX_COL].isin([1, 2, 3])]
        df["blk"] = (df[BOX_COL].diff() != 0).cumsum()

        for blk in df["blk"].unique():
            seg = df[df["blk"] == blk]
            raw = seg[channels].values.astype(np.float32)
            lbl = np.clip(seg[MSEQ_COL].values.astype(int), 0, 1)
            fs = FS_RAW

            if hp["decim"] > 1:
                raw = decimate(raw, hp["decim"], axis=0, zero_phase=False)
                lbl = lbl[:: hp["decim"]]
                fs = FS_RAW / hp["decim"]

            if raw.shape[0] < hp["window_samps"]:
                continue

            # Random circular shift for augmentation
            step = BIT_SAMPS_RAW // hp["decim"]
            offset = random.randrange(step)
            raw = np.roll(raw, offset, axis=0)
            lbl = np.roll(lbl, offset, axis=0)

            # Apply dual-band filtering + CAR + Hilbert
            lo1, hi1 = hp["filter_center_fundamental"] - hp["bp_half_width"], hp["filter_center_fundamental"] + hp["bp_half_width"]
            lo2, hi2 = hp["filter_center_harmonic"] - hp["bp_half_width"], hp["filter_center_harmonic"] + hp["bp_half_width"]

            band1 = causal_bp_filter(raw, fs, lo1, hi1, hp["bp_order_simple"])
            band2 = causal_bp_filter(raw, fs, lo2, hi2, hp["bp_order_simple"])
            sig = np.concatenate([band1, band2], axis=1)

            if hp["use_car"]:
                sig -= sig.mean(1, keepdims=True)

            analytic = hilbert(sig, axis=0)
            features = np.concatenate([np.abs(analytic), np.unwrap(np.angle(analytic), axis=0)], axis=1)

            mask = np.all(np.isfinite(features), axis=1)
            sig, lbl = features[mask], lbl[mask]

            if sig.shape[0] < hp["window_samps"]:
                continue

            Xw, Yw = make_windows(sig, lbl, hp, fs)
            if Xw.size:
                all_X.append(Xw)
                all_Y.append(Yw)

    if not all_X:
        return (np.empty((0, hp["window_samps"], len(channels))), np.empty((0,), int))
    return np.concatenate(all_X), np.concatenate(all_Y)


def build_model(hp, num_channels):
    """Build the ResNet-style 1D-CNN model."""
    inp = layers.Input(shape=(hp["window_samps"], num_channels))
    x = layers.BatchNormalization()(inp)
    act_fn = layers.Activation(hp["act"])
    shortcut = x

    # First Conv block
    x = layers.Conv1D(
        hp["num_filters_cnn"],
        hp["kernel_time_cnn"],
        padding="causal",
        kernel_regularizer=regularizers.l2(hp["l2"])
    )(x)
    x = layers.BatchNormalization()(x)
    x = act_fn(x)

    # Second Conv block
    x = layers.Conv1D(
        hp["num_filters_cnn"],
        hp["kernel_time_cnn"],
        padding="causal",
        kernel_regularizer=regularizers.l2(hp["l2"])
    )(x)
    x = layers.BatchNormalization()(x)

    # Match dimensions for skip connection
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = layers.Conv1D(hp["num_filters_cnn"], 1, padding="causal")(shortcut)

    x = layers.Add()([x, shortcut])
    x = act_fn(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.SpatialDropout1D(hp["dropout"])(x)
    x = layers.Flatten()(x)

    # Dense classifier
    x = layers.Dense(hp["dense_units"], kernel_regularizer=regularizers.l2(hp["l2"]))(x)
    x = act_fn(x)
    x = layers.Dropout(hp["dropout"])(x)
    out = layers.Dense(NUM_CLS, activation="softmax")(x)

    return tf.keras.Model(inputs=inp, outputs=out)


def objective(trial, train_files, val_files):
    """Optuna objective: train on train_files, evaluate on val_files, return val loss."""
    hp = {k: trial.suggest_categorical(k, v) for k, v in SEARCH_SPACE.items()}
    hp.update(
        filter_center_fundamental=hp["filter_center_fundamental"],
        filter_center_harmonic=hp["filter_center_harmonic"],
        bp_half_width=hp["bp_half_width"],
        bp_order_simple=hp["bp_order_simple"],
    )

    X_tr, Y_tr = load_files(train_files, hp)
    X_va, Y_va = load_files(val_files, hp)

    # Prune trials with insufficient data
    if X_tr.size == 0 or X_va.size == 0 or len(np.unique(Y_va)) < 2:
        raise TrialPruned()

    K.clear_session()
    gc.collect()

    model = build_model(hp, X_tr.shape[2])
    optimizer = tfa.optimizers.AdamW(weight_decay=hp["l2"], learning_rate=hp["lr"])
    model.compile(optimizer, "categorical_crossentropy", ["accuracy"])

    class_weights = compute_class_weight("balanced", classes=np.unique(Y_tr), y=Y_tr)
    cw = dict(zip(np.unique(Y_tr), class_weights))

    early_stop = EarlyStopping("val_loss", patience=hp["patience"], restore_best_weights=True)
    history = model.fit(
        X_tr, to_categorical(Y_tr),
        validation_data=(X_va, to_categorical(Y_va)),
        epochs=40,
        batch_size=hp["batch"],
        class_weight=cw,
        callbacks=[early_stop],
        verbose=0
    )

    val_loss = history.history["val_loss"][-1]
    if not np.isfinite(val_loss):
        raise TrialPruned()
    return val_loss


def nested_cv(data_dir: Path, results_csv: Path):
    """Run nested cross-validation and save summary CSV."""
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    files = sorted(data_dir.glob("*.csv"))
    if len(files) < 4:
        raise RuntimeError("Need at least 4 CSV files for nested CV.")

    kf_outer = KFold(n_splits=4, shuffle=True, random_state=SEED)
    summary = []

    for fold, (_, test_idx) in enumerate(kf_outer.split(files), start=1):
        train_val_files = [f for i, f in enumerate(files) if i not in test_idx]
        test_files = [files[i] for i in test_idx]

        random.shuffle(train_val_files)
        n_val = max(1, round(0.2 * len(train_val_files)))
        val_files = train_val_files[:n_val]
        train_files = train_val_files[n_val:]

        print(f"\n=== OUTER fold {fold}/4 ===")
        print(f"  train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")

        db_path = f"fold{fold}.db"
        if Path(db_path).exists():
            Path(db_path).unlink()

        study = optuna.create_study(
            direction="minimize",
            study_name=f"fold{fold}",
            storage=f"sqlite:///{db_path}",
            sampler=TPESampler(seed=SEED),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            load_if_exists=False,
        )
        study.optimize(lambda t: objective(t, train_files, val_files), n_trials=50)

        best_hp = study.best_params
        best_hp.update(
            filter_center_fundamental=best_hp["filter_center_fundamental"],
            filter_center_harmonic=best_hp["filter_center_harmonic"],
            bp_half_width=best_hp["bp_half_width"],
            bp_order_simple=best_hp["bp_order_simple"],
        )

        # Retrain on train+val
        X_tr, Y_tr = load_files(train_files + val_files, best_hp)
        X_te, Y_te = load_files(test_files, best_hp)

        model = build_model(best_hp, X_tr.shape[2])
        optimizer = tfa.optimizers.AdamW(weight_decay=best_hp["l2"], learning_rate=best_hp["lr"])
        model.compile(optimizer, "categorical_crossentropy", ["accuracy"])

        cw = dict(zip(*compute_class_weight("balanced", classes=np.unique(Y_tr), y=Y_tr)))
        model.fit(
            X_tr, to_categorical(Y_tr),
            epochs=40, batch_size=best_hp["batch"],
            class_weight=cw,
            callbacks=[EarlyStopping("loss", patience=5, restore_best_weights=True)],
            verbose=0
        )

        # Evaluate
        preds = model.predict(X_te, batch_size=best_hp["batch"], verbose=0)
        y_pred = np.argmax(preds, axis=1)
        acc = accuracy_score(Y_te, y_pred)
        precision = precision_score(Y_te, y_pred)
        recall = recall_score(Y_te, y_pred)
        f1 = f1_score(Y_te, y_pred)
        auc = roc_auc_score(Y_te, y_pred)
        tn, fp, fn, tp = confusion_matrix(Y_te, y_pred).ravel()

        print(f"  → acc={acc:.3f}, precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}, auc={auc:.3f}")

        model.save(f"fold{fold}_best_model.h5")
        with open(f"fold{fold}_best_hp.json", "w") as fh:
            json.dump(best_hp, fh, indent=2)

        summary.append({
            "fold": fold,
            "windows": len(Y_te),
            "acc": acc, "precision": precision,
            "recall": recall, "f1": f1, "auc": auc,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files
        })

    pd.DataFrame(summary).to_csv(results_csv, index=False)
    print(f"\nSaved summary → {results_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Nested CV with Optuna for I-VEP EEG decoding")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing CSV files")
    parser.add_argument("--results", type=Path, default=Path("results.csv"), help="Output CSV summary")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    nested_cv(args.data_dir, args.results)
