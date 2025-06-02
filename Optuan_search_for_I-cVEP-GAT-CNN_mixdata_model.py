#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ivep_optuna.py

Author: Milan Andras Fodor (@milaniusz)
Project: https://github.com/MILANIUSZ/I-VEP
Optuna search for I-cVEP decoding GAT-CNN model with 4 electrodes only, mixed data
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import PercentilePruner
from optuna.exceptions import TrialPruned
from optuna.trial import TrialState
from scipy.signal import butter, iirnotch, decimate, sosfilt, lfilter
from tensorflow.keras import layers, regularizers, optimizers, backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from spektral.layers import GATConv, GlobalAvgPool

# ─── DEFAULT CONFIG ─────────────────────────────────────────────────────────────

DEFAULT_CSV_DIR      = Path(r"") # Set 
DEFAULT_CSV_PATTERN  = ""*.csv" # set 
DEFAULT_PHASE        = 4
DEFAULT_BOX_IDS      = [1, 2, 3]
TARGET_CHANNELS      = ['Gtec_EEG_Pz','Gtec_EEG_POz','Gtec_EEG_Oz','Gtec_EEG_Iz']
FS_RAW               = 1200.0
TEST_SIZE_RATIO      = 0.2
EPOCHS               = 50
VERBOSE              = 0
NUM_CHANNELS         = len(TARGET_CHANNELS)
NUM_CLASSES          = 2

ORIGINAL_BEST_HP = {
    "decim": 2,
    "segment_initial_discard_s": 0.2,
    "use_notch": True,
    "notch_q": 30,
    "bp_order": 4,
    "band_choice_type": "broad_search",
    "focus_60hz_half_width": 2.0,
    "cnn_trunk_type": "conv1d_global_pool",
    "act": "relu",
    "cnn_blocks": 1,
    "kernel_time": 13,
    "num_filters": 32,
    "gat_layers": 2,
    "gat_heads": 2,
    "gat_ch": 32,
    "opt": "Adam",
    "lr": 0.001,
    "dropout": 0.3,
    "l2": 1e-5,
    "attn_l1": 1e-5,
    "batch": 64,
    "mom": 0.9,
    # bp_low/high set dynamically in objective
}

# ─── SIGNAL PROCESSING ─────────────────────────────────────────────────────────

def apply_causal_bandpass(data, low, high, order, fs):
    nyq = 0.5 * fs
    low_n, high_n = low / nyq, high / nyq
    if not (0 < low_n < high_n < 1.0):
        return data.astype(np.float32)
    sos = butter(order, [low_n, high_n], btype='band', output='sos')
    return sosfilt(sos.astype(np.float32), data, axis=0).astype(np.float32)

def apply_causal_notch(data, notch_freq, q, fs):
    if fs <= 2 * notch_freq:
        return data
    b, a = iirnotch(notch_freq, q, fs=fs)
    return lfilter(b.astype(np.float32), a.astype(np.float32), data, axis=0)

def create_windows(df_segment, hp, base_fs=FS_RAW):
    if df_segment.empty:
        return np.empty((0, hp["window_samps"], NUM_CHANNELS)), np.empty((0,)), base_fs

    raw = df_segment[TARGET_CHANNELS].values.astype(np.float32)
    lbl = df_segment["HHCVEP_mseq_label"].values.astype(int)
    fs_cur = base_fs

    # decimation
    if hp["decim"] > 1 and raw.shape[0] >= hp["decim"] * 10:
        try:
            dec = decimate(raw, q=hp["decim"], axis=0, ftype='iir', zero_phase=False)
            if np.all(np.isfinite(dec)):
                raw = dec
                lbl = lbl[::hp["decim"]]
                fs_cur /= hp["decim"]
        except ValueError:
            pass

    # notch
    if hp["use_notch"] and fs_cur > 100:
        raw = apply_causal_notch(raw, 50.0, hp["notch_q"], fs_cur)

    # bandpass
    if "bp_low" in hp and "bp_high" in hp and hp["bp_low"] < hp["bp_high"]:
        high = min(hp["bp_high"], fs_cur / 2.0 - 0.1)
        if hp["bp_low"] < high:
            raw = apply_causal_bandpass(raw, hp["bp_low"], high, hp["bp_order"], fs_cur)

    # discard initial
    d0 = int(hp["segment_initial_discard_s"] * fs_cur)
    if raw.shape[0] <= d0 + hp["window_samps"]:
        return np.empty((0, hp["window_samps"], NUM_CHANNELS)), np.empty((0,)), fs_cur
    raw, lbl = raw[d0:], lbl[d0:]

    # slide windows
    W, S = hp["window_samps"], max(1, hp["stride_samps"])
    Xs, ys = [], []
    for start in range(0, raw.shape[0] - W + 1, S):
        Xs.append(raw[start:start+W, :])
        ys.append(lbl[start])
    if not Xs:
        return np.empty((0, W, NUM_CHANNELS)), np.empty((0,)), fs_cur
    return np.stack(Xs), np.array(ys), fs_cur

# ─── MODEL BUILDING ────────────────────────────────────────────────────────────

def build_model(hp, window_samps, N_nodes):
    X_in = layers.Input((window_samps, N_nodes), name="X_input")
    A_in = layers.Input((N_nodes, N_nodes), name="A_input")

    # CNN trunk
    if hp["cnn_trunk_type"] == "conv2d_trick":
        x = layers.Reshape((window_samps, N_nodes, 1))(X_in)
        x = layers.Permute((2,1,3))(x)
        for _ in range(hp["cnn_blocks"]):
            x = layers.Conv2D(hp["num_filters"], (1, hp["kernel_time"]),
                              padding="same", kernel_regularizer=regularizers.l2(hp["l2"]))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(hp["act"])(x)
            x = layers.MaxPooling2D((1,2), padding="same")(x)
            x = layers.Dropout(hp["dropout"])(x)
        shp = K.int_shape(x)
        if shp[2] == 0:
            raise TrialPruned("Time dim zero after conv2d_trick pooling")
        x = layers.Reshape((shp[1], shp[2]*shp[3]))(x)

    else:  # conv1d_global_pool
        outs = []
        for i in range(N_nodes):
            c = layers.Lambda(lambda z: z[:,:,i:i+1])(X_in)
            for _ in range(hp["cnn_blocks"]):
                c = layers.Conv1D(hp["num_filters"], hp["kernel_time"],
                                  padding="same", kernel_regularizer=regularizers.l2(hp["l2"]))(c)
                c = layers.BatchNormalization()(c)
                c = layers.Activation(hp["act"])(c)
                c = layers.MaxPooling1D(2, padding="same")(c)
                c = layers.Dropout(hp["dropout"])(c)
            if K.int_shape(c)[1] == 0:
                raise TrialPruned("Time dim zero after conv1d_global_pool")
            c = layers.GlobalAveragePooling1D()(c)
            outs.append(c)
        x = layers.Stacked() if False else layers.Concatenate()(outs)  # hacky—Concatenate merges features
        x = layers.Reshape((N_nodes, -1))(x)

    # GAT layers
    for _ in range(hp["gat_layers"]):
        x = GATConv(channels=hp["gat_ch"],
                    attn_heads=hp["gat_heads"],
                    concat_heads=True,
                    dropout_rate=hp["dropout"],
                    activation=hp["act"],
                    kernel_regularizer=regularizers.l2(hp["l2"]),
                    attn_kernel_regularizer=regularizers.l1(hp["attn_l1"])
                   )([x, A_in])

    # global pooling + classifier
    x = GlobalAvgPool()(x)
    x = layers.Dropout(hp["dropout"])(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model([X_in, A_in], out)
    if hp["opt"] == "RMSprop":
        opt = optimizers.RMSprop(lr=hp["lr"], momentum=hp["mom"])
    else:
        opt = optimizers.Adam(lr=hp["lr"], beta_1=hp["mom"])
    model.compile(opt, "categorical_crossentropy", ["accuracy"])
    return model

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_and_filter_data(csv_dir, pattern, phase, box_ids, logger):
    files = sorted(glob.glob(str(csv_dir / pattern)))
    if not files:
        logger.critical(f"No CSVs match {pattern} in {csv_dir}")
        sys.exit(1)
    logger.info(f"Loading {len(files)} CSV files")

    dfs = []
    for fp in files:
        logger.info(f"  Reading {os.path.basename(fp)}")
        df = pd.read_csv(fp, skiprows=1, low_memory=False, encoding="latin1")
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    # ensure numeric
    for col in TARGET_CHANNELS + ["HHCVEP_mseq_label", "HHCVEP_phase_label", "HHCVEP_box_label"]:
        if col not in full.columns:
            logger.critical(f"Missing column {col}")
            sys.exit(1)
        full[col] = pd.to_numeric(full[col], errors="coerce")
    full.dropna(subset=TARGET_CHANNELS + ["HHCVEP_mseq_label", "HHCVEP_phase_label", "HHCVEP_box_label"], inplace=True)

    phase_df = full[full["HHCVEP_phase_label"] == phase].copy()
    if phase_df.empty:
        logger.critical(f"No data for phase {phase}")
        sys.exit(1)
    # split per box
    splits = {box: phase_df[phase_df["HHCVEP_box_label"] == box].reset_index(drop=True) for box in box_ids}
    return splits

# ─── OPTUNA OBJECTIVE ─────────────────────────────────────────────────────────

def objective(trial, data_splits, logger):
    K.clear_session()
    hp = ORIGINAL_BEST_HP.copy()

    # sample hps
    hp.update({
        "decim":        trial.suggest_categorical("decim", [1, 2, 4]),
        "segment_initial_discard_s": trial.suggest_float("segment_initial_discard_s", 0.05, 0.5),
        "use_notch":    trial.suggest_categorical("use_notch", [True, False]),
        "notch_q":      trial.suggest_float("notch_q", 10, 50),
        "bp_order":     trial.suggest_int("bp_order", 2, 6),
        "cnn_trunk_type": trial.suggest_categorical("cnn_trunk_type", ["conv2d_trick", "conv1d_global_pool"]),
        "act":          trial.suggest_categorical("act", ["relu", "elu", "selu", "tanh"]),
        "cnn_blocks":   trial.suggest_int("cnn_blocks", 1, 3),
        "kernel_time":  trial.suggest_categorical("kernel_time", [3,5,7,9,11,13,15]),
        "num_filters":  trial.suggest_categorical("num_filters", [8,16,32,48,64]),
        "gat_layers":   trial.suggest_int("gat_layers", 1, 4),
        "gat_heads":    trial.suggest_int("gat_heads", 1, 6),
        "gat_ch":       trial.suggest_categorical("gat_ch", [8,16,32,64]),
        "opt":          trial.suggest_categorical("opt", ["Adam", "RMSprop"]),
        "lr":           trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "dropout":      trial.suggest_float("dropout", 0.0, 0.7),
        "l2":           trial.suggest_float("l2", 1e-8, 1e-3, log=True),
        "attn_l1":      trial.suggest_float("attn_l1", 1e-7, 1e-2, log=True),
        "batch":        trial.suggest_categorical("batch", [32,64,128]),
        "mom":          trial.suggest_float("mom", 0.8, 0.99),
    })

    # bandpass choices
    fs_dec = FS_RAW / hp["decim"]
    choice = trial.suggest_categorical("band_choice_type", ["broad_search"]*4 + ["focus_60hz"])
    if choice == "focus_60hz":
        hw = trial.suggest_categorical("focus_60hz_half_width", [1.0,2.0,3.0,4.0,5.0])
        low, high = 60-hw, 60+hw
        if low<=0 or high>=fs_dec/2: raise TrialPruned("Invalid 60Hz focus band")
        hp["bp_low"], hp["bp_high"] = low, high
    else:
        max_h = min(fs_dec/2-0.01, 120.0)
        MINW=0.5
        low = trial.suggest_float("bp_low", 0.1, max_h-MINW, log=True)
        high = trial.suggest_float("bp_high", low+MINW, max_h, log=True)
        hp["bp_low"], hp["bp_high"] = low, high

    # window & stride
    min_w = max(10, hp["kernel_time"]*(2**hp["cnn_blocks"]), hp["kernel_time"]+1)
    hp["window_samps"] = trial.suggest_int("window_samps", min_w, int(fs_dec*0.75))
    hp["stride_samps"] = trial.suggest_int("stride_samps", 1, int(hp["window_samps"]*0.75))

    # split data
    train_dfs, val_dfs = [], []
    for box, df in data_splits.items():
        if df.empty: continue
        n = len(df)
        cutoff = int(n*(1-TEST_SIZE_RATIO))
        train_dfs.append(df.iloc[:cutoff])
        val_dfs.append(df.iloc[cutoff:])
    if not train_dfs:
        raise TrialPruned("No train data")
    df_tr = pd.concat(train_dfs, ignore_index=True)
    df_va = pd.concat(val_dfs,   ignore_index=True) if val_dfs else pd.DataFrame()

    # make windows
    X_tr, y_tr, fs_tr = create_windows(df_tr, hp)
    if X_tr.size==0: raise TrialPruned("No train windows")
    if df_va.empty:
        # fallback split from train
        n_val = max(1, int(len(X_tr)*TEST_SIZE_RATIO))
        X_va, y_va = X_tr[-n_val:], y_tr[-n_val:]
        X_tr, y_tr = X_tr[:-n_val], y_tr[:-n_val]
    else:
        X_va, y_va, _ = create_windows(df_va, hp)

    if X_va.size==0 or X_tr.shape[0]<hp["batch"]:
        raise TrialPruned("Insufficient train/val windows")

    # one-hot
    y_tr_cat = to_categorical(y_tr, NUM_CLASSES)
    y_va_cat = to_categorical(y_va, NUM_CLASSES)

    # adjacency
    A_tr = np.ones((len(X_tr), NUM_CHANNELS, NUM_CHANNELS), dtype=np.float32)
    A_va = np.ones((len(X_va), NUM_CHANNELS, NUM_CHANNELS), dtype=np.float32)

    # train
    K.clear_session()
    model = build_model(hp, X_tr.shape[1], NUM_CHANNELS)
    es = EarlyStopping("val_loss", patience=10, restore_best_weights=True, verbose=VERBOSE)
    hist = model.fit([X_tr, A_tr], y_tr_cat,
                     validation_data=([X_va, A_va], y_va_cat),
                     batch_size=hp["batch"], epochs=EPOCHS,
                     callbacks=[es], verbose=VERBOSE)
    vl = min(hist.history.get("val_loss", [np.inf]))
    va = max(hist.history.get("val_accuracy", [0.0]))
    if np.isnan(vl) or np.isinf(vl):
        raise TrialPruned("Bad val loss")
    trial.set_user_attr("mean_val_accuracy", float(va))
    return float(vl)

# ─── TRIAL SUMMARY CALLBACK ───────────────────────────────────────────────────

def print_summary(study, trial):
    if trial.state != TrialState.COMPLETE:
        reason = "pruned" if trial.state==TrialState.PRUNED else "failed"
        print(f"[T{trial.number}] {trial.state.name}.")
        return
    vl, va = trial.value, trial.user_attrs.get("mean_val_accuracy", 0.0)
    print(f"[T{trial.number}] loss={vl:.4f}, acc={va:.4f}")
    b = study.best_trial
    if b.number == trial.number:
        print("  ✨ new best!")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="I-VEP GAT-CNN Optuna Search")
    p.add_argument("--csv-dir",    type=Path, default=DEFAULT_CSV_DIR,
                   help="Folder with I-VEP CSVs")
    p.add_argument("--pattern",    type=str,  default=DEFAULT_CSV_PATTERN,
                   help="Filename glob pattern")
    p.add_argument("--phase",      type=int,  default=DEFAULT_PHASE,
                   help="HHCVEP_phase_label to select")
    p.add_argument("--boxes",      type=int,  nargs="+", default=DEFAULT_BOX_IDS,
                   help="HHCVEP_box_label values")
    p.add_argument("--trials",     type=int,  default=2000,
                   help="Number of Optuna trials")
    p.add_argument("--study-name", type=str,  default="ivep_gatcnn_optuna",
                   help="Optuna study name")
    p.add_argument("--db-path",    type=str,  default=None,
                   help="SQLite DB path (default: <study-name>.db)")
    args = p.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("ivep_optuna")

    # GPU setup
    print(f"TF version: {tf.__version__}")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU:", gpus)
    else:
        print("No GPU detected, using CPU")

    # load data
    splits = load_and_filter_data(args.csv_dir, args.pattern, args.phase, args.boxes, logger)

    # Optuna study
    db = args.db_path or f"{args.study_name}.db"
    sampler = TPESampler(seed=42, n_startup_trials=25)
    pruner  = PercentilePruner(percentile=33.0, n_warmup_steps=12, n_min_trials=7)
    study = optuna.create_study(direction="minimize",
                                sampler=sampler,
                                pruner=pruner,
                                study_name=args.study_name,
                                storage=f"sqlite:///{db}",
                                load_if_exists=True)
    study.set_user_attr("CSV_PATTERN", args.pattern)
    study.set_user_attr("TARGET_CHANNELS", TARGET_CHANNELS)
    print(f"Starting study '{args.study_name}' with DB '{db}'")

    # optimize
    study.optimize(lambda t: objective(t, splits, logger),
                   n_trials=args.trials,
                   show_progress_bar=True,
                   callbacks=[print_summary])

    # save best
    if study.best_trial:
        best = study.best_trial
        print(f"\n=== Best Trial #{best.number} ===")
        print(f"Loss: {best.value:.4f}, Acc: {best.user_attrs['mean_val_accuracy']:.4f}")
        print("Params:", json.dumps(best.params, indent=2))
        out = f"{args.study_name}_best_params.json"
        with open(out, "w") as f:
            json.dump(best.params, f, indent=2)
        print(f"Saved best params → {out}")

if __name__ == "__main__":
    main()
