#!/usr/bin/env python3
"""


Author: Milan Andras Fodor (@milaniusz)
Project: https://github.com/MILANIUSZ/I-VEP
Optuna search for I-cVEP decoding GAT-CNN model with 4 electrodes only, training per subejct
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import PercentilePruner
from optuna.exceptions import TrialPruned
from optuna.trial import TrialState
from scipy.signal import butter, lfilter, iirnotch, decimate, sosfilt
from tensorflow.keras import layers, regularizers, optimizers, backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from spektral.layers import GATConv, GlobalAvgPool

# --- USER CONFIG ---
CSV_DIR_PATH = r"C:\Dev\GNN IDEA\data\goodforoptuna\temp\onlyIVEPcsvs" 
CSV_PATTERN = os.path.join(CSV_DIR_PATH, "2025-02*.csv") 
NUM_CSVS_TO_USE = 4 

CURRENT_PHASE_VAL = 4
BOX_COL_NAME_IN_CSV="HHCVEP_box_label"
MSEQ_COL="HHCVEP_mseq_label"
EEG_PHASE_COL = "HHCVEP_phase_label"

TARGET_CHANNELS = ['Gtec_EEG_Pz', 'Gtec_EEG_POz', 'Gtec_EEG_Oz', 'Gtec_EEG_Iz']
ALL_BOX_IDS = [1, 2, 3] 

FS_RAW = 1200.0

# --- Fixed defaults (can be tuned by Optuna) ---
ORIGINAL_BEST_HP = { 
    "decim": 4, "segment_initial_discard_s": 0.48,
    "use_notch": False, "notch_q": 45.0,
    "bp_order": 5, "cnn_trunk_type": "conv2d_trick", "act": "selu",
    "cnn_blocks": 3, "kernel_time": 15, "num_filters": 8,
    "gat_layers": 1, "gat_heads": 3, "gat_ch": 32,
    "opt": "RMSprop", "lr": 0.0016, "dropout": 0.17, 
    "l2": 3.6e-8, "attn_l1": 2.7e-6, "batch": 32, "mom": 0.93,
}

# --- Files & settings (Optuna context) ---
TEST_SIZE_RATIO = 0.2 
EPOCHS     = 50       
VERBOSE    = 0        

NUM_CHANNELS = len(TARGET_CHANNELS)
NUM_CLASSES  = 2 

# --- Signal preprocessing (apply_causal_bandpass, apply_causal_notch, create_windows_for_ivep) ---
# These functions remain UNCHANGED.
def apply_causal_bandpass(data, low, high, order, fs):
    nyq = 0.5*fs; low_n, high_n = low/nyq, high/nyq
    if not (0 < low_n < high_n < 1.0): return data.astype(np.float32) 
    sos_coeffs = butter(order, [low_n,high_n], btype='band', output='sos')
    return sosfilt(sos_coeffs.astype(np.float32), data, axis=0).astype(np.float32)

def apply_causal_notch(data, notch_freq, q, fs):
    if fs <= 2 * notch_freq: return data
    b,a = iirnotch(notch_freq,q,fs=fs)
    return lfilter(b.astype(np.float32), a.astype(np.float32), data, axis=0)

def create_windows_for_ivep(df_continuous_segment, hp, base_fs=FS_RAW): # Takes a continuous segment now
    # df_continuous_segment is ALREADY preprocessed (decim, notch, bp, initial_discard)
    # It only contains the raw channel data and mseq labels for windowing.
    if df_continuous_segment.empty or df_continuous_segment.shape[0] < hp["window_samps"]:
        return np.empty((0, hp["window_samps"], NUM_CHANNELS)), np.empty((0,)), base_fs # fs here is after decimation

    raw_processed = df_continuous_segment[TARGET_CHANNELS].values.astype(np.float32)
    lbl_processed = df_continuous_segment[MSEQ_COL].values.astype(int)
    
    Xs, ys = [], []
    current_stride_samps = max(1, hp["stride_samps"]) 
    for start in range(0, raw_processed.shape[0] - hp["window_samps"] + 1, current_stride_samps):
        Xs.append(raw_processed[start : start + hp["window_samps"], :])
        ys.append(lbl_processed[start]) # Label at the start of the window
    
    return (np.stack(Xs), np.array(ys)) if Xs else (np.empty((0, hp["window_samps"], NUM_CHANNELS)), np.empty((0,)))


# --- Model builder (build_model) ---
# This function remains UNCHANGED.
def build_model(hp, window_samps_eff, N_nodes): 
    X_in = layers.Input((window_samps_eff, N_nodes), name="X_input")
    A_in = layers.Input((N_nodes, N_nodes), name="Adj_input") 
    x_cnn_input = X_in
    if hp["cnn_trunk_type"] == "conv2d_trick":
        current_x = layers.Reshape((window_samps_eff, N_nodes, 1))(x_cnn_input) 
        current_x = layers.Permute((2, 1, 3))(current_x) 
        for _ in range(hp["cnn_blocks"]):
            current_x = layers.Conv2D(hp["num_filters"], (1, hp["kernel_time"]), padding="same", kernel_regularizer=regularizers.l2(hp["l2"]))(current_x)
            current_x = layers.BatchNormalization()(current_x)
            current_x = layers.Activation(hp.get("act", "relu"))(current_x)
            current_x = layers.MaxPooling2D((1, 2), padding="same")(current_x) 
            current_x = layers.Dropout(hp["dropout"])(current_x)
        shp = K.int_shape(current_x) 
        if shp[2] == 0: raise TrialPruned(f"CNN (conv2d_trick) Maxpooling reduced time dim to 0.")
        x_gat_input = layers.Reshape((shp[1], shp[2] * shp[3]))(current_x)
    elif hp["cnn_trunk_type"] == "conv1d_global_pool":
        cnn_outputs = []
        for i in range(N_nodes): 
            channel_slice = layers.Lambda(lambda z: z[:, :, i:i+1])(x_cnn_input) 
            c = channel_slice
            for _ in range(hp["cnn_blocks"]):
                c = layers.Conv1D(hp["num_filters"], hp["kernel_time"], padding="same", kernel_regularizer=regularizers.l2(hp["l2"]))(c)
                c = layers.BatchNormalization()(c)
                c = layers.Activation(hp.get("act", "relu"))(c)
                c = layers.MaxPooling1D(pool_size=2, padding="same")(c)
                c = layers.Dropout(hp["dropout"])(c)
            if K.int_shape(c)[1] == 0: raise TrialPruned(f"CNN (conv1d_global_pool) Maxpooling reduced time dim to 0 for channel {i}.")
            c = layers.GlobalAveragePooling1D()(c) 
            cnn_outputs.append(c)
        if not cnn_outputs and N_nodes > 0 : raise TrialPruned("CNN (conv1d_global_pool) empty cnn_outputs.")
        elif N_nodes == 0: raise TrialPruned("N_nodes is 0.")
        x_gat_input = layers.concatenate([layers.Reshape((1, K.int_shape(co)[-1]))(co) for co in cnn_outputs], axis=1)
    else: raise ValueError(f"Unknown cnn_trunk_type: {hp['cnn_trunk_type']}")
    gat_processed_x = x_gat_input
    for _ in range(hp["gat_layers"]):
        gat_processed_x = GATConv(channels=hp["gat_ch"], attn_heads=hp["gat_heads"], concat_heads=True, 
                                  dropout_rate=hp["dropout"], activation=hp.get("act", "relu"),
                                  kernel_regularizer=regularizers.l2(hp["l2"]), 
                                  attn_kernel_regularizer=regularizers.l1(hp["attn_l1"])
                                 )([gat_processed_x, A_in])
    x_pool = GlobalAvgPool()(gat_processed_x) 
    x_pool = layers.Dropout(hp["dropout"])(x_pool)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x_pool)
    if hp["opt"] == "Adam": opt = optimizers.Adam(learning_rate=hp["lr"], beta_1=hp["mom"])
    elif hp["opt"] == "RMSprop": opt = optimizers.RMSprop(learning_rate=hp["lr"], momentum=hp["mom"])
    else: opt = optimizers.Adam(learning_rate=hp["lr"], beta_1=hp["mom"])
    model = tf.keras.Model([X_in, A_in], out)
    model.compile(opt, "categorical_crossentropy", ["accuracy"])
    return model

# --- Global List of CSV Files ---
ALL_SUBJECT_FILENAMES = sorted(glob.glob(CSV_PATTERN))
if not ALL_SUBJECT_FILENAMES:
    print(f"CRITICAL: No CSV files found. Exiting."); exit(1)
if len(ALL_SUBJECT_FILENAMES) < NUM_CSVS_TO_USE:
    print(f"CRITICAL: Found {len(ALL_SUBJECT_FILENAMES)} CSVs, but NUM_CSVS_TO_USE is {NUM_CSVS_TO_USE}. Exiting."); exit(1)
print(f"Found {len(ALL_SUBJECT_FILENAMES)} CSVs. Will use first {NUM_CSVS_TO_USE} for each Optuna trial: {ALL_SUBJECT_FILENAMES[:NUM_CSVS_TO_USE]}")


# ─── Optuna objective (MODIFIED FOR CORRECT PER-CSV SPLIT BEFORE WINDOWING) ─────────────────
def objective(trial):
    K.clear_session() 
    hp = {} 

    # --- Suggest hyperparameters (using constrained names for Optuna trial params) ---
    hp["decim"] = trial.suggest_categorical("decim_v9", [1, 2, 4])
    hp["segment_initial_discard_s"] = trial.suggest_float("segment_initial_discard_s_v9", 0.05, 0.5)
    hp["use_notch"] = trial.suggest_categorical("use_notch_v9", [True, False])
    hp["notch_q"] = trial.suggest_float("notch_q_v9", 10, 50)
    hp["bp_order"] = trial.suggest_int("bp_order_v9", 2, 6)  
    hp["cnn_trunk_type"] = trial.suggest_categorical("cnn_trunk_type_v9", ["conv2d_trick", "conv1d_global_pool"])
    hp["act"] = trial.suggest_categorical("act_v9", ["relu", "elu", "selu", "tanh"])
    hp["cnn_blocks"] = trial.suggest_int("cnn_blocks_v9", 1, 3)
    hp["kernel_time"] = trial.suggest_categorical("kernel_time_v9", [3, 5, 7, 9, 11, 13, 15]) 
    hp["num_filters"] = trial.suggest_categorical("num_filters_v9", [8, 16, 32, 48, 64])
    hp["gat_layers"] = trial.suggest_int("gat_layers_v9", 1, 4)
    hp["gat_heads"] = trial.suggest_int("gat_heads_v9", 1, 6)
    hp["gat_ch"] = trial.suggest_categorical("gat_ch_v9", [8, 16, 32, 64])
    hp["opt"] = trial.suggest_categorical("opt_v9", ["Adam", "RMSprop"])
    hp["lr"] = trial.suggest_float("lr_v9", 1e-5, 1e-2, log=True)
    hp["dropout"] = trial.suggest_float("dropout_v9", 0.0, 0.7)
    hp["l2"] = trial.suggest_float("l2_v9", 1e-8, 1e-3, log=True)
    hp["attn_l1"] = trial.suggest_float("attn_l1_v9", 1e-7, 1e-2, log=True) 
    hp["batch"] = trial.suggest_categorical("batch_v9", [32, 64, 128])
    hp["mom"] = trial.suggest_float("mom_v9", 0.8, 0.99) 
    
    fs_after_decim = FS_RAW / hp["decim"]
    band_choice_list = ["broad_search_constrained"] * 4 + ["focus_60hz"] 
    hp["band_choice_type"] = trial.suggest_categorical("band_choice_type_v9", band_choice_list) 

    if hp["band_choice_type"] == "focus_60hz":
        hp["focus_60hz_half_width"] = trial.suggest_categorical("focus_60hz_half_width_v9", [1.0, 2.0, 3.0, 4.0, 5.0])
        hp["bp_low"] = 60.0 - hp["focus_60hz_half_width"]
        hp["bp_high"] = 60.0 + hp["focus_60hz_half_width"]
        if hp["bp_high"] >= fs_after_decim / 2.0 - 0.01 or hp["bp_low"] <= 0:
            raise TrialPruned(f"Focused 60Hz band invalid for fs_decim={fs_after_decim}Hz.")
    elif hp["band_choice_type"] == "broad_search_constrained":
        hp["bp_low"] = trial.suggest_float("bp_low_v9", 0.1, 8.0, log=False) 
        min_allowable_bp_high = 25.0
        max_allowable_bp_high = fs_after_decim / 2.0 - 0.1 
        if min_allowable_bp_high >= max_allowable_bp_high: 
            raise TrialPruned(f"Cannot set bp_high (min 25Hz), Nyquist is {fs_after_decim / 2.0}Hz.")
        actual_min_bp_high = max(min_allowable_bp_high, hp["bp_low"] + 0.5) 
        if actual_min_bp_high >= max_allowable_bp_high:
            if max_allowable_bp_high > hp["bp_low"] + 0.1 : hp["bp_high"] = max_allowable_bp_high 
            else: raise TrialPruned(f"Cannot set bp_high. bp_low={hp['bp_low']:.2f}, min_high={actual_min_bp_high:.2f}, Nyq_lim={max_allowable_bp_high:.2f}")
        else:
            hp["bp_high"] = trial.suggest_float("bp_high_v9", actual_min_bp_high, max_allowable_bp_high, log=True)

    if "bp_low" not in hp or "bp_high" not in hp or hp["bp_low"] >= hp["bp_high"] or hp["bp_low"] <=0:
        raise TrialPruned(f"Invalid bandpass params: Low={hp.get('bp_low')}, High={hp.get('bp_high')}")

    min_window_duration_s = 0.010; max_window_duration_s = 0.300 
    min_window_samps_abs = int(min_window_duration_s * fs_after_decim)
    max_window_samps_abs = int(max_window_duration_s * fs_after_decim)
    min_required_samps_for_cnn = hp["kernel_time"] + (2**hp["cnn_blocks"]) # Heuristic, build_model will catch if too small
    actual_min_window_samps = max(min_window_samps_abs, min_required_samps_for_cnn, 10) 
    if actual_min_window_samps >= max_window_samps_abs:
        if max_window_samps_abs >= min_required_samps_for_cnn : hp["window_samps"] = max_window_samps_abs 
        else: raise TrialPruned(f"Max window dur ({max_window_duration_s*1000}ms) too small for CNN (needs ~{min_required_samps_for_cnn} samps) at fs={fs_after_decim}Hz.")
    else:
        hp["window_samps"] = trial.suggest_int("window_samps_v9", actual_min_window_samps, max_window_samps_abs)

    min_stride_samps = 1; max_stride_samps = max(1, int(hp["window_samps"] * 0.5)) 
    if min_stride_samps >= max_stride_samps: hp["stride_samps"] = min_stride_samps
    else: hp["stride_samps"] = trial.suggest_int("stride_samps_v9", min_stride_samps, max_stride_samps)
    
    csv_val_losses = []
    csv_val_accuracies = []
    files_to_process_this_trial = ALL_SUBJECT_FILENAMES[:NUM_CSVS_TO_USE]

    for csv_idx, csv_file_path in enumerate(files_to_process_this_trial):
        # print(f"  Trial {trial.number}, CSV {csv_idx+1}/{NUM_CSVS_TO_USE}: {os.path.basename(csv_file_path)}") # Optional print
        try:
            df_single_csv = pd.read_csv(csv_file_path, skiprows=1, low_memory=False, encoding='latin1')
            critical_cols_single_csv = TARGET_CHANNELS + [MSEQ_COL, EEG_PHASE_COL, BOX_COL_NAME_IN_CSV]
            for col in critical_cols_single_csv:
                if col in df_single_csv.columns: df_single_csv[col] = pd.to_numeric(df_single_csv[col], errors='coerce')
                else: raise ValueError(f"Crit col '{col}' missing in {os.path.basename(csv_file_path)}")
            df_single_csv = df_single_csv.dropna(subset=critical_cols_single_csv)
            df_phase_this_csv = df_single_csv[df_single_csv[EEG_PHASE_COL] == CURRENT_PHASE_VAL].copy()
            if df_phase_this_csv.empty: continue

            # --- CORRECTED SPLITTING: Raw continuous segments for train/val PER CSV ---
            per_box_raw_train_segments = []
            per_box_raw_val_segments = []

            for box_id in ALL_BOX_IDS:
                df_box_continuous = df_phase_this_csv[df_phase_this_csv[BOX_COL_NAME_IN_CSV] == box_id].copy()
                df_box_continuous.dropna(subset=TARGET_CHANNELS + [MSEQ_COL], inplace=True)
                df_box_continuous.reset_index(drop=True, inplace=True)
                if df_box_continuous.empty or len(df_box_continuous) < 20: # Need some minimum length for a meaningful split
                    if not df_box_continuous.empty: per_box_raw_train_segments.append(df_box_continuous) # Add small segments to train
                    continue

                n_total_samples_in_box_csv = len(df_box_continuous)
                split_idx_csv = int(n_total_samples_in_box_csv * (1 - TEST_SIZE_RATIO))
                
                # Ensure split_idx allows for some validation data if possible
                if split_idx_csv >= n_total_samples_in_box_csv - 5 : # e.g. need at least 5 samples for val
                    split_idx_csv = n_total_samples_in_box_csv - 5 
                if split_idx_csv <= 5: # e.g. need at least 5 samples for train
                    if not df_box_continuous.empty: per_box_raw_train_segments.append(df_box_continuous)
                    continue


                df_train_box_part_csv = df_box_continuous.iloc[:split_idx_csv]
                df_val_box_part_csv = df_box_continuous.iloc[split_idx_csv:] # This is the raw, continuous val part
                
                if not df_train_box_part_csv.empty: per_box_raw_train_segments.append(df_train_box_part_csv)
                if not df_val_box_part_csv.empty: per_box_raw_val_segments.append(df_val_box_part_csv)

            if not per_box_raw_train_segments: continue # No training data for this CSV
            
            df_train_raw_combined_csv = pd.concat(per_box_raw_train_segments, ignore_index=True)
            df_val_raw_combined_csv = pd.concat(per_box_raw_val_segments, ignore_index=True) if per_box_raw_val_segments else pd.DataFrame()

            if df_train_raw_combined_csv.empty: continue

            # --- Preprocess continuous segments THEN window ---
            # Training Data
            processed_train_data_list = []
            fs_after_decim_train = FS_RAW / hp["decim"] # Store fs for this segment

            # Apply decimation, notch, bandpass, initial discard to the raw continuous training data
            raw_train_for_proc = df_train_raw_combined_csv[TARGET_CHANNELS].values.astype(np.float32)
            labels_train_for_proc = df_train_raw_combined_csv[MSEQ_COL].values.astype(int)
            
            if hp["decim"] > 1:
                if raw_train_for_proc.shape[0] >= hp["decim"] * 5: # Min samples for decimation
                    try:
                        cutoff_d = (FS_RAW / hp["decim"]) * 0.4
                        if cutoff_d * 2 < FS_RAW:
                            sos_d = butter(5, cutoff_d, btype='low', fs=FS_RAW, output='sos')
                            raw_train_for_proc_lpf = sosfilt(sos_d, raw_train_for_proc, axis=0)
                            raw_train_for_proc = raw_train_for_proc_lpf[::hp["decim"], :]
                            labels_train_for_proc = labels_train_for_proc[::hp["decim"]]
                        # fs_after_decim_train is already set
                    except ValueError: pass # Keep original if decimation fails
                else: fs_after_decim_train = FS_RAW # No decimation actually applied
            else: fs_after_decim_train = FS_RAW


            if hp["use_notch"] and fs_after_decim_train > 100:
                raw_train_for_proc = apply_causal_notch(raw_train_for_proc, 50.0, hp["notch_q"], fs_after_decim_train)
            if hp["bp_low"] < hp["bp_high"]:
                actual_bp_high_tr = min(hp["bp_high"], fs_after_decim_train / 2.0 - 0.1)
                if hp["bp_low"] < actual_bp_high_tr:
                    raw_train_for_proc = apply_causal_bandpass(raw_train_for_proc, hp["bp_low"], actual_bp_high_tr, hp["bp_order"], fs_after_decim_train)
            
            discard_samps_train = int(hp["segment_initial_discard_s"] * fs_after_decim_train)
            if raw_train_for_proc.shape[0] > discard_samps_train:
                raw_train_for_proc = raw_train_for_proc[discard_samps_train:]
                labels_train_for_proc = labels_train_for_proc[discard_samps_train:]
            else: continue # Not enough data after discard

            # Now create a temporary DataFrame from processed continuous data for windowing
            df_processed_train_for_windowing = pd.DataFrame(raw_train_for_proc, columns=TARGET_CHANNELS)
            df_processed_train_for_windowing[MSEQ_COL] = labels_train_for_proc
            
            X_tr_csv, y_tr_labels_csv = create_windows_for_ivep(df_processed_train_for_windowing, hp) # fs passed via hp effectively
            if X_tr_csv.size == 0 or X_tr_csv.shape[0] == 0: continue

            # Validation Data (similar processing)
            X_va_csv, y_va_labels_csv = np.empty((0, hp["window_samps"], NUM_CHANNELS)), np.empty((0,))
            if not df_val_raw_combined_csv.empty:
                raw_val_for_proc = df_val_raw_combined_csv[TARGET_CHANNELS].values.astype(np.float32)
                labels_val_for_proc = df_val_raw_combined_csv[MSEQ_COL].values.astype(int)
                fs_after_decim_val = FS_RAW / hp["decim"]

                if hp["decim"] > 1:
                    if raw_val_for_proc.shape[0] >= hp["decim"] * 5:
                        try:
                            cutoff_d_val = (FS_RAW / hp["decim"]) * 0.4
                            if cutoff_d_val * 2 < FS_RAW:
                                sos_d_val = butter(5, cutoff_d_val, btype='low', fs=FS_RAW, output='sos')
                                raw_val_for_proc_lpf = sosfilt(sos_d_val, raw_val_for_proc, axis=0)
                                raw_val_for_proc = raw_val_for_proc_lpf[::hp["decim"], :]
                                labels_val_for_proc = labels_val_for_proc[::hp["decim"]]
                            # fs_after_decim_val is already set
                        except ValueError: pass
                    else: fs_after_decim_val = FS_RAW
                else: fs_after_decim_val = FS_RAW
                
                if hp["use_notch"] and fs_after_decim_val > 100:
                    raw_val_for_proc = apply_causal_notch(raw_val_for_proc, 50.0, hp["notch_q"], fs_after_decim_val)
                if hp["bp_low"] < hp["bp_high"]:
                    actual_bp_high_val = min(hp["bp_high"], fs_after_decim_val / 2.0 - 0.1)
                    if hp["bp_low"] < actual_bp_high_val:
                         raw_val_for_proc = apply_causal_bandpass(raw_val_for_proc, hp["bp_low"], actual_bp_high_val, hp["bp_order"], fs_after_decim_val)
                
                discard_samps_val = int(hp["segment_initial_discard_s"] * fs_after_decim_val)
                if raw_val_for_proc.shape[0] > discard_samps_val:
                    raw_val_for_proc = raw_val_for_proc[discard_samps_val:]
                    labels_val_for_proc = labels_val_for_proc[discard_samps_val:]
                else: # Not enough val data after discard, try to take from train
                    df_val_raw_combined_csv = pd.DataFrame() # Mark as empty to trigger fallback

                if not df_val_raw_combined_csv.empty : # Re-check after processing
                    df_processed_val_for_windowing = pd.DataFrame(raw_val_for_proc, columns=TARGET_CHANNELS)
                    df_processed_val_for_windowing[MSEQ_COL] = labels_val_for_proc
                    X_va_csv, y_va_labels_csv = create_windows_for_ivep(df_processed_val_for_windowing, hp)
            
            if X_va_csv.size == 0 or X_va_csv.shape[0] == 0 : # Fallback if no val data from val segments
                val_count_from_train_csv = max(1, int(X_tr_csv.shape[0] * TEST_SIZE_RATIO))
                min_train_batch_size_csv = max(1, hp["batch"])
                if X_tr_csv.shape[0] - val_count_from_train_csv < min_train_batch_size_csv :
                    val_count_from_train_csv = X_tr_csv.shape[0] - min_train_batch_size_csv
                if val_count_from_train_csv <= 0 or (X_tr_csv.shape[0] - val_count_from_train_csv) < 1:
                    continue # Skip this CSV if cannot form validation
                X_va_csv, y_va_labels_csv = X_tr_csv[-val_count_from_train_csv:], y_tr_labels_csv[-val_count_from_train_csv:]
                X_tr_csv, y_tr_labels_csv = X_tr_csv[:-val_count_from_train_csv], y_tr_labels_csv[:-val_count_from_train_csv]

            if X_tr_csv.shape[0] < hp["batch"] or X_va_csv.shape[0] < 1: continue

            y_tr_cat_csv, y_va_cat_csv = to_categorical(y_tr_labels_csv, num_classes=NUM_CLASSES), to_categorical(y_va_labels_csv, num_classes=NUM_CLASSES)
            A_tr_csv = np.ones((X_tr_csv.shape[0], NUM_CHANNELS, NUM_CHANNELS), dtype=np.float32) 
            A_va_csv = np.ones((X_va_csv.shape[0], NUM_CHANNELS, NUM_CHANNELS), dtype=np.float32)

            K.clear_session() 
            model_csv = build_model(hp, X_tr_csv.shape[1], NUM_CHANNELS) 
            es_csv = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=VERBOSE)
            hist_csv = model_csv.fit([X_tr_csv, A_tr_csv], y_tr_cat_csv,
                                 validation_data=([X_va_csv, A_va_csv], y_va_cat_csv),
                                 batch_size=hp["batch"], epochs=EPOCHS, callbacks=[es_csv], verbose=VERBOSE)

            val_loss_hist_csv = hist_csv.history.get("val_loss", [np.inf])
            val_acc_hist_csv = hist_csv.history.get("val_accuracy", [0.0])
            if not val_loss_hist_csv: continue 

            best_epoch_idx_csv = np.argmin(val_loss_hist_csv)
            final_val_loss_csv = val_loss_hist_csv[best_epoch_idx_csv]
            final_val_acc_csv = val_acc_hist_csv[best_epoch_idx_csv]
            if np.isnan(final_val_loss_csv) or np.isinf(final_val_loss_csv): continue
            
            csv_val_losses.append(final_val_loss_csv)
            csv_val_accuracies.append(final_val_acc_csv)
        except TrialPruned as e_prune: continue 
        except Exception as e_csv: continue

    if not csv_val_losses: raise TrialPruned("No CSVs processed successfully.")
    avg_val_loss = np.mean(csv_val_losses)
    avg_val_accuracy = np.mean(csv_val_accuracies) if csv_val_accuracies else 0.0
    trial.set_user_attr("mean_val_accuracy_across_csvs", float(avg_val_accuracy))
    trial.set_user_attr("num_csvs_processed_in_trial", len(csv_val_losses))
    if np.isnan(avg_val_loss) or np.isinf(avg_val_loss): raise TrialPruned("Avg val_loss is NaN/Inf.")
    return float(avg_val_loss)

## ─── Progress callback & Run the study ────────────────────────────────────
def print_trial_summary(study, trial): # MODIFIED to show new user_attrs
    if trial.state is not TrialState.COMPLETE:
        reason = "N/A"
        if trial.state == TrialState.PRUNED: reason = "Pruned by Optuna"
        elif trial.state == TrialState.FAIL: 
            fail_reason_tuple = trial.system_attrs.get('fail_reason', ('Unknown failure',))
            # fail_reason can sometimes be a tuple (exception_type_str, exception_str_repr)
            reason = fail_reason_tuple[0] if isinstance(fail_reason_tuple, tuple) else fail_reason_tuple
        print(f"[Trial {trial.number}] {trial.state.name}. Reason: {reason}")
        return

    avg_l_val = trial.value
    avg_a_val = trial.user_attrs.get("mean_val_accuracy_across_csvs", 0.0)
    num_proc_csvs = trial.user_attrs.get("num_csvs_processed_in_trial", 0)

    print(f"[Trial {trial.number}] avg_val_loss={avg_l_val:.4f}, avg_val_acc={avg_a_val:.4f} (over {num_proc_csvs} CSVs)")
    
    # Find the best trial based on the main objective value (avg_val_loss)
    # study.best_trial might not be updated immediately if the current trial is not better,
    # so we compare against study.best_value.
    current_best_value = study.best_value if study.best_trial else float('inf') 
    
    if trial.value <= current_best_value : # Current trial is potentially the new best (or equal to current best)
        # At this point, if trial.value is better, study.best_trial should reflect the current trial.
        # If it's equal, study.best_trial might be the older one or the current one depending on Optuna's tie-breaking.
        # We explicitly check if the current trial IS the best one registered by Optuna.
        is_new_best = study.best_trial and trial.number == study.best_trial.number
        
        b = study.best_trial # This will be the current trial if it's the new best
        best_avg_acc = b.user_attrs.get('mean_val_accuracy_across_csvs', 0.0)
        best_num_csvs = b.user_attrs.get('num_csvs_processed_in_trial', 0)
        
        if is_new_best:
            print(f"  ✨ New Best Trial! ✨ avg_loss={b.value:.4f}, avg_acc={best_avg_acc:.4f} ({best_num_csvs} CSVs)\n")
        else: # Current trial is good (equal to best) but not THE single best trial Optuna is tracking
            # Or, if it's slightly worse than an existing best_trial that just got updated by another worker (less likely here)
            # This branch helps provide context about the current best.
            print(f"  ↳ Current Best (T{b.number}): avg_loss={b.value:.4f}, avg_acc={best_avg_acc:.4f} ({best_num_csvs} CSVs)\n")

    elif study.best_trial : # Current trial is not the best, but a best trial exists
        b = study.best_trial
        best_avg_acc = b.user_attrs.get('mean_val_accuracy_across_csvs', 0.0)
        best_num_csvs = b.user_attrs.get('num_csvs_processed_in_trial', 0)
        print(f"  ↳ Best Remains (T{b.number}): avg_loss={b.value:.4f}, avg_acc={best_avg_acc:.4f} ({best_num_csvs} CSVs)\n")
    # If no best_trial yet (e.g., first few trials), no "best" comparison is printed.

# --- Now comes the if __name__ == "__main__": block ---

if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try: 
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Running on GPU: {gpus}")
        except RuntimeError as e: print(f"Error setting up GPU: {e}")
    else: print("No GPU detected. Running on CPU.")
    
    sampler = TPESampler(seed=42, n_startup_trials=30) 
    pruner  = PercentilePruner(percentile=33.0, n_warmup_steps=20, n_min_trials=15) 
    
    study_name_v9 = "ivep_gatcnn_optuna_v9_proper_split" # NEW STUDY NAME
    db_name_v9 = f"{study_name_v9}.db"
    
    study = optuna.create_study(
        direction="minimize", sampler=sampler, pruner=pruner,
        study_name=study_name_v9, storage=f"sqlite:///{db_name_v9}", load_if_exists=True
    )
    print(f"Starting/Resuming PROPER SPLIT Optuna study '{study_name_v9}' using DB '{db_name_v9}'...")
    
    study_attrs = {
        "NUM_CHANNELS": NUM_CHANNELS, "TARGET_CHANNELS": TARGET_CHANNELS, 
        "FS_RAW": FS_RAW, "ALL_CSV_FILES_AVAILABLE": ALL_SUBJECT_FILENAMES,
        "NUM_CSVS_PER_TRIAL": NUM_CSVS_TO_USE,
        "CONSTRAINED_SEARCH_RANGES_APPLIED_V9": { 
            "window_duration_ms": [10, 300], "stride_ratio_of_window": ["1_sample_min", "0.5_max_of_window"], 
            "bp_low_hz": [0.1, 8.0], "bp_high_hz": [25.0, "Nyquist_max"]
        }
    }
    for key, val in study_attrs.items(): study.set_user_attr(key,val)

    try:
        study.optimize(objective, n_trials=50000, show_progress_bar=True, callbacks=[print_trial_summary]) 
    except KeyboardInterrupt: print("Optimization stopped by user.")
    except Exception as e: 
        print(f"Optimization stopped due to UNEXPECTED error: {e}")
        import traceback; traceback.print_exc()

    if study.trials:
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if completed_trials:
            best_trial_overall = min(completed_trials, key=lambda t: t.value)
            print(f"\n=== Best Trial Overall (Proper Split Search) {best_trial_overall.number} ===")
            print(f"  Value (avg_val_loss): {best_trial_overall.value:.4f}")
            val_acc_best = best_trial_overall.user_attrs.get('mean_val_accuracy_across_csvs', 'N/A')
            num_csvs_best = best_trial_overall.user_attrs.get('num_csvs_processed_in_trial', 'N/A')
            if isinstance(val_acc_best, float): print(f"  User Attrs (avg_val_acc): {val_acc_best:.4f} (over {num_csvs_best} CSVs)")
            else: print(f"  User Attrs (avg_val_acc): {val_acc_best}")
            print("  Params:", json.dumps(best_trial_overall.params, indent=2))
            best_params_filename = f"{study_name_v9}_best_params.json"
            with open(best_params_filename, "w") as f: json.dump(best_trial_overall.params, f, indent=2)
            print(f"Best HPs from proper split search saved to {best_params_filename}")
        else: print("\nNo trials completed successfully in proper split search.")
    else: print("\nNo trials run/completed in proper split search.")
    print("Proper split Optuna search finished.")