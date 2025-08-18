
# src/preprocessing.py
"""
import pretty_midi
import numpy as np
import os
from pathlib import Path
import librosa

def midi_to_note_sequence(midi_file):
    #Converte un file MIDI in una lista di note (pitch, start, end).
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append((note.pitch, note.start, note.end))
    notes.sort(key=lambda x: x[1])  # Ordina per start time
    return notes

def notes_to_training_sequences(notes, sequence_length=50):
    #Prepara sequenze di training a partire da una lista di note.
    if len(notes) <= sequence_length:
        return np.array([]), np.array([])

    pitches = np.array([n[0] for n in notes])
    X, y = [], []
    for i in range(len(pitches) - sequence_length):
        seq_in = pitches[i:i+sequence_length]
        seq_out = pitches[i+sequence_length]
        X.append(seq_in)
        y.append(seq_out)
    return np.array(X), np.array(y)

def load_dataset(folder_path, sequence_length=50):
    #Carica tutti i file MIDI da una cartella e restituisce X e y.
    X_all = []
    y_all = []
    total_files = 0
    valid_files = 0

    for file in os.listdir(folder_path):
        if file.endswith(".mid") or file.endswith(".midi"):
            total_files += 1
            path = os.path.join(folder_path, file)
            notes = midi_to_note_sequence(path)
            X, y = notes_to_training_sequences(notes, sequence_length)
            if X.size > 0:
                X_all.append(X)
                y_all.append(y)
                valid_files += 1
            else:
                print(f" File troppo corto ignorato: {file}")

    if not X_all:
        raise ValueError(" Nessun file MIDI valido trovato o troppo pochi dati.")

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    print(f" Dataset caricato: {valid_files}/{total_files} file usati, {X_all.shape[0]} sequenze")
    return X_all, y_all

if __name__ == "__main__":
    # Trova la cartella "dati/midi" compatibile con o senza OneDrive
    possible_paths = [
        Path.home() / "OneDrive" / "Desktop" / "dati" / "midi",
        Path.home() / "Desktop" / "dati" / "midi"
    ]

    for path in possible_paths:
        if path.exists():
            midi_folder = path
            break
    else:
        print("❌ Cartella MIDI non trovata né su OneDrive né su Desktop.")
        exit(1)

    X, y = load_dataset(str(midi_folder), sequence_length=50)


def extract_log_mel_spectrogram(file_path, n_mels=128, duration=30, sr=22050, hop_length=512):
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Uniforma a dimensioni 128x128 (utile per CNN)
        if log_mel_spec.shape[1] < 128:
            pad_width = 128 - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_mel_spec = log_mel_spec[:, :128]

        return log_mel_spec

    except Exception as e:
        print(f"Errore durante l'elaborazione di {file_path}: {e}")
        raise
"""



# -*- coding: utf-8 -*-
"""
Pipeline completa GTZAN (unico script) - versione con Fuzzy feature extraction
- MFCC aggregated (tabular) + MFCC sequences (LSTM)
- Mel-spectrogram (CNN), MFCC-image (CNN)
- RandomForest, SVM, LogisticRegression, KNN
- LSTM (sequences)
- Fuzzy C-Means + features fuzzificate (gauss/triang) + ensemble RF+Fuzzy
- Grad-CAM robust + plotting mel dB con colorbar (salva PNG)
- t-SNE su vettori MFCC mean
"""
# ----------------------------
# Installazione pacchetti necessari (se mancanti)
# ----------------------------
import sys, subprocess
required = {
    "librosa","numpy","pandas","scikit-learn","matplotlib","tqdm",
    "tensorflow","scikit-fuzzy","shap","lime","opencv-python","seaborn"
}
import pkg_resources
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if missing:
    print("Installando pacchetti mancanti:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

# ----------------------------
# Import principali
# ----------------------------
import os, glob, random, json
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa, librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.manifold import TSNE

import skfuzzy as fuzz
from skfuzzy.cluster import cmeans
import skfuzzy.membership as fuzzmf

import shap
from lime.lime_tabular import LimeTabularExplainer

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ----------------------------
# CONFIGURAZIONE (modifica se necessario)
# ----------------------------
DATASET_PATH = r"C:\Users\veryf\Desktop\GTZAN\genres_original"
SAMPLE_RATE = 22050
N_MFCC = 20
SEQ_LEN = 130        # frames per sequence MFCC (LSTM)
TEST_SIZE = 0.2
RANDOM_STATE = 42
CNN_EPOCHS = 30
LSTM_EPOCHS = 30
CNN_BATCH = 32
LSTM_BATCH = 32
RF_N_ESTIMATORS = 150
CMEANS_CLUSTERS = 10
CMEANS_M = 2.0

TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
MFCC_IMG_SHAPE = (128, 128)
MFCC_N_IMG = 40
MFCC_CNN_EPOCHS = 30
MFCC_CNN_BATCH = 32

MAX_FILES = None  # imposta ad es. 200 per debug

# ----------------------------
# UTILITY: trova file audio
# ----------------------------
def find_audio_files(dataset_path, max_files=None):
    files = sorted(glob.glob(os.path.join(dataset_path, "*", "*.wav")))
    if max_files:
        files = files[:max_files]
    return files

# ----------------------------
# Estrarre features: aggregated MFCC + MFCC sequence + altri spettrali
# Aggiunta: spectral_centroid per avere un'altra feature spettrale
# ----------------------------
def extract_features_for_file(file_path, sr=SAMPLE_RATE, n_mfcc=N_MFCC, seq_len=SEQ_LEN):
    y, sr = librosa.load(file_path, sr=sr, mono=True, duration=30.0)
    if y.size == 0:
        raise ValueError("Empty file: "+file_path)
    # Sequence MFCC (n_mfcc x frames)
    mfcc_seq = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_seq_t = mfcc_seq.T  # (frames, n_mfcc)
    # pad/trim to seq_len
    if mfcc_seq_t.shape[0] < seq_len:
        pad_w = seq_len - mfcc_seq_t.shape[0]
        mfcc_seq_t = np.pad(mfcc_seq_t, ((0,pad_w),(0,0)), mode='constant')
    else:
        mfcc_seq_t = mfcc_seq_t[:seq_len, :]
    # aggregated features
    agg = {}
    for i in range(n_mfcc):
        agg[f"mfcc_{i+1}_mean"] = np.mean(mfcc_seq[i])
        agg[f"mfcc_{i+1}_std"] = np.std(mfcc_seq[i])
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    agg["chroma_mean"] = np.mean(chroma)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    agg["mel_spec_mean"] = np.mean(S)
    zcr = librosa.feature.zero_crossing_rate(y)
    agg["zcr_mean"] = np.mean(zcr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    agg["tempo"] = tempo
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    agg["spectral_centroid_mean"] = np.mean(spec_cent)
    return agg, mfcc_seq_t

# ----------------------------
# Mel-spectrogram image (per CNN mel)
# ----------------------------
def make_mel_image(file_path, sr=SAMPLE_RATE, n_mels=128, img_shape=(128,128)):
    y, sr = librosa.load(file_path, sr=sr, mono=True, duration=30.0)
    if y.size == 0:
        raise ValueError("Empty file: "+file_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
    img = cv2.resize(S_norm, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)

# ----------------------------
# Plot Mel-spectrogram con colorbar in dB e assi in italiano
# ----------------------------
def plot_mel_spectrogram_with_db(file_path, sr=SAMPLE_RATE, n_mels=128,
                                 n_fft=2048, hop_length=512, duration=3.0,
                                 idx=None, label=None, cmap='magma', figsize=(12,5),
                                 ax=None):
    y, sr = librosa.load(file_path, sr=sr, mono=True, duration=duration)
    if y.size == 0:
        raise ValueError("Empty file: " + file_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmax=sr/2)
    S_db = librosa.power_to_db(S, ref=np.max)

    created_fig = False
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        created_fig = True

    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', fmax=sr/2, cmap=cmap, ax=ax)
    cb = plt.colorbar(img, ax=ax, format='%+2.0f dB')
    cb.set_label('dB', rotation=270, labelpad=15)
    title_idx = f"idx={idx}, " if idx is not None else ""
    title_label = f"label={label}" if label is not None else ""
    title = "Mel-spectrogram"
    if title_idx or title_label:
        title = f"{title} di {title_idx}{title_label}".strip().rstrip(',')
    ax.set_title(title)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Frequenza Mel (Hz)")
    if created_fig:
        plt.tight_layout()
        plt.show()
    return S_db, img

# ----------------------------
# MFCC-image
# ----------------------------
def make_mfcc_image(file_path, n_mfcc=MFCC_N_IMG, img_shape=MFCC_IMG_SHAPE, sr=SAMPLE_RATE):
    y, sr = librosa.load(file_path, sr=sr, mono=True, duration=30.0)
    if y.size == 0:
        raise ValueError("Empty file: "+file_path)
    S = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    S_norm = (S - S.min()) / (S.max() - S.min() + 1e-9)
    img = cv2.resize(S_norm, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)

# ----------------------------
# Fuzzificazione semplice (membership functions)
# - genera membership per 'low','mid','high' per colonne scelte
# ----------------------------
def fuzzify_column(values, method='gauss', centers=None, widths=None):
    """
    values: np.array 1D
    method: 'gauss' o 'triang'
    centers: list di 3 centri [low, mid, high] (se None, calcola basandosi su min/mean/max)
    widths: lista di 3 width (sd o half-base)
    returns: dict {'low': arr, 'mid': arr, 'high': arr}
    """
    v = np.asarray(values, dtype=float)
    vmin, vmax = v.min(), v.max()
    if centers is None:
        centers = [vmin, v.mean(), vmax]
    if widths is None:
        widths = [max(1e-6,(centers[1]-centers[0])), max(1e-6,(centers[2]-centers[0])/2), max(1e-6,(centers[2]-centers[1]))]
    out = {}
    if method == 'gauss':
        out['low'] = fuzzmf.gaussmf(v, centers[0], widths[0])
        out['mid'] = fuzzmf.gaussmf(v, centers[1], widths[1])
        out['high'] = fuzzmf.gaussmf(v, centers[2], widths[2])
    else:
        # triangular: trap/trimf approximated with three trimf centered on centers
        a0, b0, c0 = vmin, (centers[0]+centers[1])/2.0, centers[1]
        out['low'] = fuzzmf.trimf(v, [a0, centers[0], b0])
        a1, b1, c1 = centers[0], centers[1], centers[2]
        out['mid'] = fuzzmf.trimf(v, [a1, centers[1], c1])
        a2, b2, c2 = centers[1], (centers[1]+centers[2])/2.0, vmax
        out['high'] = fuzzmf.trimf(v, [a2, centers[2], c2])
    return out

def fuzzify_dataframe_features(df_tab, cols_to_fuzzify=None, method='gauss'):
    """
    Aggiunge colonne fuzzificate per ogni col in cols_to_fuzzify:
    col_low, col_mid, col_high
    """
    if cols_to_fuzzify is None:
        cols_to_fuzzify = ['tempo','zcr_mean','mel_spec_mean','chroma_mean','spectral_centroid_mean']
    df_out = df_tab.copy()
    for col in cols_to_fuzzify:
        if col not in df_out.columns:
            continue
        vals = df_out[col].values
        mems = fuzzify_column(vals, method=method)
        df_out[f"{col}_low"] = mems['low']
        df_out[f"{col}_mid"] = mems['mid']
        df_out[f"{col}_high"] = mems['high']
    return df_out

# ----------------------------
# Costruisci dataset: tabular, seq, mel images, mfcc images
# ----------------------------
def build_datasets(dataset_path, max_files=None, n_mfcc=N_MFCC, seq_len=SEQ_LEN):
    files = find_audio_files(dataset_path, max_files)
    if len(files) == 0:
        raise FileNotFoundError(f"Nessun .wav trovato in {dataset_path}")
    rows, seqs, mel_imgs, mfcc_imgs, labels = [], [], [], [], []
    print(f"Found {len(files)} files — extracting features (this may take time)...")
    for fp in tqdm(files):
        try:
            agg, seq = extract_features_for_file(fp, n_mfcc=n_mfcc, seq_len=seq_len)
            mel = make_mel_image(fp)
            mfcc_img = make_mfcc_image(fp)
            genre = os.path.basename(os.path.dirname(fp))
            agg["genre"] = genre
            agg["file"] = fp
            rows.append(agg)
            seqs.append(seq)
            mel_imgs.append(mel)
            mfcc_imgs.append(mfcc_img)
            labels.append(genre)
        except Exception as e:
            print("Skipping", fp, e)
    df_tab = pd.DataFrame(rows)
    # Fuzzificazione: aggiungiamo le colonne fuzzy
    df_tab_fuzzy = fuzzify_dataframe_features(df_tab)
    X_seq = np.array(seqs)       # (N, seq_len, n_mfcc)
    X_mel = np.array(mel_imgs)   # (N, H, W)
    X_mfcc_img = np.array(mfcc_imgs)
    y = np.array(labels)
    # restituiamo anche df_tab_fuzzy per usare le colonne fuzzy
    return df_tab_fuzzy, X_seq, X_mel, X_mfcc_img, y

# ----------------------------
# Modelli tabulari
# ----------------------------
def train_tabular_models(X_train, y_train):
    models = {}
    rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['rf'] = rf
    svm = SVC(probability=True, kernel='rbf', random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)
    models['svm'] = svm
    lr = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    models['lr'] = lr
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    models['knn'] = knn
    return models

# ----------------------------
# CNN & LSTM builders (stessi di prima)
# ----------------------------
def build_cnn(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)
    x = inp if len(input_shape)==3 else layers.Reshape((*input_shape,1))(inp)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(seq_len, n_mfcc, n_classes):
    inp = layers.Input(shape=(seq_len, n_mfcc))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ----------------------------
# Fuzzy C-means utilities (già presenti)
# ----------------------------
def train_fuzzy_cmeans(X_train, y_train, n_clusters=CMEANS_CLUSTERS, m=CMEANS_M):
    data = X_train.T
    cntr, u, u0, d, jm, p, fpc = cmeans(data, c=n_clusters, m=m, error=0.005, maxiter=1000, init=None)
    unique_classes = np.unique(y_train)
    n_clusters = cntr.shape[0]
    n_classes = len(unique_classes)
    P = np.zeros((n_clusters, n_classes))
    for j in range(n_clusters):
        for i_c, c in enumerate(unique_classes):
            mask = (y_train == c)
            P[j, i_c] = np.sum(u[j, mask])
        s = np.sum(P[j])
        if s > 0:
            P[j] = P[j] / s
    return cntr, u, P, unique_classes

def cmeans_membership_from_centers(centers, X, m=CMEANS_M):
    centers = np.asarray(centers)
    X = np.asarray(X)
    n_centers, n_features = centers.shape
    n_samples = X.shape[0]
    d = np.zeros((n_centers, n_samples))
    for i in range(n_centers):
        d[i] = np.linalg.norm(X - centers[i], axis=1)
    power = 2.0/(m-1.0)
    u = np.zeros((n_centers, n_samples))
    for j in range(n_samples):
        dj = d[:, j]
        if np.any(dj == 0):
            u[:, j] = 0.0
            u[np.argmin(dj), j] = 1.0
        else:
            ratio = (dj.reshape((-1,1))/dj.reshape((1,-1)))**power
            denom = np.sum(ratio, axis=1)
            u[:, j] = 1.0/denom
    return u

def predict_fuzzy_from_clusters(centers, P_cluster_genre, X_test, m=CMEANS_M):
    u_test = cmeans_membership_from_centers(centers, X_test, m=m)
    n_clusters, n_samples = u_test.shape
    _, n_genres = P_cluster_genre.shape
    probs = np.zeros((n_samples, n_genres))
    for s in range(n_samples):
        probs[s] = np.dot(u_test[:, s], P_cluster_genre)
    preds = np.argmax(probs, axis=1)
    return preds, probs, u_test

# ----------------------------
# Grad-CAM robust + plotting (versione che salva png)
# ----------------------------
import matplotlib.gridspec as gridspec
import matplotlib
import os

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def compute_gradcam_heatmap(model, img_input, last_conv_name=None, pred_index=None):
    img_input = np.asarray(img_input, dtype=np.float32)
    if img_input.ndim != 4 or img_input.shape[0] != 1:
        raise ValueError("img_input must be shape (1,H,W,C)")

    if last_conv_name is None:
        last_conv_name = get_last_conv_layer_name(model)
        if last_conv_name is None:
            raise ValueError("Nessun layer Conv2D trovato nel modello")

    grad_model = tf.keras.models.Model(inputs=model.inputs,
                                       outputs=[model.get_layer(last_conv_name).output, model.output])

    img_tensor = tf.convert_to_tensor(img_input, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0]).numpy()
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        print("DEBUG: grads is None -> probabilmente grad tape non ha collegamenti (controlla modello)")
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    weighted = conv_outputs * pooled_grads[tf.newaxis, tf.newaxis, :]
    heatmap = tf.reduce_sum(weighted, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap = heatmap / (max_val + 1e-9)
    heatmap = heatmap.numpy()
    return heatmap, int(pred_index)

def plot_gradcam_overlay_for_file_v3(model, file_path, sample_img_for_model,
                                     sr=22050, n_mels=128, hop_length=512,
                                     last_conv_name=None, duration=30.0,
                                     cmap='magma', out_dir=None):
    img = np.array(sample_img_for_model)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    img_input = np.expand_dims(img, axis=0).astype(np.float32)

    print("DEBUG: img_input.shape", img_input.shape, "dtype", img_input.dtype,
          "min/max img:", img_input.min(), img_input.max())

    last_conv = last_conv_name or get_last_conv_layer_name(model)
    print("DEBUG: last_conv_layer:", last_conv)
    if last_conv is None:
        raise RuntimeError("Nessun Conv2D trovato nel modello. Impossibile Grad-CAM.")

    heatmap, pred_idx = compute_gradcam_heatmap(model, img_input, last_conv_name=last_conv, pred_index=None)
    print("DEBUG: pred_class_idx:", pred_idx, "heatmap shape:", heatmap.shape,
          "min/max:", float(heatmap.min()), float(heatmap.max()))

    try:
        probs = model.predict(img_input, verbose=0)[0]
        print("DEBUG: top preds (idx:prob):", sorted([(i, float(p)) for i,p in enumerate(probs)], key=lambda x:-x[1])[:5])
    except Exception as e:
        print("DEBUG: impossibile ottenere probs:", e)

    y_raw, sr = librosa.load(file_path, sr=sr, mono=True, duration=duration)
    S = librosa.feature.melspectrogram(y=y_raw, sr=sr, n_mels=n_mels, hop_length=hop_length, fmax=sr/2)
    S_db = librosa.power_to_db(S, ref=np.max)
    n_mels_calc, n_frames = S_db.shape
    print("DEBUG: S_db.shape", S_db.shape)
    time_end = (n_frames * hop_length) / sr
    extent = [0, time_end, 0, sr/2]

    heat_resized = cv2.resize(heatmap, (n_frames, n_mels_calc), interpolation=cv2.INTER_CUBIC)

    fig = plt.figure(figsize=(14,6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.25)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    im0 = ax0.imshow(S_db, origin='lower', aspect='auto', extent=extent, cmap=cmap)
    ax0.set_title(f"Mel-spectrogram (dB)\n{os.path.basename(file_path)}")
    ax0.set_xlabel("Tempo (s)")
    ax0.set_ylabel("Frequenza (Hz)")

    im1 = ax1.imshow(S_db, origin='lower', aspect='auto', extent=extent, cmap=cmap)
    ax1.imshow(heat_resized, cmap='jet', alpha=0.5, origin='lower', extent=extent)
    ax1.set_title(f"Grad-CAM overlay (pred class idx={pred_idx})")
    ax1.set_xlabel("Tempo (s)")
    ax1.set_ylabel("Frequenza (Hz)")

    cbar = fig.colorbar(im1, cax=cax, format='%+2.0f dB')
    cbar.set_label('dB', rotation=270, labelpad=12)

    fig.subplots_adjust(left=0.06, right=0.92, top=0.92, bottom=0.08)

    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(out_dir, f'gradcam_{basename}.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    print("Saved gradcam image to:", out_path)

    try:
        plt.show(block=True)
    except Exception:
        plt.pause(0.5)

    return heatmap, pred_idx, out_path

# ----------------------------
# Metriche & plotting utilities
# ----------------------------
def compute_metrics(y_true, y_pred, y_proba=None, classes=None):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    if y_proba is not None and classes is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=np.arange(len(classes)))
            metrics['roc_auc_ovr_macro'] = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
        except Exception:
            metrics['roc_auc_ovr_macro'] = None
    return metrics

def plot_confusion(cm, classes, title="Confusion matrix"):
    plt.figure(figsize=(9,7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_training_curves(history, title_prefix="Model"):
    if history is None:
        print("No history provided")
        return
    h = history.history
    def to_arr(key):
        if key not in h:
            return None
        arr = np.array(h[key], dtype=float)
        if arr.ndim > 1:
            arr = arr.flatten()
        return arr
    loss = to_arr('loss'); val_loss = to_arr('val_loss')
    acc_key = 'accuracy' if 'accuracy' in h else ('acc' if 'acc' in h else None)
    acc = to_arr(acc_key) if acc_key else None
    val_acc = to_arr('val_' + acc_key) if acc_key and ('val_' + acc_key) in h else None
    if acc is not None and np.nanmax(acc) > 1.1:
        acc = acc / 100.0
    if val_acc is not None and np.nanmax(val_acc) > 1.1:
        val_acc = val_acc / 100.0
    epochs = range(1, (len(loss) if loss is not None else 0) + 1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    if loss is not None:
        plt.plot(epochs, loss, marker='o', label='train loss')
    if val_loss is not None:
        plt.plot(epochs, val_loss, marker='o', label='val loss')
    plt.title(f"{title_prefix} - Loss"); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.2)
    plt.subplot(1,2,2)
    if acc is not None:
        plt.plot(epochs, acc, marker='o', label='train acc')
    if val_acc is not None:
        plt.plot(epochs, val_acc, marker='o', label='val acc')
    plt.title(f"{title_prefix} - Accuracy"); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(alpha=0.2)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout(); plt.show()

# ----------------------------
# LIME wrapper for LSTM (usiamo features tabulari come interpretable)
# ----------------------------
def make_lstm_predict_proba_wrapper(lstm_model, scaler_tab, seq_len=SEQ_LEN, n_mfcc=N_MFCC):
    def predict_proba_from_tab(X_tab):
        X_tab_scaled = scaler_tab.transform(X_tab)
        if X_tab.shape[1] >= n_mfcc:
            mfcc_means = X_tab_scaled[:, :n_mfcc]
        else:
            mfcc_means = np.zeros((X_tab.shape[0], n_mfcc))
        seqs = np.repeat(mfcc_means[:, np.newaxis, :], seq_len, axis=1)
        probs = lstm_model.predict(seqs, verbose=0)
        return probs
    return predict_proba_from_tab

# ----------------------------
# t-SNE su MFCC mean vectors
# ----------------------------
def plot_tsne_on_mfcc_vectors(df_tab, label_col="genre", n_samples=None, perplexity=TSNE_PERPLEXITY, n_iter=TSNE_N_ITER, random_state=RANDOM_STATE):
    mfcc_mean_cols = [c for c in df_tab.columns if c.startswith("mfcc_") and c.endswith("_mean")]
    X = df_tab[mfcc_mean_cols].values
    y = df_tab[label_col].values
    if n_samples is not None and n_samples < X.shape[0]:
        idx = np.random.RandomState(random_state).choice(np.arange(X.shape[0]), size=n_samples, replace=False)
        X = X[idx]; y = y[idx]
    print("Running t-SNE on shape:", X.shape)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state, init='pca')
    X2 = tsne.fit_transform(X)
    plt.figure(figsize=(10,8))
    unique_labels = np.unique(y)
    palette = sns.color_palette("tab10", n_colors=len(unique_labels))
    for i, lab in enumerate(unique_labels):
        mask = (y==lab)
        plt.scatter(X2[mask,0], X2[mask,1], s=15, label=lab, color=palette[i])
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.title("t-SNE of MFCC mean vectors")
    plt.xlabel("TSNE-1"); plt.ylabel("TSNE-2")
    plt.tight_layout(); plt.show()

# ----------------------------
# Funzione principale
# ----------------------------
def main():
    print("1) Costruzione dataset (estrazione MFCC, mel, MFCC-image, fuzzy features)...")
    df_tab, X_seq, X_mel, X_mfcc_img, y = build_datasets(DATASET_PATH, max_files=MAX_FILES)
    print("Shapes:", df_tab.shape, X_seq.shape, X_mel.shape, X_mfcc_img.shape)
    le = LabelEncoder(); y_enc = le.fit_transform(y); classes = le.classes_
    print("Generi:", list(classes))

    # Tabular prep - includiamo anche le colonne fuzzificate
    X_tab_df = df_tab.drop(columns=["genre","file"]).copy()
    # Ordine colonne (usiamo tutte le originali e le fuzzy aggiunte)
    feature_names = X_tab_df.columns.tolist()
    X_tab = X_tab_df.values
    scaler_tab = StandardScaler().fit(X_tab)
    X_tab_s = scaler_tab.transform(X_tab)
    X_tab_train, X_tab_test, y_tab_train, y_tab_test, idx_train, idx_test = train_test_split(
        X_tab_s, y_enc, np.arange(len(X_tab_s)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc)

    print("\n2) Training modelli tabulari (RF,SVM,LR,KNN)...")
    tab_models = train_tabular_models(X_tab_train, y_tab_train)
    for name, m in tab_models.items():
        y_pred = m.predict(X_tab_test)
        y_proba = m.predict_proba(X_tab_test) if hasattr(m, "predict_proba") else None
        print(f"\n-- {name.upper()} --")
        print("Accuracy:", accuracy_score(y_tab_test, y_pred))
        print(classification_report(y_tab_test, y_pred, target_names=classes, zero_division=0))
        plot_confusion(confusion_matrix(y_tab_test, y_pred), classes, title=f"{name.upper()} Confusion Matrix")

    # Fuzzy C-means on tabular scaled features
    print("\n3) Fuzzy C-Means (tabular scaled features)...")
    centers, u_train, P_cluster_genre, unique_classes = train_fuzzy_cmeans(X_tab_train, y_tab_train)
    fuzzy_preds_test, fuzzy_probs_test, u_test = predict_fuzzy_from_clusters(centers, P_cluster_genre, X_tab_test)
    print("Fuzzy accuracy:", accuracy_score(y_tab_test, fuzzy_preds_test))
    print(classification_report(y_tab_test, fuzzy_preds_test, target_names=classes, zero_division=0))
    plot_confusion(confusion_matrix(y_tab_test, fuzzy_preds_test), classes, title="Fuzzy Confusion")

    # Ensemble RF + Fuzzy (su tabular)
    print("\n4) Ensemble RF + Fuzzy (tabular):")
    rf = tab_models['rf']
    rf_proba_test = rf.predict_proba(X_tab_test)
    ensemble_probs = 0.6 * rf_proba_test + 0.4 * fuzzy_probs_test
    ensemble_pred = np.argmax(ensemble_probs, axis=1)
    print("Ensemble accuracy:", accuracy_score(y_tab_test, ensemble_pred))
    print(classification_report(y_tab_test, ensemble_pred, target_names=classes, zero_division=0))
    plot_confusion(confusion_matrix(y_tab_test, ensemble_pred), classes, title="Ensemble RF+Fuzzy Confusion")

    # CNN su mel-spectrogram
    print("\n5) CNN su mel-spectrogram (training)...")
    X_mel_norm = (X_mel - X_mel.min()) / (X_mel.max() - X_mel.min() + 1e-9)
    X_mel_norm = X_mel_norm[..., np.newaxis]
    files_all = df_tab['file'].values
    X_mel_train, X_mel_test, y_mel_train, y_mel_test, files_train, files_test = train_test_split(
        X_mel_norm, y_enc, files_all, test_size=TEST_SIZE, stratify=y_enc, random_state=RANDOM_STATE)

    cnn_input_shape = X_mel_train.shape[1:]
    cnn_model = build_cnn(cnn_input_shape, n_classes=len(classes))
    es = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history_cnn_mel = cnn_model.fit(X_mel_train, y_mel_train, validation_data=(X_mel_test, y_mel_test),
                                    epochs=CNN_EPOCHS, batch_size=CNN_BATCH, callbacks=[es], verbose=2)
    cnn_proba = cnn_model.predict(X_mel_test)
    cnn_pred = np.argmax(cnn_proba, axis=1)
    print("CNN (mel) accuracy:", accuracy_score(y_mel_test, cnn_pred))
    print(classification_report(y_mel_test, cnn_pred, target_names=classes, zero_division=0))
    plot_confusion(confusion_matrix(y_mel_test, cnn_pred), classes, title="CNN (mel) Confusion")
    plot_training_curves(history_cnn_mel, title_prefix="Mel-spectrogram CNN")

    # CNN su MFCC-image
    print("\n6) CNN su MFCC-image (training)...")
    X_mfcc_img_norm = (X_mfcc_img - X_mfcc_img.min()) / (X_mfcc_img.max() - X_mfcc_img.min() + 1e-9)
    X_mfcc_img_norm = X_mfcc_img_norm[..., np.newaxis]
    X_mfcc_tr, X_mfcc_te, y_mfcc_tr, y_mfcc_te = train_test_split(X_mfcc_img_norm, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=RANDOM_STATE)
    mfcc_cnn_model = build_cnn(X_mfcc_tr.shape[1:], n_classes=len(classes))
    es2 = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history_mfcc_cnn = mfcc_cnn_model.fit(X_mfcc_tr, y_mfcc_tr, validation_data=(X_mfcc_te, y_mfcc_te),
                                          epochs=MFCC_CNN_EPOCHS, batch_size=MFCC_CNN_BATCH, callbacks=[es2], verbose=2)
    mfcc_proba = mfcc_cnn_model.predict(X_mfcc_te)
    mfcc_pred = np.argmax(mfcc_proba, axis=1)
    print("MFCC-CNN accuracy:", accuracy_score(y_mfcc_te, mfcc_pred))
    print(classification_report(y_mfcc_te, mfcc_pred, target_names=classes, zero_division=0))
    plot_confusion(confusion_matrix(y_mfcc_te, mfcc_pred), classes, title="MFCC-CNN Confusion")
    plot_training_curves(history_mfcc_cnn, title_prefix="MFCC-image CNN")

    # LSTM su sequences MFCC
    print("\n7) LSTM su sequenze MFCC (training)...")
    N, seq_len, n_mfcc = X_seq.shape
    X_seq_flat = X_seq.reshape(N, seq_len * n_mfcc)
    scaler_seq = StandardScaler().fit(X_seq_flat)
    X_seq_flat_s = scaler_seq.transform(X_seq_flat)
    X_seq_s = X_seq_flat_s.reshape(N, seq_len, n_mfcc)
    X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq_s, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=RANDOM_STATE)
    lstm_model = build_lstm(seq_len, n_mfcc, n_classes=len(classes))
    es3 = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history_lstm = lstm_model.fit(X_seq_train, y_seq_train, validation_data=(X_seq_test, y_seq_test),
                                 epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, callbacks=[es3], verbose=2)
    lstm_proba = lstm_model.predict(X_seq_test)
    lstm_pred = np.argmax(lstm_proba, axis=1)
    print("LSTM accuracy:", accuracy_score(y_seq_test, lstm_pred))
    print(classification_report(y_seq_test, lstm_pred, target_names=classes, zero_division=0))
    plot_confusion(confusion_matrix(y_seq_test, lstm_pred), classes, title="LSTM Confusion")
    plot_training_curves(history_lstm, title_prefix="LSTM (MFCC seq)")

    # ROC AUC per RF (se possibile)
    print("\n8) ROC AUC (one-vs-rest) for RF (if possible):")
    try:
        y_tab_bin = label_binarize(y_tab_test, classes=np.arange(len(classes)))
        rf_auc = roc_auc_score(y_tab_bin, rf.predict_proba(X_tab_test), average='macro', multi_class='ovr')
        print("RF ROC AUC (ovr macro):", rf_auc)
    except Exception as e:
        print("ROC AUC error:", e)

    # SHAP explanation (RF)
    print("\n9) SHAP explanations (RF quick) — solo alcuni campioni:")
    try:
        explainer_rf = shap.TreeExplainer(rf)
        Xrf_sample = X_tab_test[:50]
        shap_values = explainer_rf.shap_values(Xrf_sample)
        try:
            shap.summary_plot(shap_values, features=Xrf_sample, feature_names=feature_names)
        except Exception as e:
            print("SHAP plotting skipped (env):", e)
    except Exception as e:
        print("SHAP RF error:", e)

    # Grad-CAM example (mel CNN) - con colorbar in dB e assi
    print("\n10) Grad-CAM example on mel-CNN:")
    last_conv = None
    for layer in reversed(cnn_model.layers):
        if isinstance(layer, layers.Conv2D):
            last_conv = layer.name
            break

    if last_conv:
        sample_idx = 0
        sample_file = files_test[sample_idx]
        sample_img_for_model = X_mel_test[sample_idx][..., 0]
        heatmap, pred, out_path = plot_gradcam_overlay_for_file_v3(cnn_model, sample_file, sample_img_for_model,
                                                                   sr=SAMPLE_RATE, n_mels=128, hop_length=512, duration=30.0)
        # opzionale: apri il file appena salvato su Windows
        try:
            if os.name == 'nt':
                os.startfile(out_path)
        except Exception:
            pass
    else:
        print("No Conv2D layer found for Grad-CAM.")

    # LIME explanation for LSTM via tabular wrapper
    print("\n11) LIME explanation (tabular wrapper -> LSTM):")
    try:
        predict_proba_lstm = make_lstm_predict_proba_wrapper(lstm_model, scaler_tab, seq_len=SEQ_LEN, n_mfcc=N_MFCC)
        explainer_lime = LimeTabularExplainer(training_data=X_tab_train, feature_names=feature_names,
                                              class_names=list(classes), mode='classification')
        i = 0
        exp = explainer_lime.explain_instance(X_tab_test[i], predict_proba_lstm, num_features=10)
        print("LIME explanation for sample", i, "->", exp.as_list())
        try:
            fig = exp.as_pyplot_figure()
            plt.show()
        except Exception:
            pass
    except Exception as e:
        print("LIME error:", e)

    # t-SNE su MFCC mean vectors
    print("\n12) t-SNE su vettori MFCC mean (esempio con max 1000 sample):")
    try:
        plot_tsne_on_mfcc_vectors(df_tab, n_samples=1000)
    except Exception as e:
        print("t-SNE error:", e)

    # Metriche riassuntive
    print("\n13) Metriche riassuntive:")
    summary = {}
    for name, m in tab_models.items():
        proba = m.predict_proba(X_tab_test) if hasattr(m, "predict_proba") else None
        pred = m.predict(X_tab_test)
        summary[name] = compute_metrics(y_tab_test, pred, y_proba=proba, classes=classes)
    summary['cnn_mel'] = compute_metrics(y_mel_test, cnn_pred, y_proba=cnn_proba, classes=classes)
    summary['cnn_mfcc'] = compute_metrics(y_mfcc_te, mfcc_pred, y_proba=mfcc_proba, classes=classes)
    summary['lstm'] = compute_metrics(y_seq_test, lstm_pred, y_proba=lstm_proba, classes=classes)
    print(json.dumps(summary, indent=2, default=str))

    print("\nPipeline completata.")
    print("Consigli: per sviluppo rapido riduci MAX_FILES o epoche; per training completo usa GPU.")

if __name__ == "__main__":
    main()
































