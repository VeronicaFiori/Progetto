
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
Pipeline completa GTZAN (unico script) – versione con PCA/LDA
- MFCC aggregated (tabular) + MFCC sequences (LSTM)
- Mel-spectrogram (CNN), MFCC-image (CNN)
- RandomForest, SVM, LogisticRegression, KNN
- LSTM (sequences)
- (Opzionale) Fuzzy C-Means + features fuzzificate (gauss/triang) + ensemble RF+Fuzzy
- Grad-CAM robust + plotting mel dB con colorbar (salva PNG)
- t-SNE su vettori MFCC mean
- ***NOVITÀ***: integrazione PCA e LDA prima dei classificatori tabellari e LSTM

NOTE IMPORTANTI:
- PCA/LDA non sono applicati alle CNN (mel/MFCC-image) perché romperebbero la struttura spaziale; si mantengono invariate.
- Per LSTM applichiamo PCA sui frame MFCC (per-feature), riducendo la dimensionalità per frame e rimodellando la sequenza.
- Per i modelli tabulari vengono eseguite tre varianti: baseline (scaler), PCA, LDA (e opzionale PCA→LDA quando n_features >> n_samples).
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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline

import skfuzzy as fuzz
#from skfuzzy.cluster import cmeans
import skfuzzy.membership as fuzzmf

import shap
from lime.lime_tabular import LimeTabularExplainer

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ----------------------------
# CONFIGURAZIONE (modifica se necessario)
# ----------------------------
paths = [
    r"C:\\Users\\catal\\OneDrive\\Desktop\\GTZAN\\genres_original",
    r"C:\\Users\\veryf\\Desktop\\GTZAN\\genres_original"
]

DATASET_PATH = None
for p in paths:
    if os.path.exists(p):
        DATASET_PATH = p
        break

if DATASET_PATH is None:
    raise FileNotFoundError("Nessun percorso dataset valido trovato.")

print(f"Percorso dataset selezionato: {DATASET_PATH}")

SAMPLE_RATE = 22050
N_MFCC = 40          # più info timbriche rispetto a 20
SEQ_LEN = 260        # ≈ 6 sec invece di 3 → più contesto per LSTM
TEST_SIZE = 0.3
RANDOM_STATE = 42

CNN_EPOCHS = 30
CNN_BATCH = 32
LSTM_EPOCHS = 30
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

# ---- Nuove opzioni PCA/LDA ----
# Per i modelli tabulari eseguiamo tutte le varianti: baseline, PCA e LDA.
# Per PCA si può scegliere numero componenti o varianza spiegata.
USE_PCA_VAR_EXPL = True             # se True usa soglia varianza spiegata, altrimenti n_componenti fisse
PCA_VAR_EXPL = 0.95              # varianza cumulativa target per PCA tabulare
PCA_N_COMPONENTS_TAB = 40        # usato solo se USE_PCA_VAR_EXPL=False

# LDA: componenti massime = n_classi - 1 (calcolato a runtime)
APPLY_PCA_BEFORE_LDA_WHEN_NF_GT_NS = True  # safety quando #feature >> #campioni
PCA_FOR_LDA_VAR = 0.99                      # PCA preliminare prima di LDA (se attivata)

# Per LSTM riduciamo la dimensionalità dei MFCC per frame con PCA
PCA_SEQ_N_COMPONENTS = 24         # es. da 40 → 24 componenti per frame

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
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency Mel (Hz)")
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
USE_ONLY_4_GENRES = False   # <-- metti True per usare solo classical/jazz/metal/pop

SELECTED_GENRES = ["classical", "jazz", "metal", "pop"]

def build_datasets(dataset_path, max_files=None, n_mfcc=N_MFCC, seq_len=SEQ_LEN):
    files = find_audio_files(dataset_path, max_files)
    if len(files) == 0:
        raise FileNotFoundError(f"Nessun .wav trovato in {dataset_path}")
    
    rows, seqs, mel_imgs, mfcc_imgs, labels = [], [], [], [], []
    print(f"Found {len(files)} files — extracting features (this may take time)...")

    for fp in tqdm(files):
        try:
            genre = os.path.basename(os.path.dirname(fp))
            
            # Selezione opzionale dei 4 generi
            if USE_ONLY_4_GENRES and genre not in SELECTED_GENRES:
                continue
            
            agg, seq = extract_features_for_file(fp, n_mfcc=n_mfcc, seq_len=seq_len)
            mel = make_mel_image(fp)
            mfcc_img = make_mfcc_image(fp)

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
    df_tab_fuzzy = fuzzify_dataframe_features(df_tab)

    X_seq = np.array(seqs)       # (N, seq_len, n_mfcc)
    X_mel = np.array(mel_imgs)   # (N, H, W)
    X_mfcc_img = np.array(mfcc_imgs)
    y = np.array(labels)

    print(f"Final dataset size: {len(y)} samples — Genres: {np.unique(y)}")
    return df_tab_fuzzy, X_seq, X_mel, X_mfcc_img, y


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
# LIME wrapper per LSTM (usiamo features tabulari come interpretable)
# ----------------------------

def make_lstm_proba_wrapper(lstm_model, seq_len=SEQ_LEN, n_mfcc=N_MFCC):
    def predict_proba_from_tab(X_tab):
        # forza sempre array 2D (batch_size, seq_len*n_mfcc)
        X_tab = np.atleast_2d(X_tab)
        # reshape in sequenze (batch_size, seq_len, n_mfcc)
        X_seq = X_tab.reshape(-1, seq_len, n_mfcc)
        return lstm_model.predict(X_seq, verbose=0)
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
# Grad-CAM (come prima)
# ----------------------------
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import librosa
import tensorflow as tf

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def compute_gradcam_heatmap(model, img_input, last_conv_name=None, pred_index=None):
    """
    img_input: numpy array shape (1, H, W, C), dtype float32
    restituisce heatmap normalizzata (H', W') e pred_index
    """
    img_input = np.asarray(img_input, dtype=np.float32)
    if img_input.ndim != 4 or img_input.shape[0] != 1:
        raise ValueError("img_input must be shape (1,H,W,C)")

    if last_conv_name is None:
        last_conv_name = get_last_conv_layer_name(model)
        if last_conv_name is None:
            raise ValueError("No Conv2D layer found in model")

    # modello che restituisce output del conv layer e output finale
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output]
    )

    img_tensor = tf.convert_to_tensor(img_input, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # forward pass
        conv_outputs, predictions = grad_model(img_tensor)
        # se pred_index non è dato, prendi l'argmax delle predizioni
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]).numpy())
        class_channel = predictions[:, pred_index]

    # Assicuriamoci che tape osservi conv_outputs per il calcolo dei gradienti
    tape.watch(conv_outputs)
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        # gradiente nullo: niente segnale
        H, W, K = conv_outputs.shape[1], conv_outputs.shape[2], conv_outputs.shape[3]
        return np.zeros((H, W), dtype=np.float32), pred_index

    # global average pooling sui gradienti (come in Grad-CAM)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]  # rimuove la dimensione batch
    # pesiamo ogni feature map per il suo gradiente medio
    weighted = conv_outputs * pooled_grads[tf.newaxis, tf.newaxis, :]
    heatmap = tf.reduce_sum(weighted, axis=-1)

    # ReLU e normalizzazione
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val.numpy() <= 1e-8:
        # nessun segnale significativo
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap = heatmap / (max_val + 1e-9)

    return heatmap.numpy(), pred_index


def plot_gradcam_overlay_for_file_v3(model, file_path, sample_img_for_model,
                                     sr=22050, n_mels=128, hop_length=512,
                                     last_conv_name=None, duration=30.0,
                                     cmap='magma', out_dir=None):
    """
    sample_img_for_model: array (H,W,C) o (H,W) -- se (H,W) verrà aggiunto canale
    restituisce heatmap (float array), pred_idx, out_path
    """
    # Assicuriamoci che l'input abbia canale e batch dim
    img = np.array(sample_img_for_model)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    if img.ndim == 3:
        img_input = np.expand_dims(img, axis=0).astype(np.float32)
    else:
        raise ValueError("sample_img_for_model must be 2D or 3D (H,W[,C])")

    last_conv = last_conv_name or get_last_conv_layer_name(model)
    if last_conv is None:
        raise RuntimeError("No Conv2D found in model. Can't compute Grad-CAM.")

    # compute heatmap
    heatmap, pred_idx = compute_gradcam_heatmap(model, img_input, last_conv_name=last_conv, pred_index=None)

    # stampiamo le top predictions per controllo
    try:
        probs = model.predict(img_input, verbose=0)[0]
        print("Top preds (idx:prob):", sorted([(i, float(p)) for i,p in enumerate(probs)], key=lambda x:-x[1])[:5])
    except Exception as e:
        print("Could not obtain probs:", e)

    # carichiamo l'audio e ricreiamo lo spettrogramma per avere le dimensioni esatte
    y_raw, sr = librosa.load(file_path, sr=sr, mono=True, duration=duration)
    S = librosa.feature.melspectrogram(y=y_raw, sr=sr, n_mels=n_mels, hop_length=hop_length, fmax=sr/2)
    S_db = librosa.power_to_db(S, ref=np.max)
    n_mels_calc, n_frames = S_db.shape
    time_end = (n_frames * hop_length) / sr
    extent = [0, time_end, 0, sr/2]

    # heatmap -> ridimensiona a (n_mels, n_frames)
    # cv2.resize expects (width, height) as dsize, returns array shape (height, width)
    heat_resized = cv2.resize(heatmap, (n_frames, n_mels_calc), interpolation=cv2.INTER_CUBIC)
    # heat_resized ora ha shape (n_mels_calc, n_frames) se tutto ok
    # Normalizziamo in [0,1]
    if np.nanmax(heat_resized) > 0:
        heat_resized = heat_resized - np.nanmin(heat_resized)
        heat_resized = heat_resized / (np.nanmax(heat_resized) + 1e-9)
    else:
        print("[Warning] Grad-CAM heatmap appears empty (all zeros). Saving figure without overlay.")

    # costruiamo una mappa RGBA dalla heatmap usando una colormap (jet)
    colored_heat = cm.get_cmap('jet')(heat_resized)  # shape (H, W, 4), valori in [0,1]
    # impostiamo alpha in base al valore della heatmap (es. 0..0.6)
    alpha_mask = np.clip(heat_resized, 0, 1) * 0.6
    colored_heat[..., 3] = alpha_mask

    # plot
    fig = plt.figure(figsize=(14,6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.25)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    # spettrogramma sinistro (solo mel)
    im0 = ax0.imshow(S_db, origin='lower', aspect='auto', extent=extent, cmap=cmap)
    ax0.set_title(f"Mel-spectrogram (dB)\n{os.path.basename(file_path)}")
    ax0.set_xlabel("Tempo (s)")
    ax0.set_ylabel("Frequenza (Hz)")

    # spettrogramma destro + overlay grad-cam colorato
    im1 = ax1.imshow(S_db, origin='lower', aspect='auto', extent=extent, cmap=cmap)
    # overlay RGBA: dobbiamo passare extent e origin per allineare
    ax1.imshow(colored_heat, origin='lower', aspect='auto', extent=extent, interpolation='bilinear')
    ax1.set_title(f"Grad-CAM overlay (pred class idx={pred_idx})")
    ax1.set_xlabel("Tempo (s)")
    ax1.set_ylabel("Frequenza (Hz)")

    # colorbar relativa al mel-spectrogram (dB)
    cbar = fig.colorbar(im0, cax=cax, format='%+2.0f dB')
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
        plt.show(block=False)
    except Exception:
        plt.pause(0.5)

    return heatmap, pred_idx, out_path



# ----------------------------
# Helper: esecuzione modelli tabulari con e senza PCA/LDA
# ----------------------------
def run_tabular_families(X_train, X_test, y_train, y_test, classes, feature_names):
    results = {}

    classifiers = {
        'rf': RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1),
        'svm': SVC(probability=True, kernel='rbf', random_state=RANDOM_STATE),
        'lr': LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        'knn': KNeighborsClassifier(n_neighbors=5)
    }

    # Funzione helper per training + metriche
    def fit_and_evaluate(pipe, key):
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test) if hasattr(pipe.named_steps['clf'], 'predict_proba') else None
        results[key] = {
            'model': pipe,
            'metrics': compute_metrics(y_test, y_pred, y_proba=y_proba, classes=classes),
            'y_pred': y_pred,
            'y_proba': y_proba
        }

    # ----------------- Baseline -----------------
    for name, clf in classifiers.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        fit_and_evaluate(pipe, f'{name}_baseline')

    # ----------------- PCA -----------------
    pca_n = PCA_VAR_EXPL if USE_PCA_VAR_EXPL else PCA_N_COMPONENTS_TAB
    for name, clf in classifiers.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=pca_n, svd_solver='full', random_state=RANDOM_STATE)),
            ('clf', clf)
        ])
        fit_and_evaluate(pipe, f'{name}_pca')

    # ----------------- LDA -----------------
    n_features, n_samples = X_train.shape[1], X_train.shape[0]
    lda_components = len(classes) - 1
    preproc = [('scaler', StandardScaler())]
    if APPLY_PCA_BEFORE_LDA_WHEN_NF_GT_NS and (n_features > n_samples):
        preproc.append(('pca', PCA(n_components=PCA_FOR_LDA_VAR, svd_solver='full', random_state=RANDOM_STATE)))
    preproc.append(('lda', LDA(n_components=lda_components)))

    for name, clf in classifiers.items():
        pipe = Pipeline(preproc + [('clf', clf)])
        fit_and_evaluate(pipe, f'{name}_lda')

    return results

from pathlib import Path

# Lista dei possibili percorsi
possible_paths = [
    Path(r"C:\Users\catal\OneDrive\Desktop\GTZAN\genres_original"),
    Path(r"C:\Users\veryf\Desktop\GTZAN\genres_original")
]

# Trova il primo percorso che esiste
DATASET_PATH_genre = next((p for p in possible_paths if p.exists()), None)

if DATASET_PATH_genre is None:
    raise FileNotFoundError("Nessuno dei percorsi specificati esiste.")
else:
    print(f"Dataset trovato in: {DATASET_PATH_genre}")



def file_for_genre(genre, idx=1):
    """Restituisce il path completo di un file wav di un certo genere"""
    fname = f"{genre}.{idx:05d}.wav"   # es. jazz.00001.wav
    return DATASET_PATH_genre / genre / fname

def plot_time_and_freq(files):
    n = len(files)
    fig, axs = plt.subplots(n, 2, figsize=(12, 3.2*n), squeeze=False)

    for i, (genre, filepath) in enumerate(files.items()):
        # --- Caricamento audio ---
        y, sr = librosa.load(filepath, sr=22050)

        # --- Dominio del tempo ---
        librosa.display.waveshow(y, sr=sr, ax=axs[i, 0])
        axs[i, 0].set_title(f"{genre} — dominio del tempo")
        axs[i, 0].set_xlabel("Tempo (s)")
        axs[i, 0].set_ylabel("Ampiezza")
        axs[i, 0].set_ylim(-1, 1)          # fisso da -1 a 1
        axs[i, 0].set_yticks([-1, 0, 1])   # solo questi tre valori

        # --- Dominio della frequenza ---
        N = len(y)
        spectrum = np.fft.rfft(y)
        freq = np.fft.rfftfreq(N, d=1/sr)
        amp = np.abs(spectrum)

        # Normalizzazione a [-1,1] per rispettare lo stesso asse
        amp_norm = amp / (amp.max() + 1e-12)
        amp_m1_1 = amp_norm * 2.0 - 1.0

        axs[i, 1].plot(freq, amp_m1_1, linewidth=0.8)
        axs[i, 1].set_title(f"{genre} — spettro in frequenza (normalizzato)")
        axs[i, 1].set_xlabel("Frequenza (Hz)")
        axs[i, 1].set_ylabel("Ampiezza (norm.)")
        axs[i, 1].set_ylim(-1, 1)
        axs[i, 1].set_yticks([-1, 0, 1])

    plt.tight_layout()
    plt.show()


# ----------------------------
# MAIN
# ----------------------------
def main():
    # scegli 1 file per genere
    files = {
        "jazz":  file_for_genre("jazz", 1),
        "rock":  file_for_genre("rock", 1),
        "metal": file_for_genre("metal", 1),
    }

    # fai il plot tempo + frequenza
    plot_time_and_freq(files)

    print("1) Costruzione dataset (estrazione MFCC, mel, MFCC-image, fuzzy features)...")
    df_tab, X_seq, X_mel, X_mfcc_img, y = build_datasets(DATASET_PATH, max_files=MAX_FILES)
    print("Shapes:", df_tab.shape, X_seq.shape, X_mel.shape, X_mfcc_img.shape)
    le = LabelEncoder(); y_enc = le.fit_transform(y); classes = le.classes_
    print("Generi:", list(classes))

    # --- INIZIO MODIFICA: Visualizzazione di un Mel-spectrogramma e di una MFCC-image di esempio ---
    print("\nVisualizzazione di un Mel-spectrogramma di esempio per il primo file estratto:")

    sample_file_path = df_tab['file'].iloc[0]   # percorso primo file
    sample_genre_label = df_tab['genre'].iloc[0]  # etichetta primo file

    plot_mel_spectrogram_with_db(sample_file_path, label=sample_genre_label, duration=3.0) 

    print("\nVisualizzazione di una MFCC-image di esempio (primo campione estratto):")
    first_mfcc_image = X_mfcc_img[0]  # primo campione

    plt.figure(figsize=(6, 6))
    plt.imshow(first_mfcc_image, cmap='viridis', origin='lower', aspect='auto')
    plt.title(f"MFCC-image per {sample_genre_label}")
    plt.colorbar(label='Ampiezza Normalizzata')
    plt.xlabel("Tempo (frame riscalati)")
    plt.ylabel("MFCC Coefficienti (riscalati)")
    plt.tight_layout()
    plt.show()
# --- FINE MODIFICA ---


    # Tabular prep (includiamo fuzzy)
    X_tab_df = df_tab.drop(columns=["genre","file"]).copy()
    feature_names = X_tab_df.columns.tolist()
    X_tab = X_tab_df.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_tab, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc)

    print("\n2) Training modelli tabulari: baseline, PCA e LDA...")
    tab_results = run_tabular_families(X_train, X_test, y_train, y_test, classes, feature_names)
    for key, res in tab_results.items():
        print(f"\n-- {key.upper()} --")
        print("Metrics:", json.dumps(res['metrics'], indent=2))
        y_pred = res['y_pred']
        print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion(cm, classes, title=f"{key.upper()} Confusion Matrix")

    # CNN su mel-spectrogram
    print("\n3) CNN su mel-spectrogram (training – invariata, senza PCA/LDA)...")
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

    # CNN su MFCC-image (invariata)
    print("\n4) CNN su MFCC-image (training – invariata, senza PCA/LDA)...")
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


    # LSTM su sequenze MFCC
    print("\n5a) LSTM su sequenze MFCC (senza PCA)...")
    N, seq_len, n_mfcc = X_seq.shape
    X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
        X_seq, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=RANDOM_STATE)

    lstm_model_no_pca = build_lstm(seq_len, n_mfcc, n_classes=len(classes))
    es_lstm_no_pca = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history_lstm_no_pca = lstm_model_no_pca.fit(X_seq_train, y_seq_train,
                                                validation_data=(X_seq_test, y_seq_test),
                                                epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
                                                callbacks=[es_lstm_no_pca], verbose=2)
    lstm_proba_no_pca = lstm_model_no_pca.predict(X_seq_test)
    lstm_pred_no_pca = np.argmax(lstm_proba_no_pca, axis=1)
    print("LSTM (senza PCA) accuracy:", accuracy_score(y_seq_test, lstm_pred_no_pca))
    print(classification_report(y_seq_test, lstm_pred_no_pca, target_names=classes, zero_division=0))
    plot_confusion(confusion_matrix(y_seq_test, lstm_pred_no_pca), classes, title="LSTM (senza PCA) Confusion")
    plot_training_curves(history_lstm_no_pca, title_prefix="LSTM (MFCC seq senza PCA)")

    print("\n5b) LSTM su sequenze MFCC con PCA per frame...")
    X_seq_flat = X_seq.reshape(N, seq_len * n_mfcc)
    scaler_seq_global = StandardScaler().fit(X_seq_flat)
    X_seq_flat_s = scaler_seq_global.transform(X_seq_flat)
    X_seq_s = X_seq_flat_s.reshape(N, seq_len, n_mfcc)

    X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
        X_seq_s, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=RANDOM_STATE)

    X_frames_train = X_seq_train.reshape(-1, n_mfcc)
    pca_seq = PCA(n_components=PCA_SEQ_N_COMPONENTS, svd_solver='full', random_state=RANDOM_STATE)
    pca_seq.fit(X_frames_train)

    X_seq_train_red = pca_seq.transform(X_seq_train.reshape(-1, n_mfcc)).reshape(X_seq_train.shape[0], seq_len, PCA_SEQ_N_COMPONENTS)
    X_seq_test_red  = pca_seq.transform(X_seq_test.reshape(-1, n_mfcc)).reshape(X_seq_test.shape[0],  seq_len, PCA_SEQ_N_COMPONENTS)

    lstm_model_pca = build_lstm(seq_len, PCA_SEQ_N_COMPONENTS, n_classes=len(classes))
    es_lstm_pca = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history_lstm_pca = lstm_model_pca.fit(X_seq_train_red, y_seq_train,
                                          validation_data=(X_seq_test_red, y_seq_test),
                                          epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
                                          callbacks=[es_lstm_pca], verbose=2)
    lstm_proba_pca = lstm_model_pca.predict(X_seq_test_red)
    lstm_pred_pca = np.argmax(lstm_proba_pca, axis=1)
    print("LSTM (con PCA) accuracy:", accuracy_score(y_seq_test, lstm_pred_pca))
    print(classification_report(y_seq_test, lstm_pred_pca, target_names=classes, zero_division=0))
    plot_confusion(confusion_matrix(y_seq_test, lstm_pred_pca), classes, title="LSTM (PCA sui frame) Confusion")
    plot_training_curves(history_lstm_pca, title_prefix="LSTM (MFCC seq + PCA)")


    # ROC AUC per RF migliore tra le tre varianti (se disponibile proba)
    print("\n6) ROC AUC (one-vs-rest) per il migliore RF tra baseline/PCA/LDA:")
    rf_keys = [k for k in tab_results.keys() if k.startswith('rf_')]
    best_key = None; best_acc = -1.0
    for k in rf_keys:
        acc = tab_results[k]['metrics']['accuracy']
        if acc is not None and acc > best_acc:
            best_acc = acc; best_key = k
    if best_key is not None and tab_results[best_key]['y_proba'] is not None:
        try:
            y_bin = label_binarize(y_test, classes=np.arange(len(classes)))
            rf_auc = roc_auc_score(y_bin, tab_results[best_key]['y_proba'], average='macro', multi_class='ovr')
            print(f"RF ({best_key}) ROC AUC (ovr macro):", rf_auc)
        except Exception as e:
            print("ROC AUC error:", e)

    # Grad-CAM example (mel CNN)
    print("\n7) Grad-CAM example on mel-CNN:")
    last_conv = None
    for layer in reversed(cnn_model.layers):
        if isinstance(layer, layers.Conv2D):
            last_conv = layer.name
            break

    if last_conv:
        sample_idx = 0
        sample_file = files_test[sample_idx]
        sample_img_for_model = X_mel_test[sample_idx] # [6]
        """heatmap, pred, out_path = plot_gradcam_overlay_for_file_v3(cnn_model, sample_file, sample_img_for_model,
                                                                   sr=SAMPLE_RATE, n_mels=128, hop_length=512, duration=30.0)
        try:
            if os.name == 'nt':
                os.startfile(out_path)
        except Exception:
            pass
    else:
        print("No Conv2D layer found for Grad-CAM.")"""


    # Prepara l'input per la funzione compute_gradcam_heatmap [5]
    # Questa parte replica la preparazione dell'input all'interno di plot_gradcam_overlay_for_file_v3 [1]
    img_for_gradcam_input = np.array(sample_img_for_model)
    if img_for_gradcam_input.ndim == 2:
        img_for_gradcam_input = img_for_gradcam_input[..., np.newaxis] # Aggiunge la dimensione del canale
    img_input_reshaped = np.expand_dims(img_for_gradcam_input, axis=0).astype(np.float32) # Aggiunge la dimensione del batch

    # 1. Stampa il Mel-spectrogram originale separatamente
    print("Visualizzazione del Mel-spectrogram originale...")
    try:
        y_raw, sr_loaded = librosa.load(sample_file, sr=SAMPLE_RATE, mono=True, duration=30.0) # Carica audio [7]
        # Calcola Mel-spectrogram [7]
        S = librosa.feature.melspectrogram(y=y_raw, sr=sr_loaded, n_mels=128, hop_length=512, fmax=sr_loaded/2)
        S_db = librosa.power_to_db(S, ref=np.max) # Converti in dB [7]

        n_frames_calc = S_db.shape[6]
        time_end_calc = (n_frames_calc * 512) / sr_loaded # hop_length = 512
        extent_calc = [0, time_end_calc, 0, sr_loaded/2]

        plt.figure(figsize=(10, 5))
        # Plot dello spettrogramma [2]
        im_mel = plt.imshow(S_db, origin='lower', aspect='auto', extent=extent_calc, cmap='magma')
        plt.colorbar(im_mel, format='%+2.0f dB', label='dB')
        plt.title(f"Mel-spectrogram (dB)\n{os.path.basename(sample_file)}") # [2]
        plt.xlabel("Tempo (s)") # [2]
        plt.ylabel("Frequenza (Hz)") # [2]
        plt.tight_layout()
        plt.show() # Mostra la figura

    except Exception as e:
        print(f"Errore durante la generazione del Mel-spectrogram separato: {e}")


    # 2. Calcola e stampa la heatmap Grad-CAM pura
    print("Calcolo e visualizzazione della heatmap Grad-CAM pura...")
    # Calcola la heatmap Grad-CAM [4]
    heatmap, pred_idx = compute_gradcam_heatmap(cnn_model, img_input_reshaped, last_conv_name=last_conv)

    plt.figure(figsize=(8, 6))
    # Visualizza la heatmap [8]
    plt.imshow(heatmap, cmap='jet', aspect='auto') # 'jet' è una colormap comune per le heatmap
    plt.colorbar(label='Intensità di Attivazione')
    plt.title(f"Grad-CAM Heatmap Pura (Indice Classe Predetta: {pred_idx})")
    plt.tight_layout()
    plt.show() # Mostra la figura

    # 3. Stampa l'overlay Grad-CAM sul Mel-spectrogram (già gestito dalla funzione esistente)
    print("Visualizzazione dell'overlay Grad-CAM sul Mel-spectrogram (già combinato in una figura)...")
    # Questa chiamata genera la figura con due subplot: mel-spectrogram originale e l'overlay [2, 3]
    # e salva l'immagine [9]
    heatmap_overlay, pred_overlay, out_path = plot_gradcam_overlay_for_file_v3(cnn_model, sample_file, sample_img_for_model,
                                            sr=SAMPLE_RATE, n_mels=128, hop_length=512, duration=30.0,
                                            out_dir=os.getcwd()) # Assicura il salvataggio nella directory corrente
    try:
        if os.name == 'nt':
            os.startfile(out_path) # [6]
    except Exception: # [10]
        pass

    else:
        print("Nessun layer Conv2D trovato per Grad-CAM.") # [10]

   
   
   
    # LIME example per LSTM
    print("\n8) LIME explanation sul miglior modello LSTM:")
    best_lstm_model = lstm_model_no_pca

    # Creo la funzione predict_proba specifica per LSTM
    predict_proba_for_lstm = make_lstm_proba_wrapper(best_lstm_model, seq_len=seq_len, n_mfcc=n_mfcc)

    n_features = seq_len * n_mfcc
    feature_names_lime = [f"mfcc_{i}" for i in range(n_features)]
    
    # Creo l'explainer LIME
    explainer_lime = LimeTabularExplainer(
        training_data=X_seq_train.reshape(X_seq_train.shape[0], -1),
        feature_names=feature_names_lime,
        class_names=list(classes),
        mode='classification'
    )
    i = 0
    try:
        exp = explainer_lime.explain_instance(
            X_seq_test[i].reshape(-1),
            predict_proba_for_lstm,
            num_features=10
        )
        print("LIME explanation for LSTM sample", i, "->", exp.as_list())
        try:
            fig = exp.as_pyplot_figure()
            plt.show()
        except Exception:
            pass
    except Exception as e:
        print("LIME error:", e)


    # t-SNE su MFCC mean vectors
    print("\n9) t-SNE su vettori MFCC mean (esempio con max 1000 sample):")
    try:
        plot_tsne_on_mfcc_vectors(df_tab, n_samples=1000)
    except Exception as e:
        print("t-SNE error:", e)

    # Metriche riassuntive
    print("\n10) Metriche riassuntive (tabular baseline/PCA/LDA, CNN, LSTM):")
    summary = {}
    for key, res in tab_results.items():
        summary[key] = res['metrics']
    summary['cnn_mel'] = compute_metrics(y_mel_test, cnn_pred, y_proba=cnn_proba, classes=classes)
    summary['cnn_mfcc'] = compute_metrics(y_mfcc_te, mfcc_pred, y_proba=mfcc_proba, classes=classes)
    summary['lstm_pca_seq'] = compute_metrics(y_seq_test, lstm_pred_pca, y_proba=lstm_proba_pca, classes=classes)
    print(json.dumps(summary, indent=2, default=str))

    print("\nPipeline completata.")

if __name__ == "__main__":
    main()