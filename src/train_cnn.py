import tensorflow as tf
from src.model_cnn import build_cnn_model
import os
import numpy as np
from sklearn.model_selection import train_test_split
from src.preprocessing import extract_log_mel_spectrogram
from pathlib import Path
import random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Cerca la cartella del dataset
possible_dataset_dirs = [
    Path.home() / "OneDrive" / "Desktop" / "Data" / "genres_original",
    Path.home() / "Desktop" / "Data" / "genres_original"
]

for path in possible_dataset_dirs:
    if path.exists():
        DATASET_DIR = path
        break
else:
    print("❌ Cartella dataset 'genres_original' non trovata sul Desktop/Data.")
    exit(1)

GENRES = sorted(os.listdir(DATASET_DIR))

MODEL_PATH = "models/cnn_genre_classifier.keras"
INPUT_SHAPE = (128, 128, 1)
EPOCHS = 30
BATCH_SIZE = 32

def load_data():
    X, y = [], []
    for genre_idx, genre in enumerate(GENRES):
        genre_dir = os.path.join(DATASET_DIR, genre)
        for file in os.listdir(genre_dir):
            if file.endswith(".wav"):
                try:
                    path = os.path.join(genre_dir, file)
                    spec = extract_log_mel_spectrogram(path)
                    if spec.shape[1] >= 128:
                        spec = spec[:, :128]
                        X.append(spec)
                        y.append(genre_idx)
                except Exception as e:
                    print(f"Errore su {file}: {e}")
    X = np.array(X)[..., np.newaxis]
    y = tf.keras.utils.to_categorical(y, num_classes=len(GENRES))
    return X, y

def main():
    X, y = load_data()
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    model = build_cnn_model(INPUT_SHAPE, num_classes=len(GENRES))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=[
                  tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True),
                  tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
              ])
    print(f"✅ Modello salvato in: {MODEL_PATH}")

if __name__ == "__main__":
    main()
