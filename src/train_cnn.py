import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.model_cnn import build_cnn_model
from src.preprocessing import extract_log_mel_spectrogram
from collections import Counter
import librosa.display
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Impostazioni di riproducibilità
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Percorsi e parametri
MODEL_PATH = "models/cnn_genre_classifier.keras"
INPUT_SHAPE = (128, 128, 1)
EPOCHS = 30
BATCH_SIZE = 32

# 🔧 Crea la directory models se non esiste
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Trova la cartella del dataset
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
                    print(f"❌ Errore su {file}: {e}")
    X = np.array(X)[..., np.newaxis]
    y = tf.keras.utils.to_categorical(y, num_classes=len(GENRES))
    return X, y

def debug_info(X, y):
    print(f"\n📊 Shape input: {X.shape} (deve essere: [N, 128, 128, 1])")
    print(f"🔍 Valori spettrogramma: min={X.min()}, max={X.max()}, mean={X.mean():.2f}")

    label_counts = Counter(np.argmax(y, axis=1))
    print("\n🎼 Distribuzione classi:")
    for i, count in sorted(label_counts.items()):
        print(f" - {GENRES[i]}: {count} esempi")

    # Visualizza uno spettrogramma campione
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(X[0].squeeze(), sr=22050, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spettrogramma di esempio")
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Accuratezza
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('📈 Accuratezza')
    plt.xlabel('Epoca')
    plt.ylabel('Accuratezza')
    plt.legend()

    # Perdita
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('📉 Perdita')
    plt.xlabel('Epoca')
    plt.ylabel('Perdita')
    plt.legend()

    plt.tight_layout()
    plt.show()


"""
def plot_confusion_matrix(model, X_val, y_val):
    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GENRES)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("🎼 Matrice di Confusione - Generi Musicali")
    plt.tight_layout()
    plt.show()
"""
def plot_confusion_matrix(model, X_val, y_val, GENRES, normalize=False):
    """
    Calcola e visualizza la matrice di confusione. Se 'normalize' è True, visualizza valori normalizzati.
    """
    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GENRES)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("🎼 Matrice di Confusione - Generi Musicali")
    plt.tight_layout()
    plt.show()

def main():
    X, y = load_data()
    debug_info(X, y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = build_cnn_model(INPUT_SHAPE, num_classes=len(GENRES))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )

    print(f"✅ Modello salvato in: {MODEL_PATH}")
    plot_training_history(history)
    plot_confusion_matrix(model, X_val, y_val, GENRES, normalize=False)



if __name__ == "__main__":
    main()
