
import os
from pathlib import Path
import tensorflow as tf
from src.preprocessing import load_dataset
from src.model_lstm import build_lstm_model

# Hyperparametri
SEQUENCE_LENGTH = 50
EPOCHS = 30
BATCH_SIZE = 64

# Trova la cartella "midi" in modo compatibile con o senza OneDrive
possible_paths = [
    Path.home() / "OneDrive" / "Desktop" / "dati" / "midi",
    Path.home() / "Desktop" / "dati" / "midi"
]

for path in possible_paths:
    if path.exists():
        DATA_DIR = path
        break
else:
    print("❌ Cartella 'midi' non trovata né su OneDrive né su Desktop.")
    exit(1)

MODEL_SAVE_PATH = Path("models") / "lstm_model.keras"
os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)

def main():
    print(" Caricamento dataset...")
    X, y = load_dataset(str(DATA_DIR), sequence_length=SEQUENCE_LENGTH)
    print(f"✓ Dataset caricato: {X.shape[0]} sequenze")

    vocab_size = 128  # Nota MIDI
    model = build_lstm_model(SEQUENCE_LENGTH, vocab_size)
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_SAVE_PATH), save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    print(" Inizio training...")
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks
    )

    print(f"✓ Modello salvato in: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()