
import os
from pathlib import Path
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from src.preprocessing import load_dataset
from src.model_lstm import build_lstm_model

# Hyperparametri
SEQUENCE_LENGTH = 50
EPOCHS = 30
BATCH_SIZE = 64
VOCAB_SIZE = 128

# Percorso dataset (supporta OneDrive/Desktop)
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
VOCAB_CONFIG_PATH = Path("models") / "vocab_config.json"
LOSS_PLOT_PATH = Path("models") / "training_loss.png"
os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)

def plot_history(history, save_path):
    """Plot della loss."""
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Andamento della Loss")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Plot salvato in {save_path}")

def main():
    print("📂 Caricamento dataset MIDI...")
    X, y = load_dataset(str(DATA_DIR), sequence_length=SEQUENCE_LENGTH)
    
    if X.size == 0 or y.size == 0:
        print("❌ Nessun dato caricato. Verifica i file MIDI nella cartella.")
        exit(1)

    print(f"✅ Dataset caricato: {X.shape[0]} sequenze.")

    # Check TensorFlow e GPU
    print(f"🧠 TensorFlow version: {tf.__version__}")
    if tf.config.list_physical_devices('GPU'):
        print("🚀 GPU rilevata")
    else:
        print("⚠️ Allenamento su CPU")

    # Salva la configurazione del vocabolario
    with open(VOCAB_CONFIG_PATH, "w") as f:
        json.dump({"vocab_size": VOCAB_SIZE}, f)

    # Costruisci modello
    model = build_lstm_model(SEQUENCE_LENGTH, VOCAB_SIZE)
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_SAVE_PATH), save_best_only=True, monitor="val_loss"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]

    # Training
    print("🟢 Inizio training...")
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks
    )

    print(f"✅ Modello salvato in: {MODEL_SAVE_PATH}")
    plot_history(history, LOSS_PLOT_PATH)

if __name__ == "__main__":
    main()
