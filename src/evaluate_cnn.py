import os
import subprocess
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from src.preprocessing import extract_log_mel_spectrogram

MODEL_PATH = "models/cnn_genre_classifier.keras"

# 🔍 Cerca directory dei file MIDI generati
possible_generated_dirs = [
    Path.home() / "OneDrive" / "Desktop" / "Progetto-1" / "output" / "generated",
    Path.home() / "Desktop" / "Progetto" / "output" / "generated"
]

for path in possible_generated_dirs:
    if path.exists():
        GENERATED_DIR = path
        break
else:
    raise FileNotFoundError("❌ Cartella 'output/generated' non trovata.")

# 🔍 Cerca directory dataset originale
possible_dataset_dirs = [
    Path.home() / "OneDrive" / "Desktop" / "Data" / "genres_original",
    Path.home() / "Desktop" / "Data" / "genres_original"
]

for path in possible_dataset_dirs:
    if path.exists():
        DATASET_DIR = path
        break
else:
    raise FileNotFoundError("❌ Cartella 'genres_original' non trovata.")

GENRES = sorted(os.listdir(DATASET_DIR))

# 🔍 Cerca SoundFont
possible_soundfonts = [
    Path.home() / "OneDrive" / "Desktop" / "FluidR3" / "FluidR3_GM.sf2",
    Path.home() / "Desktop" / "FluidR3" / "FluidR3_GM.sf2"
]

for path in possible_soundfonts:
    if path.exists():
        SOUNDFONT_PATH = path
        break
else:
    raise FileNotFoundError("❌ SoundFont non trovato.")


def midi_to_wav(midi_path):
    midi_path = Path(midi_path)
    wav_output_dir = midi_path.parent / "generated_wav"
    wav_output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = wav_output_dir / midi_path.with_suffix('.wav').name

    if wav_path.exists():
        print(f"⚠️ WAV già presente: {wav_path}")
        return wav_path

    result = subprocess.run([
        "fluidsynth", "-ni", "-F", str(wav_path), "-r", "44100",
        str(SOUNDFONT_PATH), str(midi_path)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Errore fluidsynth:", result.stderr)
        return None

    print(f"✅ WAV generato: {wav_path}")
    return wav_path


def classify_file(midi_path, model, predictions_counter, verbose=False):
    import librosa
    import librosa.display

    wav_path = midi_to_wav(midi_path)
    if not wav_path:
        return

    try:
        spec = extract_log_mel_spectrogram(wav_path)
    except Exception as e:
        print(f"❌ Errore spettrogramma: {e}")
        return

    if spec.shape[1] < 128:
        print(f"⚠️ File troppo corto: {wav_path}")
        return

    spec = spec[:, :128]
    input_tensor = np.expand_dims(spec, axis=(0, -1))
    prediction = model.predict(input_tensor, verbose=0)[0]
    genre_index = np.argmax(prediction)
    genre = GENRES[genre_index]
    predictions_counter[genre] += 1

    file_name = Path(midi_path).name
    print(f"\n🎵 {file_name} → {genre}")
    print("📊 Probabilità:")
    for i, g in enumerate(GENRES):
        print(f"   - {g:<10}: {prediction[i]:.2%}")

    # Carica audio per spettrogramma classico
    y, sr = librosa.load(wav_path, sr=22050)

    # Spettrogramma classico (STFT)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Visualizza entrambi gli spettrogrammi
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))

    # Log-Mel Spettrogramma
    axs[0].imshow(spec, aspect="auto", origin="lower", cmap="magma")
    axs[0].set_title("Log-Mel Spettrogramma")
    axs[0].set_xlabel("Frame")
    axs[0].set_ylabel("Mel Band")

    # Spettrogramma STFT classico
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='viridis', ax=axs[1])
    axs[1].set_title("Spettrogramma (Log-Frequenze)")
    fig.colorbar(img, ax=axs[1], format="%+2.0f dB")

    plt.suptitle(f"🎼 {file_name}", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_genre_distribution(counter):
    if not counter:
        print("⚠️ Nessuna classificazione disponibile per il grafico.")
        return

    genres = list(counter.keys())
    counts = list(counter.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(genres, counts, color="skyblue", edgecolor="black")
    plt.title("Distribuzione dei generi nei file MIDI generati", fontsize=14)
    plt.xlabel("Genere")
    plt.ylabel("Frequenza")
    plt.xticks(rotation=45)

    # Annotazioni sulle barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.3, f"{int(height)}", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def main():
    print(f"📥 Carico il modello da: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    mid_files = [f for f in os.listdir(GENERATED_DIR) if f.endswith(".mid")]
    if not mid_files:
        print("❌ Nessun file MIDI trovato.")
        return

    print(f"🎼 {len(mid_files)} file MIDI trovati.")
    predictions_counter = Counter()

    for file in mid_files:
        classify_file(GENERATED_DIR / file, model, predictions_counter, verbose=False)

    print("\n📊 Risultati:")
    for genre, count in predictions_counter.items():
        print(f" - {genre}: {count}")

    plot_genre_distribution(predictions_counter)


if __name__ == "__main__":
    main()
