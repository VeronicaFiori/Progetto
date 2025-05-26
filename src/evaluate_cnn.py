"""
import tensorflow as tf
import os
import numpy as np
from src.preprocessing import extract_log_mel_spectrogram

MODEL_PATH = "models/cnn_genre_classifier.keras"
GENRES = sorted(os.listdir("dataset"))
INPUT_SHAPE = (128, 128, 1)
GENERATED_DIR = "generated"

def classify_file(mid_path, model):
    spec = extract_log_mel_spectrogram(mid_path)
    if spec.shape[1] >= 128:
        spec = spec[:, :128]
    else:
        print(f"⚠️ File troppo corto: {mid_path}")
        return
    input_tensor = np.expand_dims(spec, axis=(0, -1))
    prediction = model.predict(input_tensor, verbose=0)[0]
    genre = GENRES[np.argmax(prediction)]
    print(f"🎵 {os.path.basename(mid_path)} → {genre}")

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    for file in os.listdir(GENERATED_DIR):
        if file.endswith(".mid"):
            classify_file(os.path.join(GENERATED_DIR, file), model)

if __name__ == "__main__":
    main()
"""
"""
import tensorflow as tf
import numpy as np
import os
from src.preprocessing import extract_log_mel_spectrogram  # la tua funzione
from pathlib import Path

MODEL_PATH = "models/cnn_genre_classifier.keras"
DATASET_DIR = Path.home() / "Desktop" / "Data" / "genres_original"

GENRES = sorted(os.listdir(DATASET_DIR))
INPUT_SHAPE = (128, 128, 1)

def load_test_data():
    X, y = [], []
    for genre_idx, genre in enumerate(GENRES):
        genre_dir = os.path.join(DATASET_DIR, genre)
        for file in os.listdir(genre_dir):
            if file.endswith(".mid"):
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
    model = tf.keras.models.load_model(MODEL_PATH)
    X_test, y_test = load_test_data()

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
"""




""""PROVA2"""
"""
import tensorflow as tf
import numpy as np
import os
from src.preprocessing import extract_log_mel_spectrogram
from pathlib import Path

MODEL_PATH = "models/cnn_genre_classifier.keras"

# Proviamo a trovare la cartella 'output/generated' in OneDrive o Desktop
possible_generated_dirs = [
    Path.home() / "OneDrive" / "Desktop" / "Progetto" / "output" / "generated",
    Path.home() / "Desktop" / "Progetto" / "output" / "generated"
]

for path in possible_generated_dirs:
    if path.exists():
        GENERATED_DIR = path
        break
else:
    print("❌ Cartella 'output/generated' non trovata né in OneDrive né in Desktop.")
    exit(1)

# Trova la cartella dataset per i generi (serve per leggere i nomi dei generi)
possible_dataset_dirs = [
    Path.home() / "OneDrive" / "Desktop" / "Data" / "genres_original",
    Path.home() / "Desktop" / "Data" / "genres_original"
]

for path in possible_dataset_dirs:
    if path.exists():
        DATASET_DIR = path
        break
else:
    print("❌ Cartella dataset 'genres_original' non trovata sul Desktop/Data o OneDrive.")
    exit(1)

GENRES = sorted(os.listdir(DATASET_DIR))

def classify_file(mid_path, model):
    spec = extract_log_mel_spectrogram(mid_path)
    if spec.shape[1] >= 128:
        spec = spec[:, :128]
    else:
        print(f"⚠️ File troppo corto: {mid_path}")
        return
    input_tensor = np.expand_dims(spec, axis=(0, -1))
    prediction = model.predict(input_tensor, verbose=0)[0]
    genre = GENRES[np.argmax(prediction)]
    print(f"🎵 {os.path.basename(mid_path)} → {genre}")


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modello caricato da {MODEL_PATH}")
    print(f"Classifico i file in: {GENERATED_DIR}")
    
    mid_files = [f for f in os.listdir(GENERATED_DIR) if f.endswith(".mid")]
    print(f"File mid trovati: {mid_files}")
    
    if not mid_files:
        print("❌ Nessun file mid da classificare.")
        return

    for file in mid_files:
        classify_file(str(GENERATED_DIR / file), model)

if __name__ == "__main__":
    main()
"""


"""PROVA3
import tensorflow as tf
import numpy as np
import os
from src.preprocessing import extract_log_mel_spectrogram
from pathlib import Path
from midi2audio import FluidSynth
import tempfile

MODEL_PATH = "models/cnn_genre_classifier.keras"

# Cartelle
possible_generated_dirs = [
    Path.home() / "OneDrive" / "Desktop" / "Progetto" / "output" / "generated",
    Path.home() / "Desktop" / "Progetto" / "output" / "generated"
]

for path in possible_generated_dirs:
    if path.exists():
        GENERATED_DIR = path
        break
else:
    print("❌ Cartella 'output/generated' non trovata né in OneDrive né in Desktop.")
    exit(1)

possible_dataset_dirs = [
    Path.home() / "OneDrive" / "Desktop" / "Data" / "genres_original",
    Path.home() / "Desktop" / "Data" / "genres_original"
]

for path in possible_dataset_dirs:
    if path.exists():
        DATASET_DIR = path
        break
else:
    print("❌ Cartella dataset 'genres_original' non trovata sul Desktop/Data o OneDrive.")
    exit(1)

GENRES = sorted(os.listdir(DATASET_DIR))

# Inizializza FluidSynth, indica il path a un soundfont (scaricalo se non ce l'hai)
# Es: https://github.com/FluidSynth/fluidsynth/wiki/SoundFonts
SOUNDFONT_PATH = "FluidR3_GM.sf2"  # metti il path corretto al tuo .sf2

fs = FluidSynth(sound_font=SOUNDFONT_PATH)

def midi_to_wav(midi_path):
    # Crea un file WAV temporaneo per la conversione
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name
    try:
        fs.midi_to_audio(midi_path, tmp_wav_path)
        return tmp_wav_path
    except Exception as e:
        print(f"Errore nella conversione MIDI->WAV di {midi_path}: {e}")
        return None

def classify_file(midi_path, model):
    wav_path = midi_to_wav(midi_path)
    if not wav_path:
        print(f"Impossibile convertire {midi_path}")
        return
    spec = extract_log_mel_spectrogram(wav_path)
    # elimina il file wav temporaneo subito dopo l'uso
    os.remove(wav_path)
    
    if spec.shape[1] >= 128:
        spec = spec[:, :128]
    else:
        print(f"⚠️ File troppo corto dopo conversione: {midi_path}")
        return
    input_tensor = np.expand_dims(spec, axis=(0, -1))
    prediction = model.predict(input_tensor, verbose=0)[0]
    genre = GENRES[np.argmax(prediction)]
    print(f"🎵 {os.path.basename(midi_path)} → {genre}")

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modello caricato da {MODEL_PATH}")
    print(f"Classifico i file in: {GENERATED_DIR}")
    
    mid_files = [f for f in os.listdir(GENERATED_DIR) if f.endswith(".mid")]
    print(f"File midi trovati: {mid_files}")
    
    if not mid_files:
        print("❌ Nessun file midi da classificare.")
        return

    for file in mid_files:
        classify_file(str(GENERATED_DIR / file), model)

if __name__ == "__main__":
    main()
"""

import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import tempfile
from src.preprocessing import extract_log_mel_spectrogram
import subprocess

import time
import fluidsynth  # pyfluidsynth

from midi2audio import FluidSynth


MODEL_PATH = "models/cnn_genre_classifier.keras"

# Proviamo a trovare la cartella 'output/generated' in OneDrive o Desktop
possible_generated_dirs = [
    Path.home() / "OneDrive" / "Desktop" / "Progetto" / "output" / "generated",
    Path.home() / "Desktop" / "Progetto" / "output" / "generated"
]

for path in possible_generated_dirs:
    if path.exists():
        GENERATED_DIR = path
        break
else:
    print("❌ Cartella 'output/generated' non trovata né in OneDrive né in Desktop.")
    exit(1)

# Trova la cartella dataset per i generi
possible_dataset_dirs = [
    Path.home() / "OneDrive" / "Desktop" / "Data" / "genres_original",
    Path.home() / "Desktop" / "Data" / "genres_original"
]

for path in possible_dataset_dirs:
    if path.exists():
        DATASET_DIR = path
        break
else:
    print("❌ Cartella dataset 'genres_original' non trovata sul Desktop/Data o OneDrive.")
    exit(1)

GENRES = sorted(os.listdir(DATASET_DIR))


#SOUNDFONT_PATH = str(Path.home() / "Desktop" / "FluidR3" / "FluidR3_GM.sf2")





def midi_to_wav(midi_path):
    soundfont_path = Path.home() / "Desktop" / "FluidR3" / "FluidR3_GM.sf2"
    print(f"[DEBUG] Verifica path SoundFont: {soundfont_path}")
    print(f"[DEBUG] Esiste il file? {soundfont_path.exists()}")
    if not soundfont_path.exists():
        print(f"❌ SoundFont non trovato: {soundfont_path}")
        return None
    #Salva solo momentaneamente il nuovo audio .wav
    #wav_path = str(Path(midi_path).with_suffix('.wav'))
    
    #salva in sotto cartella nuovo file .wav
    midi_path = Path(midi_path)

    # Cartella dove salvare i .wav
    wav_output_dir = midi_path.parent / "generated_wav"
    wav_output_dir.mkdir(exist_ok=True)  # crea la cartella se non esiste

    # Percorso finale del file .wav
    wav_path = wav_output_dir / midi_path.with_suffix('.wav').name
    #Se esiste già, non rigenerarlo!
    if wav_path.exists():
         print(f"⚠️ WAV già esistente, uso quello: {wav_path}")
         return wav_path
    try:
        result = subprocess.run([
            "fluidsynth",
            "-ni",
            "-F", str(wav_path),
            "-r", "44100",
            str(soundfont_path),
            str(midi_path)
    ], capture_output=True, text=True)


        if result.returncode != 0:
            print("❌ Errore fluidsynth:", result.stderr)
            return None

        print(f"✅ File WAV salvato in: {wav_path}")
        return wav_path

    except Exception as e:
        print(f"❌ Eccezione durante la conversione: {e}")
        return None



def classify_file(midi_path, model):
    wav_path = midi_to_wav(midi_path)
    if not wav_path:
        print(f"⚠️ Impossibile convertire {midi_path}")
        return
    spec = extract_log_mel_spectrogram(wav_path)
    #os.remove(wav_path)  # rimuovi il WAV temporaneo

    if spec.shape[1] >= 128:
        spec = spec[:, :128]
    else:
        print(f"⚠️ File troppo corto dopo conversione: {midi_path}")
        return
    input_tensor = np.expand_dims(spec, axis=(0, -1))
    prediction = model.predict(input_tensor, verbose=0)[0]
    genre = GENRES[np.argmax(prediction)]
    print(f"🎵 {os.path.basename(midi_path)} → {genre}")

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Modello caricato da: {MODEL_PATH}")
    print(f"🔍 Classifico i file in: {GENERATED_DIR}")

    mid_files = [f for f in os.listdir(GENERATED_DIR) if f.endswith(".mid")]
    print(f"🎼 File MIDI trovati: {mid_files}")

    if not mid_files:
        print("❌ Nessun file MIDI da classificare.")
        return

    for file in mid_files:
        classify_file(str(GENERATED_DIR / file), model)

if __name__ == "__main__":
    main()
