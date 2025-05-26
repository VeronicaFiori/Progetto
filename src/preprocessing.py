
# src/preprocessing.py

import pretty_midi
import numpy as np
import os
from pathlib import Path
import librosa

def midi_to_note_sequence(midi_file):
    """Converte un file MIDI in una lista di note (pitch, start, end)."""
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append((note.pitch, note.start, note.end))
    notes.sort(key=lambda x: x[1])  # Ordina per start time
    return notes

def notes_to_training_sequences(notes, sequence_length=50):
    """Prepara sequenze di training a partire da una lista di note."""
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
    """Carica tutti i file MIDI da una cartella e restituisce X e y."""
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


def extract_log_mel_spectrogram(mid_path, n_mels=128, duration=30, sr=22050):
    y, _ = librosa.load(mid_path, sr=sr, duration=duration)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec)
    return log_mel_spec
