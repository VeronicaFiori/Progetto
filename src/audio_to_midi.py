# wav_to_midi.py
import os
import librosa
import numpy as np
import pretty_midi
from pathlib import Path

def convert_wav_to_midi(wav_path, midi_path):
    y, sr = librosa.load(wav_path)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0, sr=sr)
    
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # piano
    
    prev_note = None
    for t, freq in zip(times, f0):
        if freq is not None and not np.isnan(freq):
            midi_note = int(librosa.hz_to_midi(freq))
            if prev_note is None or prev_note.pitch != midi_note:
                if prev_note is not None:
                    prev_note.end = t
                    instrument.notes.append(prev_note)
                prev_note = pretty_midi.Note(velocity=100, pitch=midi_note, start=t, end=t + 0.1)
        else:
            if prev_note is not None:
                prev_note.end = t
                instrument.notes.append(prev_note)
                prev_note = None
    if prev_note is not None:
        instrument.notes.append(prev_note)

    midi.instruments.append(instrument)
    midi.write(midi_path)

def process_folder(wav_folder, midi_output_folder):
    os.makedirs(midi_output_folder, exist_ok=True)
    for file in os.listdir(wav_folder):
        if file.endswith(".wav"):
            wav_path = os.path.join(wav_folder, file)
            midi_path = os.path.join(midi_output_folder, file.replace(".wav", ".mid"))
            convert_wav_to_midi(wav_path, midi_path)
            print(f"✓ MIDI generato: {midi_path}")

if __name__ == "__main__":
    # Trova il percorso Desktop in modo compatibile con o senza OneDrive
    possible_desktops = [
        Path.home() / "OneDrive" / "Desktop" / "dati",
        Path.home() / "Desktop" / "dati"
    ]

    for path in possible_desktops:
        if path.exists():
            base_data = path
            break
    else:
        print("❌ Cartella 'dati' non trovata né su OneDrive né su Desktop.")
        exit(1)

    wav_folder = base_data / "wav"
    midi_folder = base_data / "midi"

    if not wav_folder.exists():
        print(f"❌ La cartella non esiste: {wav_folder}")
        exit(1)

    process_folder(str(wav_folder), str(midi_folder))
