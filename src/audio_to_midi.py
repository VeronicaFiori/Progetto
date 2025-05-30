# wav_to_midi.py
"""
import os
import librosa
import numpy as np
import pretty_midi
from pathlib import Path

def convert_wav_to_midi(wav_path, midi_path):
    y, sr = librosa.load(wav_path)
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0, sr=sr)

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    prev_note = None
    for t, freq in zip(times, f0):
        if not np.isnan(freq):
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
"""
# wav_to_midi.py
import os
from pathlib import Path
from basic_pitch.inference import predict

import pretty_midi
import soundfile as sf
import librosa

def normalize_audio(input_path, output_path):
    y, sr = librosa.load(input_path, sr=None)
    y = y / max(abs(y))  # Normalizzazione
    sf.write(output_path, y, sr)


def convert_output_to_midi(output, midi_path):
    note_events = output["note"]  # Lista di dizionari
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for event in note_events:
        start = float(event["start"])
        end = float(event["end"])
        pitch = int(event["pitch"])
        velocity = int(event["amplitude"] * 127)  # Scala ampiezza a 0–127

        note_obj = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=end
        )
        instrument.notes.append(note_obj)

    pm.instruments.append(instrument)
    pm.write(midi_path)
    print(f"✓ MIDI creato manualmente: {midi_path}")

def convert_wav_to_midi(wav_path, midi_path):
    temp_path = wav_path.replace(".wav", "_normalized.wav")

    normalize_audio(wav_path, temp_path)

    print(f"Predicting MIDI for {temp_path}...")
    output = predict(temp_path)

    if "note" in output:
        convert_output_to_midi(output, midi_path)
    else:
        print(f"⚠️ Nessuna nota trovata in {wav_path}")

    # Pulisci il file temporaneo se vuoi
    os.remove(temp_path)





def process_folder(wav_folder, midi_output_folder):
    os.makedirs(midi_output_folder, exist_ok=True)
    for file in os.listdir(wav_folder):
        if file.endswith(".wav") or file.endswith(".mp3"):
            wav_path = os.path.join(wav_folder, file)  # ✅ NO parentesi quadre
            print(f"🎧 Elaborazione: {repr(wav_path)} ({type(wav_path)})")

            midi_path = os.path.join(midi_output_folder, file.rsplit(".", 1)[0] + ".mid")
            convert_wav_to_midi(wav_path, midi_path)


if __name__ == "__main__":
    # Percorsi compatibili con OneDrive o Desktop
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
