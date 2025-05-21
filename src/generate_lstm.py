

# src/generate_lstm.py

import numpy as np
import pretty_midi
import tensorflow as tf
import os
from src.model_lstm import build_lstm_model

SEQUENCE_LENGTH = 50
VOCAB_SIZE = 128
MODEL_PATH = "models/lstm_model.keras"
OUTPUT_DIR = "output/generated"

def generate_music(seed_sequence, model, length=100):
    """Genera nuova sequenza musicale."""
    generated = list(seed_sequence)
    for _ in range(length):
        input_seq = np.array(generated[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH)
        prediction = model.predict(input_seq, verbose=0)
        next_note = np.argmax(prediction)
        generated.append(next_note)
    return generated

def get_next_filename(directory, prefix="generated_", extension=".mid"):
    """Restituisce il prossimo nome disponibile nella forma generated_1.mid, etc."""
    os.makedirs(directory, exist_ok=True)
    existing = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
    numbers = [int(f[len(prefix):-len(extension)]) for f in existing if f[len(prefix):-len(extension)].isdigit()]
    next_number = max(numbers, default=0) + 1
    return os.path.join(directory, f"{prefix}{next_number}{extension}")

def sequence_to_midi(sequence, output_file):
    """Converte una sequenza di pitch in file MIDI."""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    start = 0
    duration = 0.5
    for pitch in sequence:
        note = pretty_midi.Note(velocity=100, pitch=int(pitch), start=start, end=start+duration)
        instrument.notes.append(note)
        start += duration
    midi.instruments.append(instrument)
    midi.write(output_file)
    print(f"🎼 File MIDI salvato come {output_file}")

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    seed_sequence = np.random.randint(0, VOCAB_SIZE, SEQUENCE_LENGTH)
    generated_sequence = generate_music(seed_sequence, model, length=200)
    
    output_file = get_next_filename(OUTPUT_DIR)
    sequence_to_midi(generated_sequence, output_file)

if __name__ == "__main__":
    main()
