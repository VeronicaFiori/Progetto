import pickle
from pathlib import Path
import numpy as np
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.layers import BatchNormalization as BatchNorm

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def generate():
    """Generate a piano midi file using a trained model."""
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)
    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)

    model = create_network(normalized_input, n_vocab)
    model.load_weights('weights.keras')

    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)
    plot_statistics(prediction_output)

def prepare_sequences(notes, pitchnames, n_vocab):
    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    sequence_length = 100
    network_input = []
    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[n] for n in sequence_in])

    normalized_input = np.reshape(network_input, (len(network_input), sequence_length, 1))
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def sample_with_temperature(predictions, temperature=0.8):
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions + 1e-8) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(predictions), p=predictions)

def generate_notes(model, network_input, pitchnames, n_vocab, num_notes=500):
    start = np.random.randint(0, len(network_input) - 1)
    int_to_note = {number: note for number, note in enumerate(pitchnames)}
    pattern = network_input[start]
    prediction_output = []

    for _ in range(num_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = sample_with_temperature(prediction[0], temperature=0.8)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output

def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            chord_notes = [note.Note(int(n)) for n in notes_in_chord]
            for n in chord_notes:
                n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    output_dir = Path("output/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Trova il prossimo numero disponibile per il file MIDI
    existing_files = list(output_dir.glob("test_output*.mid"))
    existing_indices = [
        int(f.stem.replace("test_output", ""))
        for f in existing_files if f.stem.replace("test_output", "").isdigit()
    ]
    next_index = max(existing_indices, default=0) + 1
    output_path = output_dir / f"test_output{next_index}.mid"

    # Salva il file MIDI
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_path)
    print("✅ File MIDI generato in:", output_path)


def plot_statistics(prediction_output):
    sns.set(style="whitegrid")
    note_counts = Counter(prediction_output)
    chords = [p for p in prediction_output if '.' in p or p.isdigit()]
    single_notes = [p for p in prediction_output if not ('.' in p or p.isdigit())]

    # Frequenze
    plt.figure(figsize=(14, 5))
    top_notes = note_counts.most_common(20)
    labels, values = zip(*top_notes)
    plt.bar(labels, values, color="skyblue")
    plt.title("🎼 Frequenze delle 20 note/accordi più comuni")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Note vs accordi
    plt.figure(figsize=(6, 4))
    plt.pie([len(single_notes), len(chords)], labels=["Note", "Accordi"], autopct="%1.1f%%", startangle=140, colors=["#7fcdbb", "#ef8a62"])
    plt.title("Distribuzione: Note vs Accordi")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    generate()
