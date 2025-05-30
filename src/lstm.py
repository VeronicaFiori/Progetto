""" This module prepares midi file data and feeds it to the neural network for training """
import glob
import pickle
import numpy as np
from pathlib import Path
from music21 import converter, instrument, note, chord

import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization as BatchNorm
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

# Ricerca automatica della cartella MIDI
possible_desktops = [
    Path.home() / "OneDrive" / "Desktop" / "dati" / "midi",
    Path.home() / "Desktop" / "dati" / "midi"
]

for path in possible_desktops:
    if path.exists():
        MIDI_DIR = path
        break
else:
    print("❌ Cartella 'dati/midi' non trovata né su OneDrive né su Desktop.")
    exit(1)


def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()
    n_vocab = len(set(notes))
    network_input, network_output = prepare_sequences(notes, n_vocab)
    model = create_network(network_input, n_vocab)
    history = train(model, network_input, network_output)
    plot_training_history(history)


def get_notes():
    """ Get all the notes and chords from the midi files in the target directory """
    notes = []

    for file in MIDI_DIR.glob("*.mid"):
        print(f"Parsing {file}")
        try:
            midi = converter.parse(file)
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() if s2 else midi.flat.notes
        except Exception as e:
            print(f"Errore nel parsing di {file}: {e}")
            continue

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    Path("data").mkdir(exist_ok=True)
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100
    pitchnames = sorted(set(notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output)

    return (network_input, network_output)


def create_network(network_input, n_vocab):
    """ Build the LSTM model """
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
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def train(model, network_input, network_output):
    """ Train the neural network """
    filepath = "weights.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    history = model.fit(
        network_input,
        network_output,
        epochs=1,
        batch_size=128,
        validation_split=0.2,  # <- opzionale: usa il 20% dei dati per la validazione
        callbacks=[checkpoint]
    )
    return history


def plot_training_history(history):
    """ Visualizza il grafico della loss e dell'accuracy """
    plt.figure(figsize=(12, 5))
    sns.set(style="whitegrid")

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Loss (Train)", color="blue")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Loss (Val)", color="red")
    plt.title("Andamento della Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    if "accuracy" in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Accuracy (Train)", color="green")
        if "val_accuracy" in history.history:
            plt.plot(history.history["val_accuracy"], label="Accuracy (Val)", color="orange")
        plt.title("Andamento dell'Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    train_network()
