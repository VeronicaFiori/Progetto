""" This module prepares midi file data and feeds it to the neural network for training """
import glob
import pickle
import numpy
from pathlib import Path
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.layers import BatchNormalization as BatchNorm
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
    train(model, network_input, network_output)

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
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ Create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
    """ Train the neural network """
    filepath = "weights.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    model.fit(network_input, network_output, epochs=1, batch_size=128, callbacks=[checkpoint])

if __name__ == '__main__':
    train_network()
