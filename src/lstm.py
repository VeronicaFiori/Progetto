
import pickle
import numpy as np
from pathlib import Path
from music21 import converter, instrument, note, chord, key

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization as BatchNorm

import matplotlib.pyplot as plt
import seaborn as sns


# 🔍 Trova cartella MIDI
MIDI_DIR = next((p for p in [
    Path.home() / "OneDrive" / "Desktop" / "dati" / "midi",
    Path.home() / "Desktop" / "dati" / "midi"
] if p.exists()), None)

if MIDI_DIR is None:
    raise FileNotFoundError("❌ Cartella 'dati/midi' non trovata.")


def extract_notes(score):
    """Estrai note e accordi da uno spartito"""
    notes = []
    for el in score.flat.notes:
        if isinstance(el, note.Note):
            notes.append(str(el.pitch))
        elif isinstance(el, chord.Chord):
            notes.append('.'.join(str(n) for n in el.normalOrder))
    return notes


def smart_transpose(score, semitones):
    """Trasponi il MIDI rispettando la tonalità (major/minor)"""
    try:
        orig_key = score.analyze('key')
        transposed = score.transpose(semitones)
        new_key = transposed.analyze('key')

        # Evita trasposizioni che cambiano tipo (es. maggiore → minore)
        if orig_key.mode == new_key.mode:
            return transposed
    except:
        pass
    return None


def get_notes(sequence_length=100, min_len=100):
    """Estrai le note da MIDI e applica augmentation musicale"""
    all_notes = []

    for file in MIDI_DIR.glob("*.mid"):
        try:
            midi = converter.parse(file)
            notes = extract_notes(midi)
            if len(notes) < min_len:
                continue
            all_notes.extend(notes)

            # Trasposizione sensata (entro +/- 3 semitoni)
            for shift in [-3, -2, -1, 1, 2, 3]:
                transposed = smart_transpose(midi, shift)
                if transposed:
                    all_notes.extend(extract_notes(transposed))

        except Exception as e:
            print(f"⚠️ Errore parsing {file.name}: {e}")
            continue

    # Rimuovi outlier (eventi troppo rari)
    counts = {n: all_notes.count(n) for n in set(all_notes)}
    filtered = [n for n in all_notes if counts[n] >= 10]

    Path("data").mkdir(exist_ok=True)
    with open("data/notes", "wb") as f:
        pickle.dump(filtered, f)

    print(f"✅ {len(filtered)} eventi musicali salvati (filtrati)")
    return filtered


def prepare_sequences(notes, sequence_length=100):
    """Prepara le sequenze input/output"""
    pitchnames = sorted(set(notes))
    note_to_int = {note: i for i, note in enumerate(pitchnames)}
    n_vocab = len(pitchnames)

    input_seq, output_seq = [], []
    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        input_seq.append([note_to_int[n] for n in seq_in])
        output_seq.append(note_to_int[seq_out])

    X = np.reshape(input_seq, (len(input_seq), sequence_length, 1)) / float(n_vocab)
    y = to_categorical(output_seq, num_classes=n_vocab)
    return X, y, n_vocab


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


def train_model(model, X_train, y_train, X_val, y_val):
    """Training con early stopping"""
    checkpoint = ModelCheckpoint("weights.keras", monitor='val_loss', save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=128,
        callbacks=[checkpoint, earlystop],
        verbose=1
    )
    return history


def plot_training(history):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Val Loss')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('accuracy', []), label='Train Accuracy')
    plt.plot(history.history.get('val_accuracy', []), label='Val Accuracy')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_pipeline():
    notes = get_notes()
    X, y, n_vocab = prepare_sequences(notes)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    model = create_network(network_input=X, n_vocab=n_vocab)
    history = train_model(model, X_train, y_train, X_val, y_val)
    plot_training(history)


if __name__ == '__main__':
    train_pipeline()
