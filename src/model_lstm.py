

# src/model_lstm.py

import tensorflow as tf

def build_lstm_model(sequence_length, vocab_size):
    """Costruisce il modello LSTM."""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 100, input_length=sequence_length),
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
