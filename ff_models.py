import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

def create_ff_data(sequences, n):
    X, y = [], []
    for sequence in sequences:
        for i in range(n, len(sequence)):
            X.append(sequence[i-n:i])
            y.append(sequence[i])
    # Convert to numpy arrays with proper dtype
    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)
    return X, y

def build_ff(vocab_size, n):
    model = Sequential([
        Embedding(vocab_size, 10, input_length=n),
        Flatten(),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model