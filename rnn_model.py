from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

def build_rnn(vocab_size, max_len):
    model = Sequential([
        Embedding(vocab_size, 8, input_length=max_len-1),
        SimpleRNN(16),
        Dense(vocab_size, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
