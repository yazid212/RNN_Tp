from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


def prepare_data(sentences):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)

    sequences = tokenizer.texts_to_sequences(sentences)
    vocab_size = len(tokenizer.word_index) + 1

    # Create input sequences for RNN
    input_sequences = []
    for sequence in sequences:
        for i in range(1, len(sequence)):
            n_gram_sequence = sequence[:i + 1]
            input_sequences.append(n_gram_sequence)

    # Pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

    # Create predictors and labels
    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]

    # Convert to proper dtype
    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    return X, y, tokenizer, vocab_size, max_sequence_len, sequences