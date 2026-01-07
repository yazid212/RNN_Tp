from data import sentences
from preprocessing import prepare_data
from rnn_model import build_rnn
from ff_models import build_ff, create_ff_data

print("Préparation des données...")
X, y, tokenizer, vocab_size, max_len, sequences = prepare_data(sentences)

print(f"Taille du vocabulaire: {vocab_size}")
print(f"Longueur maximale: {max_len}")

# Question 5: Entraînement du RNN
print("\n" + "="*50)
print("Entraînement du RNN...")
print("="*50)
model_rnn = build_rnn(vocab_size, max_len)
model_rnn.fit(X, y, epochs=200, verbose=0)
model_rnn.save("rnn.h5")
print("✓ RNN entraîné et sauvegardé (rnn.h5)")

# Entraînement FF-2 (Question 7)
print("\n" + "="*50)
print("Entraînement FF-2 (contexte: 2 mots)...")
print("="*50)
X2, y2 = create_ff_data(sequences, 2)
ff2 = build_ff(vocab_size, 2)
ff2.fit(X2, y2, epochs=200, verbose=0)
ff2.save("ff2.h5")
print("✓ FF-2 entraîné et sauvegardé (ff2.h5)")

# Entraînement FF-6 (Question 7)
print("\n" + "="*50)
print("Entraînement FF-6 (contexte: 6 mots)...")
print("="*50)
X6, y6 = create_ff_data(sequences, 6)
ff6 = build_ff(vocab_size, 6)
ff6.fit(X6, y6, epochs=200, verbose=0)
ff6.save("ff6.h5")
print("✓ FF-6 entraîné et sauvegardé (ff6.h5)")

print("\n" + "="*50)
print("Tous les modèles sont entraînés et sauvegardés!")
print("="*50)