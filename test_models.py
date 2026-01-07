from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data import sentences
from preprocessing import prepare_data

print("="*60)
print("TP5 : Comparaison RNN vs Feed-Forward")
print("="*60)

# Préparation des données (identique à l'entraînement)
X, y, tokenizer, vocab_size, max_len, sequences = prepare_data(sentences)

# Chargement des modèles
print("\nChargement des modèles...")
rnn = load_model("rnn.h5")
ff2 = load_model("ff2.h5")
ff6 = load_model("ff6.h5")
print("✓ Modèles chargés")

print("\n" + "="*60)
print("QUESTION 6: Test de prédiction du RNN")
print("="*60)

# Question 6: Test avec séquence partielle
test_sentence = ["le", "chien", "que", "le", "chat"]
print(f"\nSéquence test: {' '.join(test_sentence)} _____")

# Conversion en indices
test_seq = tokenizer.texts_to_sequences([" ".join(test_sentence)])
# Padding
test_seq = pad_sequences(test_seq, maxlen=max_len-1, padding="pre")
# Prédiction
prediction = rnn.predict(test_seq, verbose=0)
predicted_word = tokenizer.index_word[prediction.argmax()]
print(f"Mot prédit par le RNN: '{predicted_word}'")

print("\n" + "="*60)
print("QUESTION 7: Comparaison FF-2, FF-6 et RNN")
print("="*60)

# Phrase complète du PDF
phrase_complete = ["le", "chat", "que", "le", "chien", "a", "vu"]
print(f"\nPhrase: {' '.join(phrase_complete)} _____")
print("Mot attendu: 'mange'")

# Test FF-2 (2 derniers mots seulement)
context_2 = phrase_complete[-2:]  # ["a", "vu"]
x2 = tokenizer.texts_to_sequences([" ".join(context_2)])
x2 = pad_sequences(x2, maxlen=2, padding="pre")
pred_ff2 = tokenizer.index_word[ff2.predict(x2, verbose=0).argmax()]
print(f"\n1. FF-2 (contexte = 2 derniers mots):")
print(f"   Mots fournis: {' '.join(context_2)}")
print(f"   → Prédiction: '{pred_ff2}'")

# Test FF-6 (6 derniers mots)
context_6 = phrase_complete[-6:]  # ["chat", "que", "le", "chien", "a", "vu"]
x6 = tokenizer.texts_to_sequences([" ".join(context_6)])
x6 = pad_sequences(x6, maxlen=6, padding="pre")
pred_ff6 = tokenizer.index_word[ff6.predict(x6, verbose=0).argmax()]
print(f"\n2. FF-6 (contexte = 6 derniers mots):")
print(f"   Mots fournis: {' '.join(context_6)}")
print(f"   → Prédiction: '{pred_ff6}'")

# Test RNN (séquence complète)
xr = tokenizer.texts_to_sequences([" ".join(phrase_complete)])
xr = pad_sequences(xr, maxlen=max_len-1, padding="pre")
pred_rnn = tokenizer.index_word[rnn.predict(xr, verbose=0).argmax()]
print(f"\n3. RNN (séquence complète):")
print(f"   Mots fournis: {' '.join(phrase_complete)}")
print(f"   → Prédiction: '{pred_rnn}'")

print("\n" + "="*60)
print("TEST AVEC CONTEXTE PROGRESSIF (Question 7)")
print("="*60)

# Test avec contexte qui s'allonge progressivement
contextes = [
    ["a", "vu"],
    ["chien", "a", "vu"],
    ["le", "chien", "a", "vu"],
    ["que", "le", "chien", "a", "vu"],
    ["chat", "que", "le", "chien", "a", "vu"],
    ["le", "chat", "que", "le", "chien", "a", "vu"]
]

print("\nÉvolution des prédictions RNN avec contexte croissant:")
for i, contexte in enumerate(contextes, 1):
    x_test = tokenizer.texts_to_sequences([" ".join(contexte)])
    x_test = pad_sequences(x_test, maxlen=max_len-1, padding="pre")
    pred = tokenizer.index_word[rnn.predict(x_test, verbose=0).argmax()]
    print(f"{i}. Contexte ({len(contexte)} mots): {' '.join(contexte):30} → '{pred}'")

print("\n" + "="*60)
print("RÉCAPITULATIF ET ANALYSE")
print("="*60)
print(f"FF-2 (2 mots):  '{pred_ff2}'")
print(f"FF-6 (6 mots):  '{pred_ff6}'")
print(f"RNN (complet):  '{pred_rnn}'")

print("\n" + "="*60)
print("RÉPONSE À LA QUESTION 7:")
print("="*60)
print("""
Le RNN fournit la prédiction la plus cohérente lorsque le contexte s'allonge car:

1. QUANTITÉ D'INFORMATION EXPLOITÉE:
   - FF-2: Utilise seulement 2 mots de contexte
   - FF-6: Utilise 6 mots de contexte (fenêtre fixe)
   - RNN: Utilise TOUTE la séquence (longueur variable)

2. PRÉSENCE D'UNE MÉMOIRE:
   - FF-2/FF-6: Pas de mémoire, chaque prédiction est indépendante
   - RNN: Possède une mémoire interne (état caché) qui accumule l'information

3. CAPACITÉ DE GÉNÉRALISATION:
   - FF: Limité par la fenêtre de contexte fixe
   - RNN: Peut capturer les dépendances à long terme (ex: sujet "le chat" 
          qui reste pertinent même après plusieurs mots intermédiaires)

CONCLUSION: Le RNN est supérieur pour le traitement du langage car il
maintient une mémoire du contexte complet et peut ainsi prédire des mots
qui dépendent d'éléments éloignés dans la phrase.
""")