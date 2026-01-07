Here’s the properly formatted **`README.md`** file code you asked for:

```markdown
# 📊 TP5: RNN vs Feed-Forward - Résultats et Analyse

## 🎯 Objectif
Comparer les performances des réseaux de neurones récurrents (RNN) et des réseaux feed-forward (FF) pour la prédiction de mots dans des phrases en français.

---

## 📁 Structure du Projet
```text
TP5_RNN_IA_NLP/
├── data.py              # Corpus de phrases
├── preprocessing.py     # Préparation des données
├── rnn_model.py         # Architecture RNN
├── ff_models.py         # Architectures FF-2 et FF-6
├── train_models.py      # Script d'entraînement
├── test_models.py       # Script de test et analyse
└── README.md            # Ce fichier
```

---

## 🚀 Comment exécuter

### Installation
```bash
pip install tensorflow numpy
```

### Entraînement
```bash
python train_models.py
```
➡️ Entraîne 3 modèles (RNN, FF-2, FF-6) pendant 200 epochs chacun.

### Test
```bash
python test_models.py
```
➡️ Génère l'analyse complète et les résultats.

---

## 📊 Résultats Obtenus

### ✅ Question 6: Test du RNN
**Séquence test :** `le chien que le chat _____`  
**Mot prédit par le RNN :** `effraie`  
**Analyse :** Correct ! Correspond à la phrase du corpus :  
`le chat que le chien effraie se cache`

---

### ✅ Question 7: Comparaison des 3 modèles
**Phrase test :** `le chat que le chien a vu _____`  
**Mot attendu :** `mange`

- **FF-2 (2 derniers mots) :** `a vu → mange`  
- **FF-6 (6 derniers mots) :** `chat que le chien a vu → mange`  
- **RNN (séquence complète) :** `le chat que le chien a vu → mange`

📌 **Résultat :** Les 3 modèles prédisent correctement `mange` !

---

## 🔍 Démonstration Clé: Avantage du RNN
Test avec contexte progressif :

| Contexte                  | Prédiction | Statut   |
|---------------------------|------------|----------|
| "a vu"                    | "le"       | ❌       |
| "chien a vu"              | "chien"    | ❌       |
| "le chien a vu"           | "chien"    | ❌       |
| "que le chien a vu"       | "voisin"   | ❌       |
| "chat que le chien a vu"  | "mange"    | ✅       |
| "séquence complète"       | "mange"    | ✅       |

---

## 📈 Analyse des Résultats

### Pourquoi FF-2 prédit correctement ?
- **Pattern local :** La séquence "a vu" est presque toujours suivie de "mange" dans ce corpus spécifique.  
- **Limitation :** Basé uniquement sur les 2 derniers mots, pas sur la structure globale de la phrase.

### Pourquoi le RNN est supérieur ?
- **Mémoire contextuelle :** Retient le sujet "le chat" même après 5 mots intermédiaires.  
- **Dépendances longues :** Relie un sujet distant (le chat) à son verbe (mange).  
- **Adaptabilité :** Fonctionne avec des contextes de longueur variable.  

---

## 🎯 Conclusion

| Modèle | Avantages            | Limitations                        |
|--------|----------------------|------------------------------------|
| FF-2   | Simple, rapide       | Contexte limité (2 mots)           |
| FF-6   | Contexte plus large  | Fenêtre fixe, pas de mémoire       |
| RNN    | Mémoire temporelle   | Plus complexe à entraîner          |

➡️ **Le RNN est le plus cohérent car il :**
- Exploite toute la séquence.  
- Maintient une mémoire du sujet.  
- Capture les relations sujet-verbe éloignées.  
- S’adapte aux séquences de longueur variable.  

---

## 📋 Spécifications Techniques
- **Modèles :** RNN (SimpleRNN 16 unités), FF-2, FF-6  
- **Données :** 6 phrases, vocabulaire de 18 mots  
- **Performance :** 100% d’accuracy sur le corpus d’entraînement  
```

Would you like me to also add **visual plots of training accuracy/loss curves** (in Markdown with placeholders for graphs), so the README looks more like a research report?
