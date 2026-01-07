# 📊 TP5: RNN vs Feed-Forward - Résultats et Analyse

## 🎯 Objectif
Comparer les performances des réseaux de neurones récurrents (RNN) et des réseaux feed-forward (FF) pour la prédiction de mots dans des phrases en français.



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
🚀 Comment exécuterInstallationBashpip install tensorflow numpy
EntraînementBashpython train_models.py
Entraîne 3 modèles (RNN, FF-2, FF-6) pendant 200 epochs chacun.TestBashpython test_models.py
Génère l'analyse complète et les résultats.📊 Résultats Obtenus✅ Question 6: Test du RNNPlaintextSéquence test: le chien que le chat _____
Mot prédit par le RNN: 'effraie'
Analyse: Correct! Correspond à la phrase du corpus: "le chat que le chien effraie se cache"
✅ Question 7: Comparaison des 3 modèlesTest principal:Phrase: le chat que le chien a vu _____Mot attendu: 'mange'FF-2 (2 derniers mots): a vu → 'mange'FF-6 (6 derniers mots): chat que le chien a vu → 'mange'RNN (séquence complète): le chat que le chien a vu → 'mange'📌 Résultat: Les 3 modèles prédissent correctement 'mange' !🔍 Démontration Clé: Avantage du RNNTest avec contexte progressif :ContextePrédictionStatut"a vu"'le'❌ Incorrect"chien a vu"'chien'❌ Incorrect"le chien a vu"'chien'❌ Incorrect"que le chien a vu"'voisin'❌ Incorrect"chat que le chien a vu"'mange'✅ Correct"séquence complète"'mange'✅ Correct📈 Analyse des RésultatsPourquoi FF-2 prédit correctement ?Pattern local: La séquence "a vu" est presque toujours suivie de "mange" dans ce corpus spécifique.Limitation: Basé uniquement sur les 2 derniers mots, pas sur la structure globale de la phrase.Pourquoi le RNN est supérieur ?Mémoire contextuelle: Retient le sujet "le chat" même après 5 mots intermédiaires.Dépendances longues: Relie un sujet distant (le chat) à son verbe (mange).Adaptabilité: Fonctionne avec des contextes de longueur variable.🎯 ConclusionModèleAvantagesLimitationsFF-2Simple, rapideContexte limité (2 mots)FF-6Contexte plus largeFenêtre fixe, pas de mémoireRNNMémoire temporellePlus complexe à entraînerLe RNN est le plus cohérent car il :Exploite toute la séquence.Maintient une mémoire du sujet.Capture les relations sujet-verbe éloignées.S'adapte aux séquences de longueur variable.📋 Spécifications TechniquesModèles : RNN (SimpleRNN 16 unités), FF-2, FF-6.Données : 6 phrases, vocabulaire de 18 mots.Performance : 100% d'accuracy sur le corpus d'entraînement.
