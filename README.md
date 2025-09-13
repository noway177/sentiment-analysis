# Sentiment Analysis Project

Ce projet implémente un modèle d'analyse de sentiment utilisant PyTorch. Il inclut des scripts pour l'entraînement, l'évaluation et le test du modèle.

## Structure du projet

## Installation
1. Cloner ce dépôt
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation
  ```bash
  python train.py
  ```
  ```bash
  python eval.py
  ```
  ```bash
  python testfinal.py
  ```



## 📁 Structure du projet
- `data.py` : Préparation et gestion des données (chargement, nettoyage, vectorisation)
- `model.py` : Définition du modèle de sentiment (réseau de neurones PyTorch)
- `train.py` : Script d'entraînement du modèle (boucle d'entraînement, sauvegarde des poids)
- `eval.py` : Script d'évaluation du modèle (calcul des métriques, validation croisée)
- `testfinal.py` : Script de test final (prédiction sur de nouvelles données)
- `sentiment_model.pth` : Poids du modèle entraîné

## 🚀 Installation
1. Cloner ce dépôt :
   ```bash
   git clone <lien-du-repo>
   cd feeling analysis
   ```
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Utilisation
- **Entraîner le modèle** :
  ```bash
  python train.py
  ```
- **Évaluer le modèle** :
  ```bash
  python eval.py
  ```
- **Tester le modèle** :
  ```bash
  python testfinal.py
  ```

## 📊 Exemple de sortie
```
Texte : "J'adore ce produit !"
Prédiction : Positif 😊
```

## 📌 Détails techniques
- Framework : PyTorch
- Modèle : Réseau de neurones simple pour la classification binaire (positif/négatif)
- Données : Prétraitement, tokenisation, vectorisation
- Entraînement : Optimiseur Adam, fonction de perte BCE
- Évaluation : Accuracy, F1-score, matrice de confusion
