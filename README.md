# IREN — Prédiction de la mortalité chez des patients atteints d’insuffisance cardiaque

Projet d’apprentissage du **Deep Learning avec PyTorch**, réalisé en binôme avec Aymen Gassem dans le cadre d’un travail académique d’introduction aux réseaux de neurones.

## 🎯 Objectif

Étudier la capacité d’un réseau de neurones à effectuer une **classification binaire** à partir de données cliniques de patients atteints d’insuffisance cardiaque, afin de prédire la présence ou l’absence d’un événement de décès dans les données.

L’objectif est avant tout pédagogique : découvrir la construction, l’entraînement et l’évaluation d’un réseau de neurones, puis comparer différentes stratégies d’optimisation.

## 📊 Jeu de données

Le projet utilise le jeu de données [`mstz/heart_failure`](https://huggingface.co/datasets/mstz/heart_failure), accessible sur Hugging Face.

- **298 observations** de patients.
- Des variables cliniques et démographiques.
- Une variable cible binaire correspondant au décès (`is_dead`).
- Une séparation des données en un ensemble d’entraînement (80 %) et un ensemble de validation (20 %).

### Préparation des données

Les principales étapes de préparation sont :
- Chargement des données avec Hugging Face et conversion en DataFrame Pandas.
- Vérification des valeurs manquantes.
- Conversion des variables booléennes en valeurs numériques.
- Analyse des corrélations entre les variables.
- Retrait de la variable `days_in_study`, dont la corrélation avec la cible est notamment liée au suivi des patients.
- Normalisation des caractéristiques avant l’entraînement.

## 🧠 Modèle de Deep Learning

Le modèle est développé avec **PyTorch** et repose sur un réseau de neurones entièrement connecté composé de quatre couches linéaires :

`Nombre de caractéristiques → 64 → 32 → 16 → 1`

Les fonctions d’activation ReLU sont utilisées entre les couches cachées. Le projet explore ensuite différentes configurations d’entraînement pour étudier leur influence sur les performances.

## ⚙️ Expérimentations

Plusieurs configurations sont testées afin de comparer les résultats :

**Optimiseurs**
- SGD
- RMSprop
- AdaBound
- Adagrad
- Adam
- AdamW

**Hyperparamètres**
- Taux d’apprentissage (`learning rate`)
- `Weight decay`
- Taille des lots (`batch size`)
- Entraînement sur 20 époques

Le projet inclut également une expérimentation d’**ensemble learning**, qui combine les prédictions de plusieurs modèles pour étudier l’effet de leur agrégation.

## 📈 Résultats observés

Sur une exécution du notebook, le meilleur modèle individuel sélectionné selon l’accuracy et les pertes obtenues présente les paramètres suivants :

| Paramètre | Valeur |
|---|---|
| Optimiseur | RMSprop |
| Learning rate | 0,01 |
| Weight decay | 0,0001 |
| Batch size | 16 |
| Accuracy de validation observée | 73,33 % |

L’expérimentation d’ensemble learning obtient également des résultats variables selon la règle d’agrégation utilisée. Une règle consistant à prédire un décès dès qu’au moins un modèle le prédit atteint **71,67 % d’accuracy** sur cette exécution.

Ces résultats sont des observations sur un seul découpage des données. Ils ne démontrent pas une capacité de généralisation et doivent être interprétés avec prudence, compte tenu de la petite taille du jeu de données.

## 🛠️ Technologies utilisées

- Python
- Jupyter Notebook
- PyTorch
- Pandas et NumPy
- Scikit-learn
- Hugging Face Datasets
- TensorBoard

## 🚀 Lancer le projet

1. Cloner le dépôt GitHub.
2. Installer les dépendances nécessaires dans un environnement Python.
3. Ouvrir le notebook `IREN.ipynb` avec Jupyter Notebook ou JupyterLab.
4. Exécuter les cellules dans l’ordre.

Exemple d’installation des principales dépendances :

```bash
pip install jupyter pandas numpy torch scikit-learn datasets tensorboard adabound
```


## 👥 Auteurs

- Sandro Ferroni
- Aymen Gassem

Projet académique — Introduction aux réseaux de neurones, mai 2024.
