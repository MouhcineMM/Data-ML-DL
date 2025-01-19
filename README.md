# Projet de Prédiction de Films : Modèles ETL, Random Forest et Deep Learning

## Description
Ce projet vise à analyser et prédire les caractéristiques des films en utilisant des techniques de machine learning et de deep learning. À partir de données cinématographiques enrichies, le projet couvre plusieurs étapes :
1. Préparation des données (ETL).
2. Modélisation avec des approches d'apprentissage supervisé (Random Forest et Deep Learning).
3. Évaluation des performances.
4. Documentation approfondie avec un rapport rédigé en **LaTeX**.

---

## Structure des fichiers
- **`ETL_projet_math.py`** : Script d'extraction, transformation et chargement (ETL) pour préparer les données depuis le fichier source `tmdb_5000_movies.csv`. Les données sont nettoyées, enrichies et exportées en `Données_final_Chemin.csv`.
- **`Random Forest V2.py`** : Implémentation d'un modèle Random Forest pour prédire la note moyenne des films (`vote_average`). Les résultats finaux sont exportés dans `resultats_test_final.csv`.
- **`Test Deep L.py`** : Utilisation d'un modèle de deep learning basé sur TensorFlow pour affiner les prédictions des caractéristiques des films. Les résultats finaux sont exportés dans `final_results.csv` et `test_results.csv`.
- **Données CSV** :
  - `tmdb_5000_movies.csv` : Données brutes des films.
  - `Données_final_Chemin.csv` : Données nettoyées et prêtes à être utilisées par les modèles.
  - `resultats_test_final.csv`, `final_results.csv`, `test_results.csv` : Résultats des prédictions.
- **Rapport** : Documentation complète du projet, créée avec **LaTeX**, pour une présentation professionnelle et structurée.

---

## Prérequis
### Bibliothèques nécessaires
Les scripts utilisent les bibliothèques suivantes :
- Python 3.8 ou supérieur
- Pandas
- NumPy
- scikit-learn
- TensorFlow (Keras)
- json

Installez-les avec pip :
```bash
pip install pandas numpy scikit-learn tensorflow
