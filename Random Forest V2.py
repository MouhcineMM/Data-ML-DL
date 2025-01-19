# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Charger les données
BD = pd.read_csv('Données_final_Chemin.csv', index_col=0)

# Identifier les colonnes catégoriques
categorical_columns = BD.select_dtypes(include=['object']).columns

# Encoder toutes les colonnes catégoriques avec LabelEncoder
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    BD[col] = le.fit_transform(BD[col].astype(str))
    label_encoders[col] = le  # Stocker l'encodeur pour d'éventuels décodages

# Supprimer les 4 premières lignes pour l'entraînement
BD_train = BD.iloc[4:]
BD_test_final = BD.iloc[:4]

# Séparer les colonnes caractéristiques des colonnes cibles
x_train_full = BD_train.drop(columns=['vote_average'])
y_train_full = BD_train['vote_average']
x_test_final = BD_test_final.drop(columns=['vote_average'])
y_test_final = BD_test_final['vote_average']

# Créer un jeu d'entraînement et de validation
x_train, x_test, y_train, y_test = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

# Entraînement du modèle
rf = RandomForestRegressor(n_estimators=500, random_state=42, max_features=11)
rf.fit(x_train, y_train)

# Prédictions sur le test final
predictions_final = rf.predict(x_test_final)

# Export des résultats en CSV
resultats = pd.DataFrame({
    'id': BD_test_final.index,
    'valeur_reelle': y_test_final,
    'valeur_predite': predictions_final
})

resultats.to_csv('resultats_test_final.csv', sep=';', index=False)

print("Les résultats du test final ont été sauvegardés dans 'resultats_test_final.csv'.")
