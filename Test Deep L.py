import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

# Charger les donnees
data = pd.read_csv('Données_final_Chemin.csv')

# Selection des colonnes pertinentes
selected_features = ['budget', 'popularity', 'runtime', 'vote_count', 'year', 'month', 'original_language']
target = 'vote_average'

data = data[selected_features + [target]].dropna()

# Separer 4 lignes pour le test final
final_test_data = data.sample(n=4, random_state=42)
remaining_data = data.drop(final_test_data.index)

# Separer les donnees restantes en X et y
X_remaining = remaining_data[selected_features]
y_remaining = remaining_data[target]

# Diviser les donnees restantes en entrainement et test
X_train, X_test, y_train, y_test = train_test_split(X_remaining, y_remaining, test_size=0.2, random_state=42)

# Preparer le pipeline de pretraitement
numerical_features = ['budget', 'popularity', 'runtime', 'vote_count', 'year', 'month']
categorical_features = ['original_language']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Transformer les donnees
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
X_final_test_transformed = preprocessor.transform(final_test_data[selected_features])

# Construire le modele de deep learning
model = Sequential([
    Dense(128, input_dim=X_train_transformed.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Sortie lineaire pour la regression
])

# Compiler le modele
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Entrainement du modele
model.fit(
    X_train_transformed,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# Predictions sur les ensembles de test
y_test_pred = model.predict(X_test_transformed)
final_predictions = model.predict(X_final_test_transformed)

# Calcul des metriques
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f"Metriques sur l'ensemble de test:\n")
print(f"- Mean Absolute Error (MAE): {mae}")
print(f"- Mean Squared Error (MSE): {mse}")
print(f"- Root Mean Squared Error (RMSE): {rmse}")
print(f"- R2 Score: {r2}")

# Creer un DataFrame pour les 4 lignes retirees
final_results = pd.DataFrame({
    'id': final_test_data.index,
    'valeurs_reelles': final_test_data[target].values,
    'valeurs_predites': final_predictions.flatten()
})

# Creer un DataFrame pour l'ensemble de test (train/test split)
test_results = pd.DataFrame({
    'id': X_test.index,
    'valeurs_reelles': y_test.values,
    'valeurs_predites': y_test_pred.flatten()
})

# Exporter les fichiers CSV avec separateur ; et sans accents
final_results.to_csv('final_results.csv', index=False, sep=';')
test_results.to_csv('test_results.csv', index=False, sep=';')

print("Exportation terminee :\n- final_results.csv : 4 lignes retirees\n- test_results.csv : ensemble de test")
