# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


from sklearn.model_selection import train_test_split,learning_curve,cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree


BD = pd.read_csv('Données_final_Chemin.csv',index_col=0)

#Séparer les colonnes caractéristique des colonnes cibles
x = BD.drop(columns=['vote_average'])
y = BD['vote_average']

# Encodage des variables catégoriques avec un encodage ordinal (pas utilisé un One-Hot Encoding avec méthode dumming() car beaucoup trop de catégorie => beaucoup trop de variable rajouté (augmentation de la dimensionnalité))
for col in x.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])

#Création d'une BD d'entrainnement et une BD de test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

#Entrainnement du modèle
rf = RandomForestRegressor(n_estimators=500, random_state=42,max_features=11)
rf.fit(x_train, y_train)

#Évaluation du modèle
y_pred = rf.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
RMSE = mse**0.5
RMSE_relatif= (RMSE/10)*100


# print(f"Mean Squared Error: {mse}")
# print(f"R² Score: {r2}")
# print(f"RMSE: {RMSE}")
# print(f"RMSE relatif: {RMSE_relatif} %")



#Constater la corrélation entre une variable caractéristique et y (et voir similarité entre modèle et réalité)
x_grid = np.arange(min(BD['popularity']),max(BD['popularity']),0.01)
x_grid = x_grid.reshape(len(x_grid),1) 
  
# plt.scatter(x,y, color='blue') #plotting real points
# plt.plot(x_grid, rf.predic(x_grid),color='green')
# plt.show()

# #Crée un schéma d'arbre de décision du modèle
# tree_to_plot = rf.estimators_[0]
# plt.figure(figsize=(600, 100))
# plot_tree(tree_to_plot, feature_names=BD.columns.tolist(), filled=True, rounded=True, fontsize=10)
# plt.title("Decision Tree from Random Forest")
# plt.show()

# # #Courbe d'apprentissage

train_sizes = np.linspace(0.1, 1.0, 10)

# # Obtenir les scores pour la courbe d'apprentissage
train_sizes, train_scores, test_scores = learning_curve(
    rf, x, y, train_sizes=train_sizes, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)

# Avec MSE
# Moyennes et écarts-types des erreurs
train_mean = -np.mean(train_scores, axis=1)  # Transformer les erreurs en positives
train_std = np.std(train_scores, axis=1)
test_mean = -np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# # Visualisation de la courbe
# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Erreur d\'entraînement')
# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
# plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Erreur de validation')
# plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color='orange')
# plt.title("Courbe d'apprentissage")
# plt.xlabel("Taille des données d'entraînement")
# plt.ylabel("Erreur quadratique moyenne (MSE)")
# plt.legend(loc="best")
# plt.grid()
# plt.show()

# ## Avec RMSE relatif
# Moyennes et écarts-types des RMSE relatifs
train_rmse_relative = (np.sqrt(-train_scores.mean(axis=1)) / 10)*100
test_rmse_relative = (np.sqrt(-test_scores.mean(axis=1)) / 10)*100

# # Visualisation de la courbe
# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, train_rmse_relative, 'o-', color='blue', label='Erreur relative d\'entraînement')
# plt.plot(train_sizes, test_rmse_relative, 'o-', color='orange', label='Erreur relative de validation')
# plt.title("Courbe d'apprentissage avec RMSE relatif")
# plt.xlabel("Taille des données d'entraînement")
# plt.ylabel("RMSE relatif en %")
# plt.legend(loc="best")
# plt.grid()
# plt.show()
#------------------------------------------------------------------
#Part pour graph courbe d'erreur vérifier (dans doc au propre)
#Courbe d'erreur pour nb arbre
n_estimators_range = range(25, 500, 25)
train_errors = []
test_errors = []

for n in n_estimators_range:
    model = RandomForestRegressor(n_estimators=n, random_state=42)
    model.fit(x_train, y_train)
    
    # Erreur d'entraînement (RMSE)
    train_pred = model.predict(x_train)
    train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
    train_errors.append(train_rmse)
    
    # Erreur de validation (RMSE relatif)
    test_pred = model.predict(x_test)
    test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))
    test_errors.append(test_rmse)

# Visualisation
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_errors, label='Erreur d\'entraînement', color='blue')
plt.plot(n_estimators_range, test_errors, label='Erreur de validation', color='orange')
plt.title("Courbe d'erreur en fonction de n_estimators")
plt.xlabel("Nombre d'estimateurs (n_estimators)")
plt.ylabel("RMSE")
plt.legend()
plt.grid()
plt.show()

# ##Courbe d'erreur pour nb feature max
# n_feature_range = range(1, x_train.shape[1] + 1)
# train_errors = []
# test_errors = []

# for n in n_feature_range:
#     model = RandomForestRegressor(n_estimators=100, random_state=42,max_features = n)
#     model.fit(x_train, y_train)
    
#     # Erreur d'entraînement (RMSE)
#     train_pred = model.predict(x_train)
#     train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
#     train_errors.append(train_rmse)
    
#     # Erreur de validation (RMSE relatif)
#     test_pred = model.predict(x_test)
#     test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))
#     test_errors.append(test_rmse)

# # Visualisation
# plt.figure(figsize=(10, 6))
# plt.plot(n_feature_range, train_errors, label='Erreur d\'entraînement', color='blue')
# plt.plot(n_feature_range, test_errors, label='Erreur de validation', color='orange')
# plt.title("Courbe d'erreur en fonction de max_features")
# plt.xlabel("Nombre de feature pour chaque division (max_features)")
# plt.ylabel("RMSE")
# plt.legend()
# plt.grid()
# plt.show()