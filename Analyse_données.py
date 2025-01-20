# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib as mp
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

BD = pd.read_csv('C:\\Users\\conda\\OneDrive\\Bureau\\Master-1-Data-Science\\Math pour la Data Science\\Projet\\Données_final.csv',index_col=0)
import kagglehub

# Download latest version
BD = kagglehub.dataset_download("anandshaw2001/movie-rating-dataset")
liste_var=["revenue","budget","popularity","vote_average"]
# #En graph
# #voir top 10 budget
# #voir top 10 revenu
# # #voir top 10 note
# # #top 10 popularité
# #voir genre les plus utilisé
# for i in liste_var:
#     BD_top = BD.sort_values(by = i, ascending = False)
#     Top_10_truc = BD_top.head(10)
#     plt.figure(figsize=(20, 14))
#     plt.bar(Top_10_truc["original_title"], Top_10_truc[i], color='skyblue', edgecolor='black')
#     plt.xlabel('Nom du Film', fontsize=12)
#     plt.ylabel(i, fontsize=12)
#     plt.title('Top 10 Films '+i, fontsize=14)
#     plt.xticks(rotation=45, ha='right', fontsize=10)
#     plt.tight_layout()
#     plt.show()

# #le top des pires
# for i in liste_var:
#     BD_top = BD.sort_values(by = i, ascending = True)
#     Top_10_truc = BD_top.head(10)
#     plt.figure(figsize=(20, 14))
#     plt.bar(Top_10_truc["original_title"], Top_10_truc[i], color='skyblue', edgecolor='black')
#     plt.xlabel('Nom du Film', fontsize=12)
#     plt.ylabel(i, fontsize=12)
#     plt.title('Top 10 pire Films '+i, fontsize=14)
#     plt.xticks(rotation=45, ha='right', fontsize=10)
#     plt.tight_layout()
#     plt.show()


# #Faire une sous base de donnée pour les films avec vote average >8 et <2
# # faire un .describe() sur les deux sous BD
# Bd_null = BD[BD["vote_average"]<2]
# Bd_cool = BD[BD["vote_average"]>8]


# ##Faire graph des genres
# def count_genres(Base):
#     compteur= {}
#     Base_BD = Base["genres"]
#     for genres in Base_BD:
#         if isinstance(genres, str):
#             for genre in genres.split(','):
#                 genre = genre.strip()
#                 if genre in compteur:
#                     compteur[genre] += 1
#                 else :
#                     compteur[genre] =1
#     df_compteur = pd.DataFrame(list(compteur.items()), columns=["Genre", "Count"])
#     df_compteur = df_compteur.sort_values("Count", ascending = False)
#     return df_compteur

# def graph(df_compteur):
#     Top_10_truc = df_compteur.head(10)
#     plt.figure(figsize=(20, 14))
#     plt.bar(Top_10_truc["Genre"], Top_10_truc["Count"], color='skyblue', edgecolor='black')
#     plt.xlabel('Nom genre', fontsize=12)
#     plt.ylabel("Nb occurence", fontsize=12)
#     plt.title('Top 10 Genre', fontsize=14)
#     plt.xticks(rotation=45, ha='right', fontsize=10)
#     plt.tight_layout()
#     return plt.show()

# liste_BD = [BD,Bd_null,Bd_cool]
# for i in liste_BD:
#     graph(count_genres(i))
    

# #Serait bien de faire une matrice de corrélation
# x = BD.drop(columns=['vote_average'])
# x = x.copy()
# y = BD['vote_average']

# # Encodage des variables catégoriques avec un encodage ordinal 
# for col in x.select_dtypes(include=['object']).columns:
#     le = LabelEncoder()
#     x[col] = le.fit_transform(x[col])

# BD = x.join(y)
# corr = BD.corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Matrice de corrélation")
# plt.figure(figsize=(80,30))
# plt.show()