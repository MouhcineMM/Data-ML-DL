# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json

#Import de la Base de donnée
BD = pd.read_csv('tmdb_5000_movies.csv',header = 0, index_col=3)
BD = pd.DataFrame(BD)
BD_copy = BD.copy()

# #Controle doublon
# BD_Doublon = BD_copy[BD_copy["original_title"].duplicated()]
# Doublon = BD_Doublon[["original_title","release_date"]]
# #les doublons sont Batman et Out of the Blue
# Doublon = BD_copy[(BD_copy["original_title"]=="Batman") | (BD_copy["original_title"]=="Out of the Blue")]
# Doublon = Doublon[["original_title","release_date"]]

#Modif nom des doublons
BD_copy.at[268,"original_title"] ="Batman (1989)"
BD_copy.at[39269,"original_title"] ="Out of the Blue (1980)"
BD_copy.at[10844,"original_title"] ="Out of the Blue (2006)"
BD_copy.at[2661,"original_title"] ="Batman 1966 (1966)"

#Vérif plus de doublon
# BD_Doublon = BD_copy[BD_copy["original_title"].duplicated()]
# print(BD_Doublon)

#Variable "genres", "keywords", "product_compagnies" et "production contry" ont une structure similaire complexe (liste de dictionnaires)
#Nécessite de faire des modif car les données sont en JSON
BD_copy["genres"]=BD_copy["genres"].apply(json.loads)
BD_copy["keywords"]=BD_copy["keywords"].apply(json.loads)
BD_copy["production_companies"]=BD_copy["production_companies"].apply(json.loads)
BD_copy["production_countries"]=BD_copy["production_countries"].apply(json.loads)

BD_copy["genres"]=BD_copy["genres"].apply(lambda x:','.join(item["name"] for item in x))
BD_copy["keywords"]=BD_copy["keywords"].apply(lambda x:','.join(item["name"] for item in x))
BD_copy["production_companies"]=BD_copy["production_companies"].apply(lambda x:','.join(item["name"] for item in x))
BD_copy["production_countries"]=BD_copy["production_countries"].apply(lambda x:','.join(item["name"] for item in x))

#J'ai remarqué qu'il y avait une ligne avec un film ayant aucune autre données que son nom
BD_copy = BD_copy.drop(380097)

#Pour rendre les variable temporel plus exploitable, je crée la variable mois et années de diffusion
BD_copy['release_date'] = pd.to_datetime(BD_copy['release_date'])
BD_copy["year"] = BD_copy['release_date'].dt.year
BD_copy["month"] = BD_copy['release_date'].dt.month
BD_copy["year"] = BD_copy["year"].astype(int)
BD_copy["month"] = BD_copy["month"].astype(int)

#Données que je pense innutile ou peu pertinente (tagline, statut,homepage) et peut être overview si on arrive pas a trouver un moyen de la traiter
BD_copy= BD_copy.drop(["tagline", "status","homepage"],axis=1)

##Vérif que les données sont prête à l'emploi
#print(BD_copy.info())
#Seul la colonne overview comporte des données Null

#Analyse des variables avec un risque qu'il y est de 0
nb_0_budg = BD_copy[BD_copy["budget"]==0]["budget"].value_counts()
nb_0_pop =BD_copy[BD_copy["popularity"]==0]["popularity"].value_counts()
nb_0_revenu =BD_copy[BD_copy["revenue"]==0]["revenue"].value_counts()
nb_0_runtime =BD_copy[BD_copy["runtime"]==0]["runtime"].value_counts()
nb_0_vote =BD_copy[BD_copy["vote_count"]==0]["vote_count"].value_counts()

# print(nb_0_budg) #1036
# print(nb_0_revenu) #1426
# print(nb_0_runtime) #34
# print(nb_0_vote) #61
print(BD_copy.describe())
##Création fichiers csv pour données final
Données_final = BD_copy
chemin_doc='Données_final_Chemin.csv'
Données_final.to_csv(chemin_doc)