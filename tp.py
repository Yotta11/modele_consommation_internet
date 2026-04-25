import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
df = pd.read_csv('data/processed/internet_consumption_complete.csv')

# Vue d'ensemble
print(df.shape)          # (81870, 9)
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())  # Vérifier les valeurs manquantes

#pretraitement

df['statut_connexion'] = df['statut_connexion'].str.replace('DÃ©connectÃ©','Déconnecté').str.replace('ConnectÃ©','Connecté')
df['periode_journee'] = df['periode_journee'].str.replace('Matin tÃ´t (05h-09h)','Matin tôt (05h-09h)')
df.dropna(inplace=True)
df = df[(df['download_Mo'] >= 0) & (df['upload_Mo'] >= 0) & (df['total_Mo'] >= 0)]

df['datetime'] = pd.to_datetime(df['datetime'])
df['heure'] = df['datetime'].dt.hour
df['jour_num'] = df['datetime'].dt.dayofweek
le = LabelEncoder()
df['statut_encode'] = le.fit_transform(df['statut_connexion'])
df['periode_encode'] = le.fit_transform(df['periode_journee'])
df['jour_encode'] = le.fit_transform(df['jour_semaine'])
 #caracteristiques pour la prediction
features = ['download_Mo','upload_Mo','heure','jour_num','periode_encode','jour_encode']
X_regr = df[features]
y_regr = df['total_Mo']        # Régression (prédire une valeur continue)
X_clf  = df[features]
y_clf  = df['statut_encode']   # Classification (prédire une catégorie)

# decoupage train/test (80/20) + normalisation
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X_regr, y_regr, test_size=0.2, random_state=42)
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_s  = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)
X_tr_cs = scaler.fit_transform(X_tr_c)
X_te_cs = scaler.transform(X_te_c)

# Exportation du dataset nettoyé vers un nouveau fichier CSV
df.to_csv('data/processed/internet_consumption_cleaned.csv', index=False, encoding='utf-8')
print(df.shape) 
print("Le dataset nettoyé a été sauvegardé avec succès !")

