# =============================================================================
#  TP — Régression & Classification  |  Dataset Consommation Internet
#  Code corrigé — 8 bugs résolus
# =============================================================================

# ── IMPORTS (tous en tête, sans doublons) ────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                          # FIX #8 : utilisé pour heatmap

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score  # FIX #6
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import (
    mean_squared_error, r2_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)


# ── 1. CHARGEMENT & ANALYSE INITIALE ─────────────────────────────────────────
df = pd.read_csv('data/processed/internet_consumption_complete.csv')

print("=== Vue d'ensemble ===")
print(f"Dimensions  : {df.shape}")
print(f"\nTypes :\n{df.dtypes}")
print(f"\nStatistiques descriptives :\n{df.describe()}")
print(f"\nValeurs manquantes :\n{df.isnull().sum()}")


# ── 2. PRÉTRAITEMENT ──────────────────────────────────────────────────────────

# 2a. Corriger l'encodage UTF-8 cassé
df['statut_connexion'] = (df['statut_connexion']
    .str.replace('DÃ©connectÃ©', 'Déconnecté')
    .str.replace('ConnectÃ©',    'Connecté'))
df['periode_journee'] = (df['periode_journee']
    .str.replace('Matin tÃ´t (05h-09h)', 'Matin tôt (05h-09h)'))

# 2b. Supprimer les lignes manquantes
df.dropna(inplace=True)

# 2c. Supprimer les valeurs négatives (aberrantes physiquement)
df = df[(df['download_Mo'] >= 0) & (df['upload_Mo'] >= 0) & (df['total_Mo'] >= 0)]

# 2d. Feature Engineering temporel
df['datetime'] = pd.to_datetime(df['datetime'])
df['heure']    = df['datetime'].dt.hour
df['jour_num'] = df['datetime'].dt.dayofweek

# 2e. Encodage des variables catégorielles
le = LabelEncoder()
df['statut_encode']  = le.fit_transform(df['statut_connexion'])
df['periode_encode'] = le.fit_transform(df['periode_journee'])
df['jour_encode']    = le.fit_transform(df['jour_semaine'])

# Sauvegarde du dataset nettoyé
df.to_csv('data/processed/internet_consumption_cleaned.csv', index=False, encoding='utf-8')
print(f"\nDataset nettoyé : {df.shape}")
print("Sauvegardé dans data/processed/internet_consumption_cleaned.csv")


# ── 3. DÉFINITION DES FEATURES ────────────────────────────────────────────────
#
# FIX #4 (LOGIQUE) : total_Mo = download_Mo + upload_Mo (corrélation 0.98)
# → Pour la RÉGRESSION, retirer download_Mo et upload_Mo des features
#   pour éviter que le modèle "triche" en apprenant la formule directe.
# → Pour la CLASSIFICATION, on peut les conserver (cible différente).
#
features_regr = ['heure', 'jour_num', 'periode_encode', 'jour_encode']
features_clf  = ['download_Mo', 'upload_Mo', 'heure', 'jour_num', 'periode_encode', 'jour_encode']

X_regr = df[features_regr]
y_regr = df['total_Mo']       # Cible régression (valeur continue)

X_clf  = df[features_clf]
y_clf  = df['statut_encode']    # Cible classification (0=Connecté, 1=Déconnecté)


# ── 4. DÉCOUPAGE TRAIN / TEST (80/20) ────────────────────────────────────────
X_tr,   X_te,   y_tr,   y_te   = train_test_split(X_regr, y_regr, test_size=0.2, random_state=42)
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_clf,  y_clf,  test_size=0.2, random_state=42)


# ── 5. NORMALISATION ─────────────────────────────────────────────────────────
#
# FIX #5 : Deux scalers séparés pour éviter qu'un fit() écrase l'autre.
# Règle : fit() uniquement sur le train, transform() sur train ET test.
#
scaler_r = StandardScaler()                        # Scaler régression
X_tr_s   = scaler_r.fit_transform(X_tr)
X_te_s   = scaler_r.transform(X_te)

scaler_c = StandardScaler()                        # Scaler classification
X_tr_cs  = scaler_c.fit_transform(X_tr_c)
X_te_cs  = scaler_c.transform(X_te_c)


# ══════════════════════════════════════════════════════════════════════════════
#  MODÈLES DE RÉGRESSION
# ══════════════════════════════════════════════════════════════════════════════

# ── 6a. Régression Linéaire ───────────────────────────────────────────────────
print("\n" + "="*50)
print("=== Régression Linéaire ===")
lr = LinearRegression()
lr.fit(X_tr_s, y_tr)
y_pred_lr = lr.predict(X_te_s)

print(f"R²   : {r2_score(y_te, y_pred_lr):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_te, y_pred_lr)):.4f}")
print(f"Coefficients : {dict(zip(features_regr, lr.coef_))}")


# ── 6b. KNN Régression ───────────────────────────────────────────────────────
print("\n=== KNN Régression ===")
scores_r = {}
for k in [3, 5, 7, 10, 15]:
    knn = KNeighborsRegressor(n_neighbors=k)
    score = cross_val_score(knn, X_tr_s, y_tr, cv=5, scoring='r2').mean()
    scores_r[k] = score
    print(f"  k={k:2d} → R² CV = {score:.4f}")

best_k_r = max(scores_r, key=scores_r.get)
print(f"→ Meilleur k (régression) : {best_k_r}")

knn_r = KNeighborsRegressor(n_neighbors=best_k_r)
knn_r.fit(X_tr_s, y_tr)
y_pred_knn_r = knn_r.predict(X_te_s)
print(f"R²   : {r2_score(y_te, y_pred_knn_r):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_te, y_pred_knn_r)):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODÈLES DE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

# ── 7a. Naïve Bayes (Gaussien) ────────────────────────────────────────────────
print("\n" + "="*50)
print("=== Naïve Bayes (Gaussien) ===")
nb = GaussianNB()
nb.fit(X_tr_cs, y_tr_c)
y_pred_nb = nb.predict(X_te_cs)

print(classification_report(y_te_c, y_pred_nb, target_names=['Connecté', 'Déconnecté']))
print("Matrice de confusion :\n", confusion_matrix(y_te_c, y_pred_nb))


# ── 7b. Régression Logistique ─────────────────────────────────────────────────
#
# FIX #2 & #3 (CRITIQUE) : Le bloc était un copier-coller de Naïve Bayes.
# La variable 'rl' n'était jamais créée → NameError dans le tableau comparatif.
#
print("\n=== Régression Logistique ===")
rl = LogisticRegression(max_iter=1000, random_state=42)
rl.fit(X_tr_cs, y_tr_c)
y_pred_rl = rl.predict(X_te_cs)

print(classification_report(y_te_c, y_pred_rl, target_names=['Connecté', 'Déconnecté']))
print("Matrice de confusion :\n", confusion_matrix(y_te_c, y_pred_rl))
print(f"Coefficients : {dict(zip(features_clf, rl.coef_[0]))}")


# ── 7c. KNN Classification ───────────────────────────────────────────────────
#
# FIX #7 (RISQUE) : best_k recalculé spécifiquement pour la classification
# avec scoring='f1' (plus adapté au déséquilibre 69/31) plutôt que 'r2'.
#
print("\n=== KNN Classification ===")
scores_c = {}
for k in [3, 5, 7, 10, 15]:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_tr_cs, y_tr_c, cv=5, scoring='f1').mean()
    scores_c[k] = score
    print(f"  k={k:2d} → F1 CV = {score:.4f}")

best_k_c = max(scores_c, key=scores_c.get)
print(f"→ Meilleur k (classification) : {best_k_c}")

knn_c = KNeighborsClassifier(n_neighbors=best_k_c)
knn_c.fit(X_tr_cs, y_tr_c)
y_pred_knn_c = knn_c.predict(X_te_cs)

print(classification_report(y_te_c, y_pred_knn_c, target_names=['Connecté', 'Déconnecté']))
print("Matrice de confusion :\n", confusion_matrix(y_te_c, y_pred_knn_c))


# ── 7d. SVM ──────────────────────────────────────────────────────────────────
print("\n=== SVM (kernel RBF) ===")
# Sous-échantillonnage pour accélérer (SVM O(n²) sur 65k lignes)
X_svm, y_svm = resample(X_tr_cs, y_tr_c, n_samples=10000, random_state=42)

svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm.fit(X_svm, y_svm)
y_pred_svm = svm.predict(X_te_cs)

print(classification_report(y_te_c, y_pred_svm, target_names=['Connecté', 'Déconnecté']))
print("Matrice de confusion :\n", confusion_matrix(y_te_c, y_pred_svm))


# ══════════════════════════════════════════════════════════════════════════════
#  ÉVALUATION & COMPARAISON
# ══════════════════════════════════════════════════════════════════════════════

# ── 8. Tableau comparatif — Classification ────────────────────────────────────
print("\n" + "="*50)
print("=== Tableau comparatif Classification ===")

modeles_clf = {
    'Naïve Bayes':     (nb,    X_te_cs),
    'Régression Log':  (rl,    X_te_cs),   # FIX #2 : rl maintenant défini
    'KNN':             (knn_c, X_te_cs),
    'SVM':             (svm,   X_te_cs),
}

print(f"{'Modèle':<20} {'Accuracy':>10} {'AUC-ROC':>10}")
print("-" * 45)
for nom, (mod, Xte) in modeles_clf.items():
    acc   = mod.score(Xte, y_te_c)
    proba = mod.predict_proba(Xte)[:, 1]
    auc_s = roc_auc_score(y_te_c, proba)
    print(f"{nom:<20} {acc:>10.4f} {auc_s:>10.4f}")


# ── 9. Courbes ROC superposées ────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
for nom, (mod, Xte) in modeles_clf.items():
    proba = mod.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(y_te_c, proba)
    plt.plot(fpr, tpr, label=f"{nom} (AUC={auc(fpr, tpr):.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aléatoire')
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Comparaison des courbes ROC — Classification')
plt.legend()
plt.tight_layout()
plt.savefig('figures/roc_comparison.png', dpi=150)
plt.show()


# ── 10. Tableau comparatif — Régression ──────────────────────────────────────
print("\n=== Tableau comparatif Régression ===")
print(f"{'Modèle':<25} {'R²':>8} {'RMSE':>12}")
print("-" * 50)
for nom, pred, real in [
    ('Régression Linéaire', y_pred_lr,    y_te),
    ('KNN Régression',      y_pred_knn_r, y_te),
]:
    r2_val = r2_score(real, pred)
    rmse   = np.sqrt(mean_squared_error(real, pred))
    print(f"{nom:<25} {r2_val:>8.4f} {rmse:>12.4f}")


# ── 11. Visualisation heatmap corrélations (FIX #8 : utiliser seaborn) ───────
fig, ax = plt.subplots(figsize=(6, 4))
corr = df[['download_Mo', 'upload_Mo', 'total_Mo']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', ax=ax,
            linewidths=0.5, vmin=0, vmax=1)
ax.set_title('Corrélations — features numériques')
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png', dpi=150)
plt.show()

print("\n✓ TP terminé avec succès !")
