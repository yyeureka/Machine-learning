from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

SEED = 21

df = pd.read_csv('breastcancer.csv')
df.loc['M' == df['diagnosis'], 'diagnosis'] = 1
df.loc['B' == df['diagnosis'], 'diagnosis'] = 0
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']
y = y.astype('int')

# Preprocess data
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)

# Randomly split training/test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=SEED, stratify=y)

# PCA: Choose number of components
pca = PCA(random_state=SEED)
pca.fit_transform(X_train)
variance_ratio = pca.explained_variance_ratio_
cum_variance_ratio = np.cumsum(variance_ratio)
eigenvalues = pca.singular_values_
fig = plt.figure(figsize=(10, 6))
plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.1)
plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.05)
plt.xlabel('Number of components')
ax1 = fig.add_subplot(111)
ax1.plot(list(range(cum_variance_ratio.size)), cum_variance_ratio, '-o', label="Variance ratio")
ax1.set_ylabel('Cumulative Explained Variance Ratio')
plt.minorticks_on()
ax2 = ax1.twinx()
ax2.plot(list(range(eigenvalues.size)), eigenvalues, '-o', label="Eigenvalues")
ax2.set_ylabel('Eigenvalues')
plt.title('PCA: Eigenvalues')
plt.show()

# ICA: Choose number of components
components = list(np.arange(1, (X_train.shape[1] + 1), 1))
kurtosis = []
for i in components:
    ica = FastICA(n_components=i, random_state=SEED)
    ica_features = ica.fit_transform(X_train)
    tmpdf = pd.DataFrame(ica_features)
    tmpdf = tmpdf.kurt(axis=0)
    kurtosis.append(tmpdf.abs().mean())
plt.figure(figsize=(6, 6))
plt.plot(components, kurtosis, '-o')
plt.xlabel('Number of components')
plt.ylabel('Kurtosis')
plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.1)
plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.05)
plt.title("ICA: Kurtosis")
plt.show()

# RP: Choose number of components
components = list(np.arange(1, (X_train.shape[1]) + 1, 1))
recon_error = []
for i in components:
    rp = GaussianRandomProjection(n_components=i, random_state=SEED)
    rp_features = rp.fit_transform(X_train)
    recon = np.dot(rp_features, rp.components_)
    recon_error.append(np.mean((X_train - recon) ** 2))
plt.figure(figsize=(6, 6))
plt.plot(components, recon_error, '-o')
plt.xlabel('Number of components')
plt.ylabel('Reconstruction Error')
plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.1)
plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.05)
plt.title("RP: Reconstruction Error")
plt.show()

# RF: Choose number of features
rf = RandomForestClassifier(oob_score=True, random_state=SEED)
rf.fit(X_train, y_train)
print(rf.oob_score_)
print('Training score:', rf.score(X_train, y_train))
print('Test score:', rf.score(X_test, y_test))
y_pred_prob = rf.predict_proba(X_train)[:, 1]
print("AUC Score (Train):{:.3f}".format(roc_auc_score(y_train, y_pred_prob)))
y_pred_prob = rf.predict_proba(X_test)[:, 1]
print("AUC Score (Test):{:.3f}".format(roc_auc_score(y_test, y_pred_prob)))
# Plot feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), X.columns[indices], rotation=90)
plt.xlim([-1, len(importances)])
plt.tight_layout()
plt.title('RF: Feature Importances')
plt.show()

print("PCA")
t0 = time()
pca = PCA(n_components=6, random_state=SEED)
pca.fit(X_train)
pca_features = pca.transform(X_train)
t1 = time()
X_recon = pca.inverse_transform(pca_features)
recon_error = ((X_train - X_recon) ** 2).mean()
print(t1 - t0)
print(recon_error)
# plot correlation heap map
df = pd.DataFrame(pca_features)
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True)
plt.title("PCA")
plt.show()

print("ICA")
t0 = time()
ica = FastICA(n_components=29, random_state=SEED)
ica.fit(X_train)
ica_features = ica.transform(X_train)
t1 = time()
X_recon = ica.inverse_transform(ica_features)
recon_error = ((X_train - X_recon) ** 2).mean()
print(t1 - t0)
print(recon_error)
# plot correlation heap map
df = pd.DataFrame(ica_features)
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True)
plt.title("ICA")
plt.show()

print("RP")
t0 = time()
rp = GaussianRandomProjection(n_components=10, random_state=SEED)
rp.fit(X_train)
rp_features = rp.transform(X_train)
t1 = time()
X_recon = np.dot(rp_features, rp.components_)
recon_error = ((X_train - X_recon) ** 2).mean()
print(t1 - t0)
print(recon_error)
# plot correlation heap map
df = pd.DataFrame(rp_features)
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True)
plt.title("RP")
plt.show()

print("RF")
t0 = time()
rf = RandomForestClassifier(oob_score=True, random_state=SEED)
sfm = SelectFromModel(rf, threshold=0.01)
sfm.fit(X_train, y_train)
rf_features = sfm.transform(X_train)
t1 = time()
X_recon = sfm.inverse_transform(rf_features)
recon_error = ((X_train - X_recon) ** 2).mean()
print(t1 - t0)
print(recon_error)
