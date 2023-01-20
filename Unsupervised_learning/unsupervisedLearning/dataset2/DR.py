from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

SEED = 21

df = pd.read_csv('winequality-red.csv')
X = df.drop('quality', axis=1)
y = df['quality']

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

# RF
# param_test1 = {'n_estimators': range(10, 101, 10)}
# rf = RandomForestClassifier(max_depth=5,
#                             min_samples_leaf=4,
#                             min_samples_split=5,
#                             max_features='sqrt',
#                             random_state=SEED)
# cv = GridSearchCV(rf, param_grid=param_test1, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

# param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(2, 21, 2)}
# rf = RandomForestClassifier(n_estimators=90,
#                             min_samples_leaf=1,
#                             max_features='sqrt',
#                             oob_score=True,
#                             random_state=SEED)
# cv = GridSearchCV(rf, param_grid=param_test2, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

# param_test3 = {'min_samples_split': range(2, 11, 2), 'min_samples_leaf':range(1, 12, 2)}
# rf = RandomForestClassifier(n_estimators=90,
#                             max_depth=11,
#                             max_features='sqrt',
#                             oob_score=True,
#                             random_state=SEED)
# cv = GridSearchCV(rf, param_grid=param_test3, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

# param_test4 = {'max_features': range(1, 12, 2)}
# rf = RandomForestClassifier(n_estimators=90,
#                             max_depth=11,
#                             min_samples_leaf=1,
#                             min_samples_split=4,
#                             oob_score=True,
#                             random_state=SEED)
# cv = GridSearchCV(rf, param_grid=param_test4, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

# RF: Choose number of features
rf = RandomForestClassifier(oob_score=True, random_state=SEED)
rf.fit(X_train, y_train)
print(rf.oob_score_)
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
pca = PCA(n_components=5, random_state=SEED)
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
ica = FastICA(n_components=8, random_state=SEED)
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
rp = GaussianRandomProjection(n_components=6, random_state=SEED)
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
sfm = SelectFromModel(rf, threshold=0.07)
sfm.fit(X_train, y_train)
rf_features = sfm.transform(X_train)
t1 = time()
X_recon = sfm.inverse_transform(rf_features)
recon_error = ((X_train - X_recon) ** 2).mean()
print(t1 - t0)
print(recon_error)



