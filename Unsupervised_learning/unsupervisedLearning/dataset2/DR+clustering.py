from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
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

# PCA
pca = PCA(n_components=5, random_state=SEED)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)

# ICA
ica = FastICA(n_components=8, random_state=SEED)
ica.fit(X_train)
X_train_ica = ica.transform(X_train)

# RP
rp = GaussianRandomProjection(n_components=6, random_state=SEED)
rp.fit(X_train)
X_train_rp = rp.transform(X_train)

# RF
rf = RandomForestClassifier(oob_score=True, random_state=SEED)
sfm = SelectFromModel(rf, threshold=0.07)
sfm.fit(X_train, y_train)
X_train_rf = sfm.transform(X_train)

# K-means
inertias = []
sc_scores_km = []
time_km = []
inertias_pca = []
sc_scores_km_pca = []
time_km_pca = []
inertias_ica = []
sc_scores_km_ica = []
time_km_ica = []
inertias_rp = []
sc_scores_km_rp = []
time_km_rp = []
inertias_rf = []
sc_scores_km_rf = []
time_km_rf = []

k = range(2, 15)
for i in k:
    t0 = time()
    kmeans = KMeans(n_clusters=i, random_state=SEED)
    kmeans.fit(X_train)
    t1 = time()
    inertias.append(kmeans.inertia_)
    sc_scores_km.append(silhouette_score(X_train, kmeans.labels_, metric='euclidean'))
    time_km.append(t1 - t0)

    t0 = time()
    kmeans = KMeans(n_clusters=i, random_state=SEED)
    kmeans.fit(X_train_pca)
    t1 = time()
    inertias_pca.append(kmeans.inertia_)
    sc_scores_km_pca.append(silhouette_score(X_train_pca, kmeans.labels_, metric='euclidean'))
    time_km_pca.append(t1 - t0)

    t0 = time()
    kmeans = KMeans(n_clusters=i, random_state=SEED)
    kmeans.fit(X_train_ica)
    t1 = time()
    inertias_ica.append(kmeans.inertia_)
    sc_scores_km_ica.append(silhouette_score(X_train_ica, kmeans.labels_, metric='euclidean'))
    time_km_ica.append(t1 - t0)

    t0 = time()
    kmeans = KMeans(n_clusters=i, random_state=SEED)
    kmeans.fit(X_train_rp)
    t1 = time()
    inertias_rp.append(kmeans.inertia_)
    sc_scores_km_rp.append(silhouette_score(X_train_rp, kmeans.labels_, metric='euclidean'))
    time_km_rp.append(t1 - t0)

    t0 = time()
    kmeans = KMeans(n_clusters=i, random_state=SEED)
    kmeans.fit(X_train_rf)
    t1 = time()
    inertias_rf.append(kmeans.inertia_)
    sc_scores_km_rf.append(silhouette_score(X_train_rf, kmeans.labels_, metric='euclidean'))
    time_km_rf.append(t1 - t0)

# Plot Number of components vs Inertias
plt.plot(k, inertias, '-o', label='KM')
plt.plot(k, inertias_pca, '-o', label='KM+PCA')
plt.plot(k, inertias_ica, '-o', label='KM+ICA')
plt.plot(k, inertias_rp, '-o', label='KM+RP')
plt.plot(k, inertias_rf, '-o', label='KM+RF')
plt.xticks(k)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('K-means: Inertia')
plt.legend()
plt.show()

# Plot Number of components vs Silhouette Coefficient
plt.plot(k, sc_scores_km, '-o', label='KM')
plt.plot(k, sc_scores_km_pca, '-o', label='KM+PCA')
plt.plot(k, sc_scores_km_ica, '-o', label='KM+ICA')
plt.plot(k, sc_scores_km_rp, '-o', label='KM+RP')
plt.plot(k, sc_scores_km_rf, '-o', label='KM+RF')
plt.xticks(k)
plt.xlabel('Number of clusters')
plt.title('K-means: Silhouette Coefficient')
plt.legend()
plt.show()

# Plot Number of components vs Execution time
plt.plot(k, time_km, '-o', label='KM')
plt.plot(k, time_km_pca, '-o', label='KM+PCA')
plt.plot(k, time_km_ica, '-o', label='KM+ICA')
plt.plot(k, time_km_rp, '-o', label='KM+RP')
plt.plot(k, time_km_rf, '-o', label='KM+RF')
plt.xticks(k)
plt.xlabel('Number of clusters')
plt.title('K-means: Execution time')
plt.legend()
plt.show()

# EM
bic = []
sc_scores_em = []
time_em = []
bic_pca = []
sc_scores_em_pca = []
time_em_pca = []
bic_ica = []
sc_scores_em_ica = []
time_em_ica = []
bic_rp = []
sc_scores_em_rp = []
time_em_rp = []
bic_rf = []
sc_scores_em_rf = []
time_em_rf = []

n_components = range(2, 15)
for i in n_components:
    t0 = time()
    gmm = GaussianMixture(n_components=i, covariance_type='full', random_state=SEED)
    gmm.fit(X_train)
    t1 = time()
    bic.append(gmm.bic(X_train))
    sc_scores_em.append(silhouette_score(X_train, gmm.predict(X_train), metric='euclidean'))
    time_em.append(t1 - t0)

    t0 = time()
    gmm = GaussianMixture(n_components=i, covariance_type='full', random_state=SEED)
    gmm.fit(X_train_pca)
    t1 = time()
    bic_pca.append(gmm.bic(X_train_pca))
    sc_scores_em_pca.append(silhouette_score(X_train_pca, gmm.predict(X_train_pca), metric='euclidean'))
    time_em_pca.append(t1 - t0)

    t0 = time()
    gmm = GaussianMixture(n_components=i, covariance_type='full', random_state=SEED)
    gmm.fit(X_train_ica)
    t1 = time()
    bic_ica.append(gmm.bic(X_train_ica))
    sc_scores_em_ica.append(silhouette_score(X_train_ica, gmm.predict(X_train_ica), metric='euclidean'))
    time_em_ica.append(t1 - t0)

    t0 = time()
    gmm = GaussianMixture(n_components=i, covariance_type='full', random_state=SEED)
    gmm.fit(X_train_rp)
    t1 = time()
    bic_rp.append(gmm.bic(X_train_rp))
    sc_scores_em_rp.append(silhouette_score(X_train_rp, gmm.predict(X_train_rp), metric='euclidean'))
    time_em_rp.append(t1 - t0)

    t0 = time()
    gmm = GaussianMixture(n_components=i, covariance_type='full', random_state=SEED)
    gmm.fit(X_train_rf)
    t1 = time()
    bic_rf.append(gmm.bic(X_train_rf))
    sc_scores_em_rf.append(silhouette_score(X_train_rf, gmm.predict(X_train_rf), metric='euclidean'))
    time_em_rf.append(t1 - t0)

# Plot the BIC
plt.plot(n_components, bic, '-o', label='KM')
plt.plot(n_components, bic_pca, '-o', label='KM+PCA')
plt.plot(n_components, bic_ica, '-o', label='KM+ICA')
plt.plot(n_components, bic_rp, '-o', label='KM+RP')
plt.plot(n_components, bic_rf, '-o', label='KM+RF')
plt.xticks(n_components)
plt.xlabel('Number of components')
plt.title('EM: BIC score')
plt.legend()
plt.show()

# Plot Number of components vs Silhouette Coefficient
plt.plot(n_components, sc_scores_em, '-o', label='KM')
plt.plot(n_components, sc_scores_em_pca, '-o', label='KM+PCA')
plt.plot(n_components, sc_scores_em_ica, '-o', label='KM+ICA')
plt.plot(n_components, sc_scores_em_rp, '-o', label='KM+RP')
plt.plot(n_components, sc_scores_em_rf, '-o', label='KM+RF')
plt.xticks(n_components)
plt.xlabel('Number of components')
plt.title('EM: Silhouette Coefficient')
plt.legend()
plt.show()

# Plot Number of components vs Execution time
plt.plot(n_components, time_em, '-o', label='KM')
plt.plot(n_components, time_em_pca, '-o', label='KM+PCA')
plt.plot(n_components, time_em_ica, '-o', label='KM+ICA')
plt.plot(n_components, time_em_rp, '-o', label='KM+RP')
plt.plot(n_components, time_em_rf, '-o', label='KM+RF')
plt.xticks(n_components)
plt.xlabel('Number of components')
plt.title('EM: Execution time')
plt.legend()
plt.show()


