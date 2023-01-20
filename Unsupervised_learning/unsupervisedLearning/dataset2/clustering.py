from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from time import time

SEED = 21
df = pd.read_csv('winequality-red.csv')
X = df.drop('quality', axis=1)
y = df['quality']

# # Visualize data
# plt.scatter(X['alcohol'], X['sulphates'], marker='+', c=y)
# plt.xlabel('Alcohol (%)')
# plt.ylabel('Sulphates')
# plt.title('Quality of red wine')
# plt.show()
# # Plot class distribution
# sns.countplot(x='quality', data=df);
# # Plot feature correlation
# plt.figure(figsize=(10, 10))
# sns.heatmap(df.corr(), annot=True)
# plt.tight_layout()
# plt.show()

# Preprocess data
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)

# Randomly split training/test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=SEED, stratify=y)

# K-means: Use Elbow method to choose number of clusters
k = range(2, 15)
inertias = []
sc_scores_km = []
time_km = []
for i in k:
    t0 = time()
    kmeans = KMeans(n_clusters=i, random_state=SEED)
    kmeans.fit(X_train)
    t1 = time()
    inertias.append(kmeans.inertia_)
    sc_scores_km.append(silhouette_score(X_train, kmeans.labels_, metric='euclidean'))
    time_km.append(t1 - t0)
# Plot ks vs inertias
plt.plot(k, inertias, '-o')
plt.xticks(k)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('K-means: Inertia')
plt.show()

# EM: Model selection with BIC
n_components = range(2, 15)
cv_types = ['spherical', 'tied', 'diag', 'full']
bic = []
sc_scores_em = []
time_em = []
for cv_type in cv_types:
    for i in n_components:
        t0 = time()
        gmm = GaussianMixture(n_components=i, covariance_type=cv_type, random_state=SEED)
        gmm.fit(X_train)
        t1 = time()
        bic.append(gmm.bic(X_train))
        if 'full' == cv_type:
            labels = gmm.predict(X_train)
            sc_scores_em.append(silhouette_score(X_train, labels, metric='euclidean'))
            time_em.append(t1 - t0)
# Plot the BIC
bars = []
bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components): (i + 1) * len(n_components)], width=.2, color=color))
plt.xticks(n_components)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('EM: BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components)) + .65 + .2 * np.floor(bic.argmin() / len(n_components))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
plt.xlabel('Number of components')
plt.legend([b[0] for b in bars], cv_types)
plt.show()

# Plot Number of components vs Silhouette Coefficient
plt.plot(k, sc_scores_km, '-o', label='KM')
plt.plot(n_components, sc_scores_em, '-o', label='EM')
plt.xticks(n_components)
plt.xlabel('Number of components')
plt.title('Silhouette Coefficient')
plt.legend()
plt.show()

# Plot Number of components vs Execution time
plt.plot(k, time_km, '-o', label='KM')
plt.plot(n_components, time_em, '-o', label='EM')
plt.xticks(n_components)
plt.xlabel('Number of components')
plt.title('Execution time')
plt.legend()
plt.show()

# Visualize clusters
# labels
plt.scatter(X_train[:, 10], X_train[:, 9], marker='+', c=y_train)
plt.xlabel('Alcohol (%)')
plt.ylabel('Sulphates')
plt.title('Original class labels')
plt.show()

# K-means
kmeans = KMeans(n_clusters=2, random_state=SEED)
labels = kmeans.fit_predict(X_train)
# Cross tabulation
df = pd.DataFrame({'labels': labels, 'quality': y_train})
ct = pd.crosstab(df['labels'], df[ 'quality'])
print(ct)
# Visualize clusters and cluster centers
centroids = kmeans.cluster_centers_
centroids_x = centroids[:, 10]
centroids_y = centroids[:, 9]
plt.scatter(X_train[:, 10], X_train[:, 9], marker='+', c=labels)
plt.scatter(centroids_x, centroids_y, marker='D', c='k', s=50)
plt.xlabel('Alcohol (%)')
plt.ylabel('Sulphates')
plt.title('K-means clustering')
plt.show()

# EM
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=SEED)
labels = gmm.fit_predict(X_train)
# Cross tabulation
df = pd.DataFrame({'labels': labels, 'quality': y_train})
ct = pd.crosstab(df['labels'], df['quality'])
print(ct)
# Visualize clusters
plt.scatter(X_train[:, 10], X_train[:, 9], marker='+', c=labels)
plt.xlabel('Alcohol (%)')
plt.ylabel('Sulphates')
plt.title('EM clustering')
plt.show()




