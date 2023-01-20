from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import pandas as pd
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

nn = MLPClassifier(max_iter=1000, random_state=SEED)

# Grid search cross validation
# param_test1 = {'solver': ['lbfgs', 'adam'],
#                'alpha': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
#                'hidden_layer_sizes': [(2,), (5,), (10,), (20,)],
#                'activation': ['tanh', 'relu']}
# cv = GridSearchCV(nn, param_grid=param_test1, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

# param_test2 = {'solver': ['adam'],
#                'alpha': [1e-11],
#                'hidden_layer_sizes': [(2,), (2, 2), (2, 2, 2)],
#                'activation': ['tanh']}
# cv = GridSearchCV(nn, param_grid=param_test2, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

print("NN")
t0 = time()
nn = MLPClassifier(activation='tanh', alpha=1e-11, hidden_layer_sizes=(2, 2, 2), solver='adam',
                   max_iter=1000, random_state=SEED)
nn.fit(X_train, y_train)
t1 = time()
print('Execution time:{:.3f}'.format(t1 - t0))
y_train_pred = nn.predict(X_train)
y_test_pred = nn.predict(X_test)
print('Accuracy (Train):{:.3f}'.format(accuracy_score(y_train, y_train_pred)))
print('Accuracy (Test):{:.3f}'.format(accuracy_score(y_test, y_test_pred)))
print("F1 Score (Train):{:.3f}".format(f1_score(y_train, y_train_pred)))
print("F1 Score (Test):{:.3f}".format(f1_score(y_test, y_test_pred)))
y_train_pred_prob = nn.predict_proba(X_train)[:, 1]
y_test_pred_prob = nn.predict_proba(X_test)[:, 1]
print("AUC Score (Train):{:.3f}".format(roc_auc_score(y_train, y_train_pred_prob)))
print("AUC Score (Test):{:.3f}".format(roc_auc_score(y_test, y_test_pred_prob)))

# PCA + NN
pca = PCA(n_components=6, random_state=SEED)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Grid search cross validation
# param_test1 = {'solver': ['lbfgs', 'adam'],
#                'alpha': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
#                'hidden_layer_sizes': [(2,), (5,), (10,), (20,)],
#                'activation': ['tanh', 'relu']}
# cv = GridSearchCV(nn, param_grid=param_test1, cv=5, n_jobs=-1)
# cv.fit(X_train_pca, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

# param_test2 = {'solver': ['lbfgs'],
#                'alpha': [1e-11],
#                'hidden_layer_sizes': [(2,), (2, 2), (2, 2, 2)],
#                'activation': ['relu']}
# cv = GridSearchCV(nn, param_grid=param_test2, cv=5, n_jobs=-1)
# cv.fit(X_train_pca, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

print("PCA + NN")
t0 = time()
nn = MLPClassifier(activation='relu', alpha=1e-11, hidden_layer_sizes=(2,), solver='lbfgs',
                   max_iter=1000, random_state=SEED)
nn.fit(X_train_pca, y_train)
t1 = time()
print('Execution time:{:.3f}'.format(t1 - t0))
y_train_pred = nn.predict(X_train_pca)
y_test_pred = nn.predict(X_test_pca)
print('Accuracy (Train):{:.3f}'.format(accuracy_score(y_train, y_train_pred)))
print('Accuracy (Test):{:.3f}'.format(accuracy_score(y_test, y_test_pred)))
print("F1 Score (Train):{:.3f}".format(f1_score(y_train, y_train_pred)))
print("F1 Score (Test):{:.3f}".format(f1_score(y_test, y_test_pred)))
y_train_pred_prob = nn.predict_proba(X_train_pca)[:, 1]
y_test_pred_prob = nn.predict_proba(X_test_pca)[:, 1]
print("AUC Score (Train):{:.3f}".format(roc_auc_score(y_train, y_train_pred_prob)))
print("AUC Score (Test):{:.3f}".format(roc_auc_score(y_test, y_test_pred_prob)))


# Clustering + NN
km = KMeans(n_clusters=2, random_state=SEED)
km.fit(X_train)
X_train_km = km.transform(X_train)
X_test_km = km.transform(X_test)

# Grid search cross validation
# param_test1 = {'solver': ['lbfgs', 'adam'],
#                'alpha': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
#                'hidden_layer_sizes': [(2,), (5,), (10,), (20,)],
#                'activation': ['tanh', 'relu']}
# cv = GridSearchCV(nn, param_grid=param_test1, cv=5, n_jobs=-1)
# cv.fit(X_train_km, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

# param_test2 = {'solver': ['lbfgs'],
#                'alpha': [1e-9],
#                'hidden_layer_sizes': [(10,), (10, 10), (10, 10, 10)],
#                'activation': ['tanh']}
# cv = GridSearchCV(nn, param_grid=param_test2, cv=5, n_jobs=-1)
# cv.fit(X_train_km, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

print("Clustering + NN")
t0 = time()
nn = MLPClassifier(activation='tanh', alpha=1e-9, hidden_layer_sizes=(10,), solver='lbfgs',
                   max_iter=1000, random_state=SEED)
nn.fit(X_train_km, y_train)
t1 = time()
print('Execution time:{:.3f}'.format(t1 - t0))
y_train_pred = nn.predict(X_train_km)
y_test_pred = nn.predict(X_test_km)
print('Accuracy (Train):{:.3f}'.format(accuracy_score(y_train, y_train_pred)))
print('Accuracy (Test):{:.3f}'.format(accuracy_score(y_test, y_test_pred)))
print("F1 Score (Train):{:.3f}".format(f1_score(y_train, y_train_pred)))
print("F1 Score (Test):{:.3f}".format(f1_score(y_test, y_test_pred)))
y_train_pred_prob = nn.predict_proba(X_train_km)[:, 1]
y_test_pred_prob = nn.predict_proba(X_test_km)[:, 1]
print("AUC Score (Train):{:.3f}".format(roc_auc_score(y_train, y_train_pred_prob)))
print("AUC Score (Test):{:.3f}".format(roc_auc_score(y_test, y_test_pred_prob)))
