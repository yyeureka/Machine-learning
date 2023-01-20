from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

SEED = 21

# Preprocess data
df = pd.read_csv('winequality-red.csv')
# print(df['quality'].value_counts())
# print(df.head())

X = df.drop('quality', axis=1)
y = df['quality']
# Visualize data
# plt.scatter(X['alcohol'], X['sulphates'], marker='+', c=y)
# plt.xlabel('Alcohol (%)')
# plt.ylabel('Sulphates')
# plt.show()

# Randomly split training / test set
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)

# Default parameters
# Unscaled
nn = MLPClassifier(random_state=SEED)
nn.fit(X_train, y_train)
print('Training score:{:.3f}'.format(nn.score(X_train, y_train)))
print('Test score:{:.3f}'.format(nn.score(X_test, y_test)))
# Scaled
steps = [('scaler', StandardScaler()), ('mlpc', MLPClassifier(random_state=SEED))]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
print('Training score (scaled):{:.3f}'.format(pipeline.score(X_train, y_train)))
print('Test score (scaled):{:.3f}'.format(pipeline.score(X_test, y_test)))

# Grid search cross validation
steps = [('scaler', StandardScaler()), ('mlpc', MLPClassifier(random_state=SEED))]
pipeline = Pipeline(steps)

# param_test1 = {'mlpc__solver': ['lbfgs', 'adam'],
#                'mlpc__alpha': [0.00001, 0.0001, 0.001, 0.01],
#                'mlpc__hidden_layer_sizes': [(5, 2), (10,), (20,), (50, 50, 50), (50, 100, 50), (100,), (100, 30)],
#                'mlpc__activation': ['tanh', 'relu']}
# cv = GridSearchCV(pipeline, param_grid=param_test1, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

# param_test2 = {'mlpc__solver': ['adam'],
#                'mlpc__alpha': [0.005, 0.01, 0.05, 0.1],
#                'mlpc__hidden_layer_sizes': [(50, 100, 50)],
#                'mlpc__activation': ['relu']}
# cv = GridSearchCV(pipeline, param_grid=param_test2, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

# Optimal parameters
nn = MLPClassifier(activation='relu', alpha=0.05, hidden_layer_sizes=(50, 100, 50), solver='adam', max_iter=800, random_state=SEED)
steps = [('scaler', StandardScaler()), ('mlpc', nn)]
pipeline = Pipeline(steps)
t0 = time()
pipeline.fit(X_train, y_train)
t1 = time()
print('training time:{:.3f}'.format(t1 - t0))

# Evaluation
y_pred = pipeline.predict(X_test)
print('Tuned training score:{:.3f}'.format(pipeline.score(X_train, y_train)))
print('Tuned test score:{:.3f}'.format(pipeline.score(X_test, y_test)))
print('Confusion Matrix\n', confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# Learning curve
plt.title('Neural Network: Loss Curve')
plt.plot(nn.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()



