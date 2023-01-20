from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
# dataset information
# print(df['quality'].value_counts())
# print('Data size: ', df.shape)

X = df.drop('quality', axis=1)
y = df['quality']
print(X.columns.values)
# Visualize data
# plt.scatter(X['alcohol'], X['sulphates'], marker='+', c=y)
# plt.xlabel('Alcohol (%)')
# plt.ylabel('Sulphates')
# plt.title('Quality of red wine')
# plt.show()

# Randomly split training / test set
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)

# Default parameters
# Unscaled
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Training score:{:.3f}'.format(knn.score(X_train, y_train)))
print('Test score:{:.3f}'.format(knn.score(X_test, y_test)))
# Scaled
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
print('Training score (scaled):{:.3f}'.format(pipeline.score(X_train, y_train)))
print('Test score (scaled):{:.3f}'.format(pipeline.score(X_test, y_test)))

# Grid search cross validation
# steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
# pipeline = Pipeline(steps)
# parameters = {'knn__n_neighbors': np.arange(1, 50)}
# cv = GridSearchCV(pipeline, param_grid=parameters, cv=5)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

# Optimal parameters
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=40))]
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
training_size = [0.2, 0.4, 0.6, 0.8, 1]
train_accuracy = np.empty(len(training_size))
test_accuracy = np.empty(len(training_size))
for i in range(len(training_size)):
    if 1 == training_size[i]:
        X1, y1 = X_train, y_train
    else:
        X1, X2, y1, y2 = \
            train_test_split(X, y, test_size=(1 - training_size[i]), random_state=SEED, stratify=y)

    pipeline.fit(X1, y1)
    train_accuracy[i] = pipeline.score(X1, y1)
    test_accuracy[i] = pipeline.score(X_test, y_test)

plt.title('k-NN: Learning Curve')
plt.plot(training_size, test_accuracy, label='Test Accuracy')
plt.plot(training_size, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.show()





