from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

SEED = 21

# Preprocess data
df = pd.read_csv('winequality-red.csv')
df.loc[df['quality'] <= 5, 'quality'] = 0
df.loc[df['quality'] > 5, 'quality'] = 1
# print(df['quality'].value_counts())
# print(df.head())

X = df.drop('quality', axis=1)
y = df['quality']
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
# Scaled
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
print('Training score (scaled):{:.3f}'.format(pipeline.score(X_train, y_train)))
print('Test score (scaled):{:.3f}'.format(pipeline.score(X_test, y_test)))

# Grid search cross validation
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
parameters = {'knn__n_neighbors': np.arange(1, 50)}
cv = GridSearchCV(pipeline, param_grid=parameters, cv=5)
cv.fit(X_train, y_train)
print('Best parameters', cv.best_params_)
print('Best score', cv.best_score_)

# Optimal parameters
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=12))]
pipeline = Pipeline(steps)
t0 = time()
pipeline.fit(X_train, y_train)
t1 = time()
print('training time:{:.3f}'.format(t1 - t0))

# Evaluation
y_pred = pipeline.predict(X_test)
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
print('Tuned training score:{:.3f}'.format(pipeline.score(X_train, y_train)))
print('Tuned test score:{:.3f}'.format(pipeline.score(X_test, y_test)))
print('Tuned test AUC score:{:.3f}'.format(roc_auc))
print('Confusion Matrix\n', confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='K-nearest Neighbors')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['Blind Guess', 'ROC curve (area = {:.2f})'.format(roc_auc)], loc='lower right')
plt.title('K-nearest Neighbors ROC Curve')
plt.show()

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
