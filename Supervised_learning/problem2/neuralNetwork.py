from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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
#                'mlpc__hidden_layer_sizes': [(10,), (20,), (100,), (5, 2), (100, 30), (50, 50, 50), (50, 100, 50)],
#                'mlpc__activation': ['tanh', 'relu']}
# cv = GridSearchCV(pipeline, param_grid=param_test1, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

# param_test2 = {'mlpc__solver': ['adam'],
#                'mlpc__alpha': [0.00005, 0.0001, 0.0005],
#                'mlpc__hidden_layer_sizes': [(100, 30), (100, 100)],
#                'mlpc__activation': ['relu']}
# cv = GridSearchCV(pipeline, param_grid=param_test2, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

# Optimal parameters
nn = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(100, 30), solver='adam', max_iter=1200, random_state=SEED)
steps = [('scaler', StandardScaler()), ('mlpc', nn)]
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
print('Confusion Matrix\n', confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Decision Tree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['Blind Guess', 'ROC curve (area = {:.2f})'.format(roc_auc)], loc='lower right')
plt.title('Decision Tree ROC Curve')
plt.show()

# Learning curve
plt.title('Neural Network: Loss Curve')
plt.plot(nn.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
