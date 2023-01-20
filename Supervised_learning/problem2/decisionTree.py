from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
dt = DecisionTreeClassifier(random_state=SEED)
dt.fit(X_train, y_train)
print('Training score:{:.3f}'.format(dt.score(X_train, y_train)))
print('Test score:{:.3f}'.format(dt.score(X_test, y_test)))

# Grid search cross validation
# param_grid = {'criterion': ['gini', 'entropy'],
#               'max_depth': np.arange(3, 20, 1),
#               'min_samples_split': np.arange(2, 10, 1),
#               'min_samples_leaf': np.arange(1, 10, 1)}
# cv = GridSearchCV(dt, param_grid, cv=5)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

# Optimal parameters
dt = DecisionTreeClassifier(criterion='gini', max_depth=16, min_samples_leaf=1, min_samples_split=3, random_state=SEED)
t0 = time()
dt = dt.fit(X_train, y_train)
t1 = time()
print('training time:{:.3f}'.format(t1 - t0))

# Evaluation
y_pred = dt.predict(X_test)
y_pred_prob = dt.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
print('Tuned training score:{:.3f}'.format(dt.score(X_train, y_train)))
print('Tuned test score:{:.3f}'.format(dt.score(X_test, y_test)))
print('Tuned test AUC score:{:.3f}'.format(roc_auc))
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

# Plot decision tree
# plt.figure()
# plot_tree(dt, filled=True)
# plt.show()

# Plot feature importances
# print(dt.feature_importances_)
# importances = dt.feature_importances_
# x = range(len(importances))
# labels = np.array(X.columns.values)
# plt.bar(x, importances, tick_label=labels)
# plt.xticks(rotation=30)  # label name is too long, rotate it by 90 degree
# plt.show()

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

    dt.fit(X1, y1)
    train_accuracy[i] = dt.score(X1, y1)
    test_accuracy[i] = dt.score(X_test, y_test)

plt.title('Decision Tree: Learning Curve')
plt.plot(training_size, test_accuracy, label='Test Accuracy')
plt.plot(training_size, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.show()
