from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
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
gb = GradientBoostingClassifier(random_state=SEED)
gb.fit(X_train, y_train)
print('Training score:{:.3f}'.format(gb.score(X_train, y_train)))
print('Test score:{:.3f}'.format(gb.score(X_test, y_test)))

# Grid search cross validation
# param_test1 = {'n_estimators': range(20, 81, 10)}
# gb = GradientBoostingClassifier(learning_rate=0.1,
#                                 subsample=0.8,
#                                 max_depth=5,
#                                 min_samples_leaf=4,
#                                 min_samples_split=5,
#                                 random_state=SEED)
# cv = GridSearchCV(gb, param_grid=param_test1, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

# param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(2, 20, 2)}
# gb = GradientBoostingClassifier(learning_rate=0.1,
#                                 n_estimators=60,
#                                 subsample=0.8,
#                                 min_samples_leaf=4,
#                                 random_state=SEED)
# cv = GridSearchCV(gb, param_grid=param_test2, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

# param_test3 = {'min_samples_split': range(18, 30, 2), 'min_samples_leaf': range(1, 10, 1)}
# gb = GradientBoostingClassifier(learning_rate=0.1,
#                                 n_estimators=60,
#                                 subsample=0.8,
#                                 max_depth=9,
#                                 random_state=SEED)
# cv = GridSearchCV(gb, param_grid=param_test3, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

# gb = GradientBoostingClassifier(learning_rate=0.1,
#                                 n_estimators=60,
#                                 subsample=0.8,
#                                 max_depth=9,
#                                 min_samples_leaf=4,
#                                 min_samples_split=18,
#                                 random_state=SEED)
# gb.fit(X_train, y_train)
# print('learning_rate = 0.1, n_estimators = 60:')
# print('Tuned training score:', gb.score(X_train, y_train))
# print('Tuned test score:', gb.score(X_test, y_test))
#
# gb = GradientBoostingClassifier(learning_rate=0.05,
#                                 n_estimators=120,
#                                 subsample=0.8,
#                                 max_depth=9,
#                                 min_samples_leaf=4,
#                                 min_samples_split=18,
#                                 random_state=SEED)
# gb.fit(X_train, y_train)
# print('learning_rate = 0.05, n_estimators = 120:')
# print('Tuned training score:', gb.score(X_train, y_train))
# print('Tuned test score:', gb.score(X_test, y_test))

# Optimal parameters
gb = GradientBoostingClassifier(learning_rate=0.01,
                                n_estimators=600,
                                subsample=0.8,
                                max_depth=9,
                                min_samples_leaf=4,
                                min_samples_split=18,
                                random_state=SEED)
t0 = time()
gb.fit(X_train, y_train)
t1 = time()
print('training time:{:.3f}'.format(t1 - t0))
print('learning_rate = 0.01, n_estimators = 600:')
print('Tuned training score:{:.3f}'.format(gb.score(X_train, y_train)))
print('Tuned test score:{:.3f}'.format(gb.score(X_test, y_test)))

# Evaluation
y_pred = gb.predict(X_test)
print('Confusion Matrix\n', confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# Learning curve
n_estimators = range(100, 601, 20)
train_accuracy = np.empty(len(n_estimators))
test_accuracy = np.empty(len(n_estimators))
for i in range(len(n_estimators)):
    gb = GradientBoostingClassifier(learning_rate=0.01,
                                    n_estimators=n_estimators[i],
                                    subsample=0.8,
                                    max_depth=9,
                                    min_samples_leaf=4,
                                    min_samples_split=18,
                                    random_state=SEED)

    gb.fit(X_train, y_train)
    train_accuracy[i] = gb.score(X_train, y_train)
    test_accuracy[i] = gb.score(X_test, y_test)

plt.title('Gradient Boosting: Learning Curve')
plt.plot(n_estimators, test_accuracy, label='Test Accuracy')
plt.plot(n_estimators, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Estimator number')
plt.ylabel('Accuracy')
plt.show()



