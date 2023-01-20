from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import mlrose_hiive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

SEED = 21

# Preprocess data
df = pd.read_csv('winequality-red.csv')
df.loc[df['quality'] <= 5, 'quality'] = 0
df.loc[df['quality'] > 5, 'quality'] = 1

X = df.drop('quality', axis=1)
y = df['quality']

# Randomly split training / test set
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


# Gradient descent
# Grid search
# nn_GD = mlrose_hiive.NeuralNetwork(algorithm='gradient_descent', random_state=SEED)
# steps = [('scaler', StandardScaler()), ('nn', nn_GD)]
# pipeline = Pipeline(steps)
# param_test1 = {'nn__learning_rate': [0.00001, 0.0001, 0.001, 0.01],
#                'nn__hidden_nodes': [(10,), (20,), (100,), (5, 2), (100, 30)]}
# cv = GridSearchCV(pipeline, param_grid=param_test1, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

t0 = time()
nn_GD = mlrose_hiive.NeuralNetwork(hidden_nodes=(10,), activation='relu', algorithm='gradient_descent', max_iters=10000,
                                   learning_rate=0.001, clip_max=1e+10, max_attempts=10, random_state=SEED, curve=True)
steps = [('scaler', StandardScaler()), ('nn', nn_GD)]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
t1 = time()
print(t1 - t0)
y_train_pred = pipeline.predict(X_train)
y_train_accuracy = accuracy_score(y_train, y_train_pred)
print(y_train_accuracy)
y_test_pred = pipeline.predict(X_test)
y_test_accuracy = accuracy_score(y_test, y_test_pred)
print(y_test_accuracy)
print(f1_score(y, pipeline.predict(X)))

# # Randomized hill climbing
# # Grid search
# # nn_RHC = mlrose_hiive.NeuralNetwork(hidden_nodes=(10,), algorithm='random_hill_climb', random_state=SEED)
# # steps = [('scaler', StandardScaler()), ('nn', nn_RHC)]
# # pipeline = Pipeline(steps)
# # param_test1 = {'nn__learning_rate': [0.001, 0.01, 0.1, 1, 3, 5, 10]}
# # cv = GridSearchCV(pipeline, param_grid=param_test1, cv=5, n_jobs=-1)
# # cv.fit(X_train, y_train)
# # print('Best parameters', cv.best_params_)
# # print('Best score', cv.best_score_)

t0 = time()
nn_RHC = mlrose_hiive.NeuralNetwork(hidden_nodes=(10,), activation='relu', algorithm='random_hill_climb', max_iters=10000,
                                    learning_rate=3, clip_max=1e+10, max_attempts=10, random_state=SEED, curve=True)
steps = [('scaler', StandardScaler()), ('nn', nn_RHC)]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
t1 = time()
print(t1 - t0)
y_train_pred = pipeline.predict(X_train)
y_train_accuracy = accuracy_score(y_train, y_train_pred)
print(y_train_accuracy)
y_test_pred = pipeline.predict(X_test)
y_test_accuracy = accuracy_score(y_test, y_test_pred)
print(y_test_accuracy)
print(f1_score(y, pipeline.predict(X)))

# # Simulated annealing
# # Grid search
# # nn_SA = mlrose_hiive.NeuralNetwork(hidden_nodes=(10,), algorithm='simulated_annealing', random_state=SEED)
# # steps = [('scaler', StandardScaler()), ('nn', nn_SA)]
# # pipeline = Pipeline(steps)
# # param_test1 = {'nn__learning_rate': [45.5, 46, 46.5],
# #                'nn__schedule': [mlrose_hiive.GeomDecay(), mlrose_hiive.ArithDecay(), mlrose_hiive.ExpDecay()]}
# # cv = GridSearchCV(pipeline, param_grid=param_test1, cv=5, n_jobs=-1)
# # cv.fit(X_train, y_train)
# # print('Best parameters', cv.best_params_)
# # print('Best score', cv.best_score_)

t0 = time()
nn_SA = mlrose_hiive.NeuralNetwork(hidden_nodes=(10,), activation='relu', algorithm='simulated_annealing', max_iters=10000,
                                   learning_rate=46, clip_max=1e+10, schedule=mlrose_hiive.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001),
                                   max_attempts=10, random_state=SEED, curve=True)
steps = [('scaler', StandardScaler()), ('nn', nn_SA)]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
t1 = time()
print(t1 - t0)
y_train_pred = pipeline.predict(X_train)
y_train_accuracy = accuracy_score(y_train, y_train_pred)
print(y_train_accuracy)
y_test_pred = pipeline.predict(X_test)
y_test_accuracy = accuracy_score(y_test, y_test_pred)
print(y_test_accuracy)
print(f1_score(y, pipeline.predict(X)))

# Genetic algorithm
# # Grid search
# nn_GA = mlrose_hiive.NeuralNetwork(hidden_nodes=(10,), algorithm='genetic_alg', random_state=SEED)
# steps = [('scaler', StandardScaler()), ('nn', nn_GA)]
# pipeline = Pipeline(steps)
# param_test1 = {'nn__learning_rate': [1, 5],
#                'nn__pop_size': [200],
#                'nn__mutation_prob': [0.1]}
# cv = GridSearchCV(pipeline, param_grid=param_test1, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)
# param_test2 = {'nn__learning_rate': [1],
#                'nn__pop_size': [100, 200, 300],
#                'nn__mutation_prob': [0.1]}
# cv = GridSearchCV(pipeline, param_grid=param_test2, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)
# param_test2 = {'nn__learning_rate': [1],
#                'nn__pop_size': [200],
#                'nn__mutation_prob': [0.1, 0.2, 0.3]}
# cv = GridSearchCV(pipeline, param_grid=param_test2, cv=5, n_jobs=-1)
# cv.fit(X_train, y_train)
# print('Best parameters', cv.best_params_)
# print('Best score', cv.best_score_)

t0 = time()
nn_GA = mlrose_hiive.NeuralNetwork(hidden_nodes=(10,), activation='relu', algorithm='genetic_alg', max_iters=10000,
                                   learning_rate=1, clip_max=1e+10, pop_size=200, mutation_prob=0.1, max_attempts=10,
                                   random_state=SEED, curve=True)
steps = [('scaler', StandardScaler()), ('nn', nn_GA)]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
t1 = time()
print(t1 - t0)
y_train_pred = pipeline.predict(X_train)
y_train_accuracy = accuracy_score(y_train, y_train_pred)
print(y_train_accuracy)
y_test_pred = pipeline.predict(X_test)
y_test_accuracy = accuracy_score(y_test, y_test_pred)
print(y_test_accuracy)
print(f1_score(y, pipeline.predict(X)))

# Convergence property
plt.plot(nn_GD.fitness_curve, label='GD')
plt.plot(nn_RHC.fitness_curve, label='RHC')
plt.plot(nn_SA.fitness_curve, label='SA')
plt.plot(nn_GA.fitness_curve, label='GA')
plt.legend()
plt.title('NN Convergence')
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.show()
