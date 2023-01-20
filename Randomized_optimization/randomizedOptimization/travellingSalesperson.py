import mlrose_hiive
import numpy as np

SEED = 21

# Create list of city coordinates
coords = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

# Initialize fitness function object using coords
fitness = mlrose_hiive.TravellingSales(coords=coords)
# Define optimization problem object
problem = mlrose_hiive.TSPOpt(length=8, fitness_fn=fitness, maximize=False)

# Solve problem using the genetic algorithm
best_state, best_fitness, curve = mlrose_hiive.genetic_alg(problem, max_attempts=10, mutation_prob=0.1, random_state=SEED)

print(best_state)
print(best_fitness)
