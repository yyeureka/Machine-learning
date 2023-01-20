import mlrose_hiive
import numpy as np

SEED = 21

# Initialize fitness function object
edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
fitness = mlrose_hiive.MaxKColor(edges)
# Initialize optimization problem object
problem = mlrose_hiive.DiscreteOpt(length=5, fitness_fn=fitness, maximize=True, max_val=5)  # TODO

# Randomized hill climbing
best_state, best_fitness, curve = mlrose_hiive.random_hill_climb(problem, max_attempts=10, max_iters=np.inf, restarts=0,
                                                                 init_state=None, curve=True, fevals=False, random_state=SEED,
                                                                 state_fitness_callback=None, callback_user_info=None)
print(curve)
# best_state, best_fitness, curve = mlrose_hiive.random_hill_climb(problem, max_attempts=10, max_iters=np.inf, restarts=10, curve=True, random_state=SEED)
# print(curve)

# Simulated annealing
# Define decay schedule
schedule = mlrose_hiive.ExpDecay()
best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(),
                                                                   max_attempts=10, max_iters=np.inf, init_state=None,
                                                                   curve=True, fevals=False, random_state=SEED,
                                                                   state_fitness_callback=None, callback_user_info=None)
print(curve)
# best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(problem, schedule=schedule, max_attempts=10, max_iters=10, curve=True, random_state=SEED)
# print(curve)

# Genetic algorithm
best_state, best_fitness, curve = mlrose_hiive.genetic_alg(problem, pop_size=200, pop_breed_percent=0.75,
                                                           elite_dreg_ratio=0.99, minimum_elites=0, minimum_dregs=0,
                                                           mutation_prob=0.1, max_attempts=10, max_iters=np.inf,
                                                           curve=True, fevals=False, random_state=SEED,
                                                           state_fitness_callback=None, callback_user_info=None,
                                                           hamming_factor=0.0, hamming_decay_factor=None)
print(curve)

# MIMIC
best_state, best_fitness, curve = mlrose_hiive.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10,
                                                     max_iters=np.inf, curve=True, fevals=False, random_state=SEED,
                                                     state_fitness_callback=None, callback_user_info=None, noise=0.0)
print(curve)
