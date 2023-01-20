import mlrose_hiive
import numpy as np
import matplotlib.pyplot as plt
from time import time

SEED = 21

# Initialize fitness function object
problem_size = 20
weights = list(np.random.randint(low=1, high=100, size=problem_size))
values = list(np.random.randint(low=1, high=100, size=problem_size))
fitness = mlrose_hiive.Knapsack(weights, values)
# Initialize optimization problem object
problem = mlrose_hiive.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True)

# Hyperparameters tuning
# Simulated annealing
# schedule = [mlrose_hiive.GeomDecay(), mlrose_hiive.ArithDecay(), mlrose_hiive.ExpDecay()]
# label = ['Geom', 'Arith', 'Exp']
# for i in range(len(schedule)):
#     best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(problem, schedule=schedule[i],
#                                                                        max_attempts=10, max_iters=np.inf, init_state=None,
#                                                                        curve=True, fevals=False, random_state=SEED,
#                                                                        state_fitness_callback=None, callback_user_info=None)
#     plt.plot(range(len(curve)), curve, label='{}'.format(label[i]))
# plt.legend()
# plt.title('SA - Schedule')
# plt.xlabel('Iterations')
# plt.ylabel('Fitness')
# plt.show()
# Genetic algorithm
# pop_size = [100, 200, 300, 400]
# mutation_prob = [0.1, 0.3, 0.5]
# for i in range(len(pop_size)):
#     for j in range(len(mutation_prob)):
#         best_state, best_fitness, curve = mlrose_hiive.genetic_alg(problem, pop_size=pop_size[i], mutation_prob=mutation_prob[j],
#                                                                    curve=True, random_state=SEED)
#         plt.plot(range(len(curve)), curve, label='P:{}, M:{}'.format(pop_size[i], mutation_prob[j]))
# plt.legend()
# plt.title('GA - Population & Mutation')
# plt.xlabel('Iterations')
# plt.ylabel('Fitness')
# plt.show()
# MIMIC
pop_size = [100, 200, 300, 400]
keep_pct = [0.2, 0.4, 0.6]
for i in range(len(pop_size)):
    for j in range(len(keep_pct)):
        best_state, best_fitness, curve = mlrose_hiive.mimic(problem, pop_size=pop_size[i], keep_pct=keep_pct[j], max_attempts=10,
                                                             max_iters=np.inf, curve=True, fevals=False, random_state=SEED,
                                                             state_fitness_callback=None, callback_user_info=None, noise=0.0)
        plt.plot(range(len(curve)), curve, label='P:{}, S:{}'.format(pop_size[i], keep_pct[j]))
plt.legend()
plt.title('MIMIC - Population & Samples')
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.show()

# # Convergence property
# # Randomized hill climbing
# best_state, best_fitness, curve = mlrose_hiive.random_hill_climb(problem, curve=True, random_state=SEED)
# plt.plot(range(len(curve)), curve, label='RHC')
# # Simulated annealing
# best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(), curve=True, random_state=SEED)
# plt.plot(range(len(curve)), curve, label='SA')
# # Genetic algorithm
# best_state, best_fitness, curve = mlrose_hiive.genetic_alg(problem, pop_size=400, mutation_prob=0.5, curve=True, random_state=SEED)
# plt.plot(range(len(curve)), curve, label='GA')
# # MIMIC
# best_state, best_fitness, curve = mlrose_hiive.mimic(problem, pop_size=200, keep_pct=0.2, curve=True, random_state=SEED)
# plt.plot(range(len(curve)), curve, label='MIMIC')
# plt.legend()
# plt.title('Convergence')
# plt.xlabel('Iteration')
# plt.ylabel('Fitness')
# plt.show()

# problem_size = [20, 30, 40, 50]
# optimal_RHC = np.empty(len(problem_size))
# time_RHC = np.empty(len(problem_size))
# iterations_RHC = np.empty(len(problem_size))  # TODO
# optimal_SA = np.empty(len(problem_size))
# time_SA = np.empty(len(problem_size))
# iterations_SA = np.empty(len(problem_size))
# optimal_GA = np.empty(len(problem_size))
# time_GA = np.empty(len(problem_size))
# iterations_GA = np.empty(len(problem_size))
# optimal_MIMIC = np.empty(len(problem_size))
# time_MIMIC = np.empty(len(problem_size))
# iterations_MIMIC = np.empty(len(problem_size))
#
# for i in range(len(problem_size)):
#     # Initialize fitness function object
#     weights = list(np.random.randint(low=1, high=100, size=problem_size[i]))
#     values = list(np.random.randint(low=1, high=100, size=problem_size[i]))
#     fitness = mlrose_hiive.Knapsack(weights, values)  # TODO
#     # Initialize optimization problem object
#     problem = mlrose_hiive.DiscreteOpt(length=problem_size[i], fitness_fn=fitness, maximize=True)  # TODO
#
#     # Randomized hill climbing
#     t0 = time()
#     best_state, best_fitness, curve = mlrose_hiive.random_hill_climb(problem, max_attempts=10, max_iters=np.inf, restarts=0,
#                                                                      init_state=None, curve=True, fevals=False, random_state=SEED,
#                                                                      state_fitness_callback=None, callback_user_info=None)
#     t1 = time()
#
#     optimal_RHC[i] = best_fitness
#     time_RHC[i] = t1 - t0
#     iterations_RHC[i] = len(curve)
#
#     # Simulated annealing
#     t0 = time()
#     schedule = mlrose_hiive.ExpDecay()
#     best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(),
#                                                                        max_attempts=10, max_iters=np.inf, init_state=None,
#                                                                        curve=True, fevals=False, random_state=SEED,
#                                                                        state_fitness_callback=None, callback_user_info=None)
#     t1 = time()
#
#     optimal_SA[i] = best_fitness
#     time_SA[i] = t1 - t0
#     iterations_SA[i] = len(curve)
#
#     # Genetic algorithm
#     t0 = time()
#     best_state, best_fitness, curve = mlrose_hiive.genetic_alg(problem, pop_size=200, pop_breed_percent=0.75,
#                                                                elite_dreg_ratio=0.99, minimum_elites=0, minimum_dregs=0,
#                                                                mutation_prob=0.1, max_attempts=10, max_iters=np.inf,
#                                                                curve=True, fevals=False, random_state=SEED,
#                                                                state_fitness_callback=None, callback_user_info=None,
#                                                                hamming_factor=0.0, hamming_decay_factor=None)
#     t1 = time()
#
#     optimal_GA[i] = best_fitness
#     time_GA[i] = t1 - t0
#     iterations_GA[i] = len(curve)
#
#     # MIMIC
#     t0 = time()
#     best_state, best_fitness, curve = mlrose_hiive.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10,
#                                                          max_iters=np.inf, curve=True, fevals=False, random_state=SEED,
#                                                          state_fitness_callback=None, callback_user_info=None, noise=0.0)
#     t1 = time()
#
#     optimal_MIMIC[i] = best_fitness
#     time_MIMIC[i] = t1 - t0
#     iterations_MIMIC[i] = len(curve)
#
# plt.figure()
# plt.plot(problem_size, optimal_RHC, label='RHC')
# plt.plot(problem_size, optimal_SA, label='SA')
# plt.plot(problem_size, optimal_GA, label='GA')
# plt.plot(problem_size, optimal_MIMIC, label='MIMIC')
# plt.legend()
# plt.title('Optimal fitness')
# plt.xlabel('Problem size')
# plt.ylabel('Optimal fitness')
# plt.show()
#
# plt.figure()
# plt.plot(problem_size, time_RHC, label='RHC')
# plt.plot(problem_size, time_SA, label='SA')
# plt.plot(problem_size, time_GA, label='GA')
# plt.plot(problem_size, time_MIMIC, label='MIMIC')
# plt.legend()
# plt.title('Time')
# plt.xlabel('Problem size')
# plt.ylabel('Time(s)')
# plt.show()
#
# plt.figure()
# plt.plot(problem_size, iterations_RHC, label='RHC')
# plt.plot(problem_size, iterations_SA, label='SA')
# plt.plot(problem_size, iterations_GA, label='GA')
# plt.plot(problem_size, iterations_MIMIC, label='MIMIC')
# plt.legend()
# plt.title('Iterations')
# plt.xlabel('Problem size')
# plt.ylabel('Iterations')
# plt.show()
