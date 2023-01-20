import mlrose_hiive
import numpy as np
import matplotlib.pyplot as plt
from time import time

SEED = 21


# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):
    # Initialize counter
    fitness_cnt = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) and (state[j] != state[i] + (j - i))and (state[j] != state[i] - (j - i)):
                # If no attacks, then increment counter
                fitness_cnt += 1

    return fitness_cnt


# Initialize fitness function object
fitness = mlrose_hiive.CustomFitness(queens_max)
# Initialize optimization problem object
problem_size = 20
problem = mlrose_hiive.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True, max_val=problem_size)

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
#     print(best_fitness)
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
# pop_size = [200, 300, 400, 500]
# keep_pct = [0.2, 0.4, 0.6]
# for i in range(len(pop_size)):
#     for j in range(len(keep_pct)):
#         best_state, best_fitness, curve = mlrose_hiive.mimic(problem, pop_size=pop_size[i], keep_pct=keep_pct[j], max_attempts=10,
#                                                              max_iters=np.inf, curve=True, fevals=False, random_state=SEED,
#                                                              state_fitness_callback=None, callback_user_info=None, noise=0.0)
#         plt.plot(range(len(curve)), curve, label='P:{}, S:{}'.format(pop_size[i], keep_pct[j]))
# plt.legend()
# plt.title('MIMIC - Population & Samples')
# plt.xlabel('Iterations')
# plt.ylabel('Fitness')
# plt.show()

# Convergence property
# Randomized hill climbing
best_state, best_fitness, curve = mlrose_hiive.random_hill_climb(problem, curve=True, random_state=SEED)
plt.plot(range(len(curve)), curve, label='RHC')
# Simulated annealing
best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(), curve=True, random_state=SEED)
plt.plot(range(len(curve)), curve, label='SA')
# Genetic algorithm
best_state, best_fitness, curve = mlrose_hiive.genetic_alg(problem, pop_size=400, mutation_prob=0.3, curve=True, random_state=SEED)
plt.plot(range(len(curve)), curve, label='GA')
# MIMIC
best_state, best_fitness, curve = mlrose_hiive.mimic(problem, pop_size=300, keep_pct=0.2, curve=True, random_state=SEED)
plt.plot(range(len(curve)), curve, label='MIMIC')
plt.legend()
plt.title('Convergence')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.show()

# Problem complexity
problem_size = [20, 40, 60, 80, 100]
optimal_RHC = np.empty(len(problem_size))
time_RHC = np.empty(len(problem_size))
iters_RHC = np.empty(len(problem_size))
evals_RHC = np.empty(len(problem_size))
optimal_SA = np.empty(len(problem_size))
time_SA = np.empty(len(problem_size))
iters_SA = np.empty(len(problem_size))
evals_SA = np.empty(len(problem_size))
optimal_GA = np.empty(len(problem_size))
time_GA = np.empty(len(problem_size))
iters_GA = np.empty(len(problem_size))
evals_GA = np.empty(len(problem_size))
optimal_MIMIC = np.empty(len(problem_size))
time_MIMIC = np.empty(len(problem_size))
iters_MIMIC = np.empty(len(problem_size))
evals_MIMIC = np.empty(len(problem_size))

for i in range(len(problem_size)):
    # Initialize optimization problem object
    problem = mlrose_hiive.DiscreteOpt(length=problem_size[i], fitness_fn=fitness, maximize=True, max_val=problem_size[i])

    # Randomized hill climbing
    t0 = time()
    best_state, best_fitness, curve, evals = mlrose_hiive.random_hill_climb(problem, curve=True, fevals=True, random_state=SEED)
    t1 = time()

    optimal_RHC[i] = best_fitness
    time_RHC[i] = t1 - t0
    iters_RHC[i] = len(curve)
    evals_RHC[i] = sum(evals.values())

    # Simulated annealing
    t0 = time()
    schedule = mlrose_hiive.ExpDecay()
    best_state, best_fitness, curve, evals = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(),
                                                                              curve=True, fevals=True, random_state=SEED)
    t1 = time()

    optimal_SA[i] = best_fitness
    time_SA[i] = t1 - t0
    iters_SA[i] = len(curve)
    evals_SA[i] = sum(evals.values())

    # Genetic algorithm
    t0 = time()
    best_state, best_fitness, curve, evals = mlrose_hiive.genetic_alg(problem, pop_size=400, mutation_prob=0.3,
                                                                      curve=True, fevals=True, random_state=SEED)
    t1 = time()

    optimal_GA[i] = best_fitness
    time_GA[i] = t1 - t0
    iters_GA[i] = len(curve)
    evals_GA[i] = sum(evals.values())

    # MIMIC
    t0 = time()
    best_state, best_fitness, curve, evals = mlrose_hiive.mimic(problem, pop_size=300, keep_pct=0.2, curve=True,
                                                                fevals=True, random_state=SEED)
    t1 = time()

    optimal_MIMIC[i] = best_fitness
    time_MIMIC[i] = t1 - t0
    iters_MIMIC[i] = len(curve)
    evals_MIMIC[i] = sum(evals.values())

plt.figure()
plt.plot(problem_size, optimal_RHC, label='RHC')
plt.plot(problem_size, optimal_SA, label='SA')
plt.plot(problem_size, optimal_GA, label='GA')
plt.plot(problem_size, optimal_MIMIC, label='MIMIC')
plt.legend()
plt.title('Optimal fitness')
plt.xlabel('Problem sizes')
plt.ylabel('Optimal fitness')
plt.show()

plt.figure()
plt.plot(problem_size, time_RHC, label='RHC')
plt.plot(problem_size, time_SA, label='SA')
plt.plot(problem_size, time_GA, label='GA')
plt.plot(problem_size, time_MIMIC, label='MIMIC')
plt.legend()
plt.title('Time')
plt.xlabel('Problem sizes')
plt.ylabel('Time(s)')
plt.show()

plt.figure()
plt.plot(problem_size, iters_RHC, label='RHC')
plt.plot(problem_size, iters_SA, label='SA')
plt.plot(problem_size, iters_GA, label='GA')
plt.plot(problem_size, iters_MIMIC, label='MIMIC')
plt.legend()
plt.title('Iterations')
plt.xlabel('Problem sizes')
plt.ylabel('Iterations')
plt.show()

plt.figure()
plt.plot(problem_size, evals_RHC, label='RHC')
plt.plot(problem_size, evals_SA, label='SA')
plt.plot(problem_size, evals_GA, label='GA')
plt.plot(problem_size, evals_MIMIC, label='MIMIC')
plt.legend()
plt.title('Function calls')
plt.xlabel('Problem sizes')
plt.ylabel('Function calls')
plt.show()
