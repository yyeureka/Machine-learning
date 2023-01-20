import mlrose_hiive
import numpy as np
import matplotlib.pyplot as plt
from time import time

SEED = 21

# Initialize fitness function object
fitness = mlrose_hiive.FlipFlop()
# Initialize optimization problem object
problem = mlrose_hiive.DiscreteOpt(length=10, fitness_fn=fitness, maximize=True, max_val=2)

# # Hyperparameters tuning
# # Simulated annealing
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
# # Genetic algorithm
# pop_size = [200, 300, 400, 500]
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
# # MIMIC
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
print(curve)
plt.plot(range(len(curve)), curve, label='RHC')
# Simulated annealing
best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(), curve=True, random_state=SEED)
plt.plot(range(len(curve)), curve, label='SA')
# Genetic algorithm
best_state, best_fitness, curve = mlrose_hiive.genetic_alg(problem, pop_size=200, mutation_prob=0.1, curve=True, random_state=SEED)
plt.plot(range(len(curve)), curve, label='GA')
# MIMIC
best_state, best_fitness, curve = mlrose_hiive.mimic(problem, pop_size=200, keep_pct=0.2, curve=True, random_state=SEED)
plt.plot(range(len(curve)), curve, label='MIMIC')
plt.legend()
plt.title('Convergence')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.show()
