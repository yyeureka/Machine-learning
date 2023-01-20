import mlrose_hiive
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

fitness = np.empty(256)
for i in range(256):
    state = np.zeros(8)
    num = [int(x) for x in list('{0:0b}'.format(i))]
    for j in range(8 - len(num), 8):
        state[j] = num[j - (8 - len(num))]
    fitness[i] = mlrose_hiive.OneMax().evaluate(state)
plt.plot(fitness)
plt.title('One Max')
plt.ylabel('Fitness')
plt.show()

fitness = np.empty(256)
for i in range(256):
    state = np.zeros(8)
    num = [int(x) for x in list('{0:0b}'.format(i))]
    for j in range(8 - len(num), 8):
        state[j] = num[j - (8 - len(num))]
    fitness[i] = mlrose_hiive.FourPeaks().evaluate(state)
plt.plot(fitness)
plt.title('Four Peaks')
plt.ylabel('Fitness')
plt.show()


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


state = list(permutations(range(0, 5)))
fitness = np.empty(120)
for i in range(len(state)):
    fitness[i] = queens_max(state[i])
plt.plot(fitness)
plt.title('N Queens')
plt.ylabel('Fitness')
plt.show()






