from hiive.mdptoolbox import mdp
from gym.envs.toy_text.frozen_lake import generate_random_map
from openai import converter
import numpy as np
import matplotlib.pyplot as plt

SEED = 0

# 8*8
np.random.seed(SEED)
random_map = generate_random_map(size=8)
print(np.reshape(random_map, (8, 1)))
T, R = converter("FrozenLake-v0", desc=random_map)

gamma = [0.1, 0.4, 0.7, 0.99]
value_VI = np.empty(len(gamma))
iter_VI = np.empty(len(gamma))
time_VI = np.empty(len(gamma))
value_PI = np.empty(len(gamma))
iter_PI = np.empty(len(gamma))
time_PI = np.empty(len(gamma))
value_QL = np.empty(len(gamma))
time_QL = np.empty(len(gamma))

for i in range(len(gamma)):
    vi = mdp.ValueIteration(T, R, gamma=gamma[i], epsilon=0.01, max_iter=1000)
    vi.run()
    value_VI[i] = np.max(vi.V)
    iter_VI[i] = vi.iter
    time_VI[i] = vi.time

    pi = mdp.PolicyIterationModified(T, R, gamma=gamma[i], epsilon=0.01, max_iter=1000)
    pi.run()
    value_PI[i] = np.max(pi.V)
    iter_PI[i] = pi.iter
    time_PI[i] = pi.time

    ql = mdp.QLearning(T, R, gamma=gamma[i], alpha=0.9, epsilon=0.9, n_iter=200000)
    ql.run()
    value_QL[i] = np.max(ql.V)
    time_QL[i] = ql.time

plt.figure()
plt.plot(gamma, value_VI, '-o', label='VI')
plt.plot(gamma, value_PI, '-o', label='PI')
plt.plot(gamma, value_QL, '-o', label='QL')
plt.legend()
plt.title('Max Utility')
plt.xlabel('Discount factor')
plt.show()

plt.figure()
plt.plot(gamma, iter_VI, '-o', label='VI')
plt.plot(gamma, iter_PI, '-o', label='PI')
plt.legend()
plt.title('Number of iterations')
plt.xlabel('Discount factor')
plt.show()

plt.figure()
plt.plot(gamma, time_VI, '-o', label='VI')
plt.plot(gamma, time_PI, '-o', label='PI')
plt.plot(gamma, time_QL, '-o', label='QL')
plt.legend()
plt.title('Time(s)')
plt.xlabel('Discount factor')
plt.show()
