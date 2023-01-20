from hiive.mdptoolbox import mdp
from gym.envs.toy_text.frozen_lake import generate_random_map
from openai import converter
import numpy as np
import matplotlib.pyplot as plt

SEED = 0

# 4*4
np.random.seed(SEED)
random_map = generate_random_map(size=4)
print(np.reshape(random_map, (4, 1)))
T, R = converter("FrozenLake-v0", desc=random_map)

# VI
epsilon = [0.001, 0.01, 0.1]
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.xlabel('Iterations')
plt.minorticks_on()
ax2 = ax1.twinx()
time = []
for i in range(len(epsilon)):
    vi = mdp.ValueIteration(T, R, gamma=0.99, epsilon=epsilon[i], max_iter=1000)
    vi.run()
    time.append(vi.time)
    ax1.plot([a_dict['Iteration'] for a_dict in vi.run_stats], [a_dict['Max V'] for a_dict in vi.run_stats], label='{}'.format(epsilon[i]))
    ax2.plot([a_dict['Iteration'] for a_dict in vi.run_stats], [a_dict['Error'] for a_dict in vi.run_stats], label='{}'.format(epsilon[i]))
# Plot convergence
plt.legend()
ax1.set_ylabel('Max Utility')
ax2.set_ylabel('Delta')
plt.title('VI Convergence')
plt.show()
# Plot processing time
plt.figure()
plt.plot(epsilon, time, '-o')
plt.xlabel('Epsilon')
plt.ylabel('Time(s)')
plt.title('VI Time')
plt.show()

# PI
pi = mdp.PolicyIteration(T, R, gamma=0.99, max_iter=1000)
pi.run()
# Plot convergence
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.xlabel('Iterations')
plt.minorticks_on()
ax2 = ax1.twinx()
ax1.plot([a_dict['Iteration'] for a_dict in pi.run_stats], [a_dict['Max V'] for a_dict in pi.run_stats])
ax2.plot([a_dict['Iteration'] for a_dict in pi.run_stats], [a_dict['Error'] for a_dict in pi.run_stats])
ax1.set_ylabel('Max Utility')
ax2.set_ylabel('Delta')
plt.title('PI Convergence')
plt.show()
# Plot processing time
print(pi.time)

# QL
# epsilon = [0, 0.2, 0.4, 0.6, 0.8, 1]
# time_QL = np.empty(len(epsilon))
# for i in range(len(epsilon)):
#     np.random.seed(SEED)
#     ql = mdp.QLearning(T, R, gamma=0.99, alpha=0.9, epsilon=epsilon[i], n_iter=100000)
#     ql.run()
#     plt.plot([a_dict['Iteration'] for a_dict in ql.run_stats], [a_dict['Max V'] for a_dict in ql.run_stats], label='E:{}'.format(epsilon[i]))
#     time_QL[i] = ql.time
# plt.legend()
# plt.xlabel('Iteration')
# plt.ylabel('Max Utility')
# plt.title('QL convergence')
# plt.show()
# # Time
# plt.figure()
# plt.plot(epsilon, time_QL, '-o', label='QL')
# plt.title('Time(s)')
# plt.xlabel('Epsilon')
# plt.show()
#
# alpha = [0.1, 0.3, 0.5, 0.7, 0.9]
# time_QL = np.empty(len(alpha))
# for i in range(len(alpha)):
#     np.random.seed(SEED)
#     ql = mdp.QLearning(T, R, gamma=0.99, alpha=alpha[i], epsilon=0.6, n_iter=100000)
#     ql.run()
#     plt.plot([a_dict['Iteration'] for a_dict in ql.run_stats], [a_dict['Max V'] for a_dict in ql.run_stats], label='A:{}'.format(alpha[i]))
#     time_QL[i] = ql.time
# plt.legend()
# plt.xlabel('Iteration')
# plt.ylabel('Max Utility')
# plt.title('QL convergence')
# plt.show()
# # Time
# plt.figure()
# plt.plot(alpha, time_QL, '-o', label='QL')
# plt.title('Time(s)')
# plt.xlabel('Learning rate')
# plt.show()

ql = mdp.QLearning(T, R, gamma=0.99, alpha=0.9, epsilon=0.6, n_iter=400000)
ql.run()

# # Plot policy
# plt.figure()
# plt.plot(vi.policy, label='VI')
# plt.plot(pi.policy, label='PI')
# plt.legend()
# plt.xlabel('States')
# plt.ylabel('Actions')
# plt.title('Policy')
# plt.show()
