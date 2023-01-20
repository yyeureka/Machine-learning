from hiive.mdptoolbox import example, mdp
import numpy as np
import matplotlib.pyplot as plt

SEED = 0

problem_size = [20, 100, 400]
value_VI = np.empty(len(problem_size))
iter_VI = np.empty(len(problem_size))
time_VI = np.empty(len(problem_size))
value_PI = np.empty(len(problem_size))
iter_PI = np.empty(len(problem_size))
time_PI = np.empty(len(problem_size))
value_QL = np.empty(len(problem_size))
time_QL = np.empty(len(problem_size))

for i in range(len(problem_size)):
    np.random.seed(SEED)
    T, R = example.forest(S=problem_size[i])

    vi = mdp.ValueIteration(T, R, gamma=0.9, epsilon=0.01, max_iter=1000)
    vi.run()
    value_VI[i] = np.max(vi.V)
    iter_VI[i] = vi.iter
    time_VI[i] = vi.time

    pi = mdp.PolicyIterationModified(T, R, gamma=0.9, epsilon=0.01, max_iter=1000)
    pi.run()
    value_PI[i] = np.max(pi.V)
    iter_PI[i] = pi.iter
    time_PI[i] = pi.time

    ql = mdp.QLearning(T, R, gamma=0.9, alpha=0.9, epsilon=0.4, n_iter=200000)
    ql.run()
    value_QL[i] = np.max(ql.V)
    time_QL[i] = ql.time

plt.figure()
plt.plot(problem_size, value_VI, '-o', label='VI')
plt.plot(problem_size, value_PI, '-o', label='PI')
plt.plot(problem_size, value_QL, '-o', label='QL')
plt.legend()
plt.title('Max Utility')
plt.xlabel('Problem sizes')
plt.show()

plt.figure()
plt.plot(problem_size, iter_VI, '-o', label='VI')
plt.plot(problem_size, iter_PI, '-o', label='PI')
plt.legend()
plt.title('Number of iterations')
plt.xlabel('Problem sizes')
plt.show()

plt.figure()
plt.plot(problem_size, time_VI, '-o', label='VI')
plt.plot(problem_size, time_PI, '-o', label='PI')
plt.plot(problem_size, time_QL, '-o', label='QL')
plt.legend()
plt.title('Time(s)')
plt.xlabel('Problem sizes')
plt.show()
