
# learner = []
# parameter = []

#         learner.append(ql)
#         parameter.append([alpha[i], epsilon[j]])
# # Plot max utility
# plt.figure()
# for i in range(len(learner)):
#     plt.plot([a_dict['Iteration'] for a_dict in learner[i].run_stats],
#              [a_dict['Max V'] for a_dict in learner[i].run_stats],
#              label='A:{}, E:{}'.format(parameter[i][0], parameter[i][1]))
# plt.legend(loc='upper right')
# plt.xlabel('Iterations')
# plt.ylabel('Max Utility')
# plt.title('QL Convergence')
# plt.show()
# # Plot delta
# plt.figure()
# for i in range(len(learner)):
#     plt.plot([a_dict['Iteration'] for a_dict in learner[i].run_stats],
#              [a_dict['Error'] for a_dict in learner[i].run_stats],
#              label='A:{}, E:{}'.format(parameter[i][0], parameter[i][1]))
# plt.legend()
# plt.xlabel('Iterations')
# plt.ylabel('Delta')
# plt.title('QL Convergence')
# plt.show()
# Plot processing time
ql = mdp.QLearning(T, R, gamma=0.99, alpha=0.1, epsilon=0.9, n_iter=10000)
ql.run()
print(ql.time)
print(ql.run_stats[0])
plt.plot([a_dict['Iteration'] for a_dict in ql.run_stats], [a_dict['Reward'] for a_dict in ql.run_stats], label='Reward')
plt.plot([a_dict['Iteration'] for a_dict in ql.run_stats], [a_dict['Max V'] for a_dict in ql.run_stats], label='Max V')
plt.plot([a_dict['Iteration'] for a_dict in ql.run_stats], [a_dict['Mean V'] for a_dict in ql.run_stats], label='Mean V')
plt.legend()
plt.show()
