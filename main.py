# Reference
# 1. https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter9-drl/dqn-cartpole-9.6.1.py
# 2. City University, INM707 Deep Learning 3: Optimization, Workshop 6\
# 3. https://github.com/the-deep-learners/TensorFlow-LiveLessons/blob/master/notebooks/cartpole_dqn.ipynb
# 4. https://github.com/pythonlessons/Reinforcement_Learning/tree/master/03_CartPole-reinforcement-learning_Dueling_DDQN
# 5. https://github.com/pythonlessons/Reinforcement_Learning/tree/master/02_CartPole-reinforcement-learning_DDQN
# 6  https://github.com/gsurma/cartpole
# 7. https://github.com/lazyprogrammer/machine_learning_examples

"""Contains the code for running and testing part 1, requires imports from util and agent"""
import numpy as np
import pandas as pd
from environment import small_zone_1, zone_1
from util import plot_reward, plot_steps, get_mini_stations_dict, save_results, get_stations_dict
from agent import Q_Agent

env = small_zone_1()
stations = get_mini_stations_dict()


# env = zone_1()
# stations = get_stations_dict()

def run_q_agent(policy='ε–greedy', save=False):
	"""Runs a q agent according to a policy"""
	agent = Q_Agent()
	all_iterations, all_rewards, step_count = agent.train(env, iter_n=1000, policy=policy)
	plot_reward(all_iterations, all_rewards)
	plot_steps(all_iterations, step_count)
	# print("best route is {}".format(agent.test(env)))
	# if save:
	# 	save_results(all_iterations, all_rewards, step_count)
	# optimum_route = agent.test(env, stations)
	# print(optimum_route)


def random_search():
	"""Random search to determine starting point for the model and best params"""
	gamma = 0.7
	alpha = 0.3
	epsilon = 1
	exploration_rate_decay = 0.87

	max_tries = 10
	best_score = -1000
	scores = {}

	for attempt in range(max_tries):

		agent = Q_Agent(epsilon=1, alpha=alpha, gamma=gamma, exploration_rate_decay=exploration_rate_decay)
		_, rewards, steps = agent.train(env, iter_n=300, policy='ε–greedy', print_results=False)
		print(np.mean(rewards))
		scores[attempt] = np.mean(rewards)

		print(
			"Score:{}, gamma {}, alpha {}, epsilon {}, e_decay_rate{}".format(
				scores[attempt], gamma, alpha, epsilon, exploration_rate_decay))

		if scores[attempt] > best_score:
			best_score = scores[attempt]
			print(best_score)
			best_gamma = gamma
			best_alpha = alpha
			best_epsilon = epsilon
			best_decay = exploration_rate_decay

		gamma = best_gamma + (np.random.randint(-1, 2) / 10)
		gamma = min(1, gamma)
		gamma = max(0, gamma)
		alpha = best_alpha + (np.random.randint(-1, 2) / 10)
		alpha = min(1, alpha)
		alpha = max(0, alpha)
		epsilon = 1
		exploration_rate_decay = best_decay + np.random.randint(-1, 2) / 100
		exploration_rate_decay = min(0.99, exploration_rate_decay)
		exploration_rate_decay = max(0.7, exploration_rate_decay)

	print("Best validation_accuracy:", best_score)
	print("Best settings:")
	print("best gamma:", best_gamma)
	print("best alpha:", best_alpha)
	print("best epsilon:", best_epsilon)
	print("best decay:", best_decay)


def grid_search_epsilon(environmnet, policy='ε–greedy', parameter='epsilon'):
	"""Grid search for epsilon values"""
	parameter_values = []
	avg_scores = []
	avg_steps = []

	count = 1
	decay_search = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.99]
	for param_num in decay_search:

		agent = Q_Agent(exploration_rate_decay=param_num, epsilon=1)
		all_iterations, all_rewards, step_count = agent.train(environmnet, print_results=True, iter_n=1000,
															  policy=policy)
		avg_scores.append(np.mean(all_rewards))
		avg_steps.append(np.mean(step_count))
		parameter_values.append(param_num)
		rewards_data = np.array([all_iterations, all_rewards])
		step_data = np.array([all_iterations, step_count])

		np.savetxt(
			'/Users/matthewgalloway/Documents/RF/q_learning/' + parameter + '_inv/' + parameter + '_rewards_' + str(
				param_num) + '.csv', rewards_data.transpose(), delimiter=",")
		np.savetxt(
			'/Users/matthewgalloway/Documents/RF/q_learning/' + parameter + '_inv/' + parameter + '_steps_' + str(
				param_num) + '.csv', step_data.transpose(), delimiter=",")
		if count % 50 == 0:
			print('iteration {} of 10'.format(count))

		count += 1
	results = {
		'param_values': parameter_values,
		'avg_scores': avg_scores,
		'avg_steps': avg_steps,

	}
	print(results)
	return pd.DataFrame(results)


def grid_search_param(environmnet, policy='ε–greedy', parameter='alpha'):
	"""Grid search for alpha or gamma adjustable via the parameter field"""

	parameter_values = []
	avg_scores = []
	avg_steps = []

	count = 1

	for param_num in np.linspace(0.2, 1, 9):
		if parameter == 'alpha':
			agent = Q_Agent(alpha=param_num)
		elif parameter == 'gamma':
			agent = Q_Agent(gamma=param_num)

		all_iterations, all_rewards, step_count = agent.train(environmnet, print_results=True, iter_n=1000,
															  policy=policy)
		avg_scores.append(np.mean(all_rewards))
		avg_steps.append(np.mean(step_count))
		parameter_values.append(param_num)
		rewards_data = np.array([all_iterations, all_rewards])
		step_data = np.array([all_iterations, step_count])

		np.savetxt(
			'/Users/matthewgalloway/Documents/RF/q_learning/' + parameter + '_inv/' + parameter + '_rewards_' + str(
				param_num) + '.csv', rewards_data.transpose(), delimiter=",")
		np.savetxt(
			'/Users/matthewgalloway/Documents/RF/q_learning/' + parameter + '_inv/' + parameter + '_steps_' + str(
				param_num) + '.csv', step_data.transpose(), delimiter=",")
		if count % 50 == 0:
			print('iteration {} of 10'.format(count))

		count += 1
	results = {
		'alpha_values': parameter_values,
		'avg_scores': avg_scores,
		'avg_steps': avg_steps,

	}
	print(results)
	return pd.DataFrame(results)



if __name__ == '__main__':

	"""Commented out our optimisation steps. 
	Currently only the q agent will on the default epsilon greedy policy """

	# grid_search_param(env, parameter='alpha')
	# grid_search_epsilon(env)
	# random_search()
	run_q_agent(policy='ε–greedy', save=False)

