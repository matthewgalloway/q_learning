# Reference
# 1. https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter9-drl/dqn-cartpole-9.6.1.py
# 2. City University, INM707 Deep Learning 3: Optimization, Workshop 6\
# 3. https://github.com/the-deep-learners/TensorFlow-LiveLessons/blob/master/notebooks/cartpole_dqn.ipynb
# 4. https://github.com/pythonlessons/Reinforcement_Learning/tree/master/03_CartPole-reinforcement-learning_Dueling_DDQN
# 5. https://github.com/pythonlessons/Reinforcement_Learning/tree/master/02_CartPole-reinforcement-learning_DDQN
# 6  https://github.com/gsurma/cartpole
# 7. https://github.com/lazyprogrammer/machine_learning_examples

import numpy as np


"""Module contains the two Agents used for experiments"""


class Q_Agent:
	"""Q Learning Agent .
		Contains Q table and learning parameters.
			Updates Q table based on specified policy, default is ε–greedy"""

	def __init__(self, alpha=0.2, gamma=0.7, exploration_rate_decay=0.8, epsilon=1):
		self.epsilon = epsilon
		self.exploration_rate_decay = exploration_rate_decay
		self.gamma = gamma
		self.alpha = alpha
		self.Q = np.zeros((13, 13))
		self.round_reward = 0
		self.all_rewards = []
		self.all_iterations = []
		self.total_run = 0
		self.step_count = []

		self.k_n = np.ones((13, 13))
		self.n = 1
		self.c = 20



	def train(self, env, iter_n=125, policy='ε–greedy', print_results=True):
		""" Trains Q agent updating q table according to a policy """

		for i in range(iter_n):  # looping through the number of training runs

			state = env.reset()

			if print_results & (i % 10 == 0):  # prints results to the terminal
				self.print_results(i, iter_n)

			if i != 0: # Update totals in memory if not the first run
				self.update_totals(i, no_steps)

			self.epsilon *= self.exploration_rate_decay
			episode_complete = False
			no_steps = 0
			while not episode_complete:  # looping through a single episode updating Q table
				if state != 12:
					available_actions = env.available_actions(state)
					action = self.define_action(state, available_actions, policy)
					new_state, reward = env.step(action)
					# print('state is {}, available actions are {}, next state is {}.,rewards is{}'.format(state,available_actions,new_state,reward))
					no_steps += 1
					self.round_reward += reward
					self.update_q(new_state, state, action, reward)
					if (state == 7) & (new_state == 12):
						episode_complete = True
					action_memory = available_actions
					last_state = state
					state = new_state
				else:
					state = last_state
					available_actions = action_memory
					action = self.define_action(state, available_actions, policy)
					new_state, reward = env.step(action)
					# print('state is {}, available actions are {}, next state is {}.,rewards is{}'.format(state,available_actions,new_state,reward))
					no_steps += 1
					self.round_reward += reward
					self.update_q(new_state, state, action, reward)
					if (state == 7) & (new_state == 12):
						episode_complete = True
					state = new_state


		return self.all_iterations, self.all_rewards, self.step_count

	def print_results(self, i, iter_n):
		"""Prints Results to the terminal"""
		print('iteration {} of {}, average reward is {}'.format(i, iter_n,
																sum(self.all_rewards[i - 10:i]) / 10)
			  )
		print('alpha is {}, gamma is {}, epsilon is {}, decay rate is {}'.format(self.alpha, self.gamma,
																				 self.epsilon,
																				 self.exploration_rate_decay)
			  )

	def update_totals(self, i, no_steps):
		"""Updates totals for that iteration in agents memory"""
		self.all_rewards.append(self.round_reward)
		self.round_reward = 0
		self.all_iterations.append(i)
		self.step_count.append(no_steps)

	def define_action(self, state, available_actions, policy):
		"""Choses action based on policy based on available actions provided by environemnt class"""
		# if state == 12:
		# 	np.random.choice(available_actions)
		if type(available_actions) == int:  # Check to see if there is only one action possible
			next_action = available_actions

		else:  # Follows policy if more than one action
			if policy == 'ε–greedy':
				exploration_rate_threshold = np.random.uniform(0, 1)
				if exploration_rate_threshold < self.epsilon:
					next_action = np.random.choice(available_actions)
				else:
					next_action = available_actions[np.argmax(self.Q[state, available_actions])]
			elif policy == 'random':
				next_action = np.random.choice(available_actions)
			elif policy == 'UCB':
				next_action = available_actions[np.argmax(self.Q[state, available_actions] + self.c * np.sqrt((np.log(self.n)) / self.k_n[state, available_actions]))]
				self.n += 1
				self.k_n[state, next_action] += 1
		return next_action

	def update_q(self, new_state, state, action, reward):
		"""updates the Q table in agent memory"""
		self.Q[state, action] = self.Q[state, action] + self.alpha * (
				reward + self.gamma * np.max(self.Q[new_state, :]) - self.Q[state, action])

	def display_q(self):
		"""returns the q table"""
		return np.round(self.Q)


	def test(self, env):
		"""returns best route based on agents memory"""
		routes = {k: [] for k in range(12)}
		scores = {k: [] for k in range(12)}
		total_rewards =0
		for starting_station in routes:
			state = env.set_state(starting_station)
			episode_complete = False
			while not episode_complete:
				if state != 12:
					available_actions = env.available_actions(state)
					action = available_actions[np.argmax(self.Q[state, available_actions])]
					new_state, reward = env.step(action)
					if (state == 7) & (new_state == 12):
						episode_complete = True
					action_memory = available_actions
					last_state = state
					state = new_state
					total_rewards += reward
				else:
					state = last_state
					available_actions = action_memory
					action = available_actions[np.argmax(self.Q[state, available_actions])]
					new_state, reward = env.step(action)
					if (state == 7) & (new_state == 12):
						episode_complete = True
					state = new_state
					total_rewards += reward
				routes[starting_station].append(new_state)
				scores[starting_station].append(total_rewards)
				total_rewards = 0
		print(scores)
		return routes



