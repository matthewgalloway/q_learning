import numpy as np

"""Module contains the environments used for part 1"""


class Environment:
	"""Environment for the agent to take actions within
		Defines state space, available actions and rewards at each step"""

	def __init__(self):
		self.state = 0

	def set_state(self, state):
		self.state = state
		return self.state

	def reset(self):
		states = list(self.rewards.keys())
		states.remove(12)
		self.state = np.random.choice(states)
		# self.state = 0
		return self.state

	def update_terminal(self, terminal_state):
		self.terminal_state = terminal_state

	def update_actions_rewards(self, actions, rewards):
		self.rewards = rewards
		self.actions = actions

	def available_actions(self, state):
		if type(state) == int:
			pass
		else:
			state = np.asscalar(np.reshape(state, [1]))
		return self.actions[state]

	def step(self, action):
		if (self.state == self.terminal_state) & (action == 12):
			reward = 50
			self.state = action
			return self.state, reward
		else:
			self.state = action
			reward = self.rewards[self.state]
			return self.state, reward

	def current_state(self):
		return self.state



def small_zone_1(penalty=-1, goal=100):
	"""Returns actions, rewards and terminal states for a subsect of zone 1"""

	underground_map = Environment()

	actions = {0: (3, 12),
			   1: (3, 2, 12),
			   2: (1, 5, 12),
			   3: (0, 1, 4, 9, 12),
			   4: (3, 5, 12),
			   5: (2, 4, 6, 11, 12),
			   6: (5, 7, 12),
			   7: (6, 12),
			   8: (6, 11, 12),
			   9: (3, 10, 12),
			   10: (9, 11, 12),
			   11: (8, 10, 12),
			   12: 12
			   }

	start = -1
	central = -4
	northern = -5
	victoria = -1
	get_off = -20

	rewards = {0: start,
			   1: northern,
			   2: northern,
			   3: central,
			   4: central,
			   5: central,
			   6: central,
			   7: central,
			   8: victoria,
			   9: victoria,
			   10: victoria,
			   11: victoria,
			   12: get_off
			   }

	# old_rewards = {0: penalty,
	# 			   1: penalty,
	# 			   2: penalty,
	# 			   3: penalty,
	# 			   4: penalty,
	# 			   5: penalty,
	# 			   6: penalty,
	# 			   7: goal,
	# 			   8: penalty,
	# 			   9: penalty,
	# 			   10: penalty,
	# 			   11: penalty,
	# 			   12: get_off
	# 			   }

	underground_map.update_actions_rewards(actions, rewards)
	underground_map.update_terminal(7)

	return underground_map

def zone_1(penalty=-1, goal=100):
	"""Returns actions, rewards and terminal states for all of zone 1"""
	underground_map = Environment()

	actions = {0: (17, 23),
			   1: (5, 26),
			   2: (21, 22, 29),
			   3: (13, 17, 22),
			   4: (20, 30),
			   5: (1, 15, 24),
			   6: (21),
			   7: (16, 29),
			   8: (10, 18, 25),
			   9: (16, 18),
			   10: (8, 30, 34),
			   11: (17, 33),
			   12: (17),
			   13: (3),
			   14: (31, 33),
			   15: (32, 34),
			   16: (7, 9, 27, 31,),
			   17: (0, 11, 12, 13, 27),
			   18: (8, 25, 31),
			   19: (21),
			   20: (4, 6),
			   21: (2, 19),
			   22: (2, 3, 21, 23),
			   23: (0, 22),
			   24: (15, 25, 26),
			   25: (8, 18, 24),
			   26: (24, 1),
			   27: (16),
			   28: (32, 34),
			   29: (2, 7),
			   30: (4, 10),
			   31: (14, 16, 18, 24),
			   32: (15, 28),
			   33: (14, 24),
			   34: (10, 15, 28, 32, 34)
			   }

	rewards = {0: penalty,
			   1: penalty,
			   2: penalty,
			   3: penalty,
			   4: penalty,
			   5: penalty,
			   6: penalty,
			   7: penalty,
			   8: penalty,
			   9: penalty,
			   10: penalty,
			   11: penalty,
			   12: penalty,
			   13: penalty,
			   14: penalty,
			   15: penalty,
			   16: penalty,
			   17: penalty,
			   18: penalty,
			   19: penalty,
			   20: penalty,
			   21: penalty,
			   22: penalty,
			   23: penalty,
			   24: penalty,
			   25: penalty,
			   26: penalty,
			   27: penalty,
			   28: penalty,
			   29: penalty,
			   30: penalty,
			   31: penalty,
			   32: penalty,
			   33: penalty,
			   34: goal
			   }

	underground_map.update_actions_rewards(actions, rewards)
	underground_map.update_terminal(34)

	return underground_map