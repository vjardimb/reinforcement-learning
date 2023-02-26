import math

import numpy as np


def discretize_state(state, limits, no_bins):
	"""
	Transforms a continuous state space in discrete coordinates.
	"""
	disc_state = [0]*(len(state))
	for i, coord in enumerate(state):
		total_range = limits[i][1] - limits[i][0]

		if coord < limits[i][0]:
			bin_index = 0
		elif coord > limits[i][1]:
			bin_index = no_bins[i] - 1
		else:
			scale_portion = (coord - limits[i][0])/total_range
			bin_index = int(np.floor(no_bins[i]*scale_portion))

		disc_state[i] = int(bin_index)

	return tuple(disc_state)


def update_q_table(Q_table, new_state, current_state, action, reward, lr, disc_factor=0.99):
	"""
	Updates Q table based on rewards using bellman's equation.
	"""
	learned_Q_val = reward + disc_factor*np.max(Q_table[new_state])
	old_Q_val = Q_table[current_state][action]

	Q_table[current_state][action] = (1 - lr)*old_Q_val + lr*learned_Q_val

	return Q_table


def get_opt_action(Q_table, state):
	"""
	Chooses action based on greedy policy i.e. looks at the action the the highest expected reward.
	"""
	return int(np.argmax(Q_table[state]))


def get_l_rate(episode, min_rate=0.01):
	"""
	Returns learning rate based on the episode number. As time passes, the lr decreases exponentially.
	"""
	return max(min_rate, min(1.0, 1.0 - math.log10((episode + 1) / 25)))


def exploration_rate(episode, min_rate=0.1):
	"""
	Defines the exploration vs exploitation ratio, the higher exploration rate the more the agent chooses to explore the
	environment instead of taken the action that rewards you the most
	"""
	return max(min_rate, min(1.0, 1.0 - math.log10((episode + 1) / 25)))
