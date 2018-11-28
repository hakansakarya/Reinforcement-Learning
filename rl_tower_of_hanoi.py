import random
import numpy as np
from copy import deepcopy
import time

class Tower_of_Hanoi(object):
	def __init__(self,state):
		#initialize Tower of Hanoi object
		self.num_of_disks = sum(1 for pin in state for disk in pin)
		self.pins = state


	def __eq__(self, other):
		#Method for object comparison
		for i, pin in enumerate(self.pins):
			if pin != other.pins[i]:
				return False
		return True
        
	def __str__(self):
		#to string method for printing
		pin1 = "+".join(str(x) for x in self.pins[0])
		pin2 = "+".join(str(x) for x in self.pins[1])
		pin3 = "+".join(str(x) for x in self.pins[2])
		return "Pin 1: {0} Pin 2: {1} Pin 3: {2}".format(pin1, pin2, pin3)

	def move_disk(self, action):
		#move disk from pin to pin given an action
		source_pin_idx, target_pin_idx = action
		pins = deepcopy(self.pins)
		disk = pins[source_pin_idx].pop()
		pins[target_pin_idx].append(disk)
		return Tower_of_Hanoi(pins)

	
	def is_goal_state(self):
		#returns whether a state is a goal state or not
		if len(self.pins[2]) == 2:
			last = self.pins[2][1]
			if last == 'B':
				return True
		return False

	def bigger_disk_on_top(self):
		#returns true if the bigger disk A is on top given the current state of the environment
		for pin in self.pins:
			if not pin == []:
				if pin[0] == 'B' and len(pin) == 2:
					return True
		return False

class MDP(object):
	def __init__(self):
		#initialize mdp environment
		self.gamma = 0.9
		self.states = [
						Tower_of_Hanoi([['A', 'B'], [], []]),
						Tower_of_Hanoi([['A'], ['B'], []]),
						Tower_of_Hanoi([['A'], [], ['B']]),
						Tower_of_Hanoi([[], ['A'], ['B']]),
						Tower_of_Hanoi([[], [], ['A', 'B']]),
						Tower_of_Hanoi([[], ['A', 'B'], []]),
						Tower_of_Hanoi([['B'], ['A'], []]),
						Tower_of_Hanoi([['B'], [], ['A']]),
						Tower_of_Hanoi([['B', 'A'], [], []]),
						Tower_of_Hanoi([[], ['B', 'A'], []]),
						Tower_of_Hanoi([[], [], ['B', 'A']]),
						Tower_of_Hanoi([[], ['B'], ['A']])
						]
		self.actions = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
		self.policies = [None for state in range(len(self.states))]
		self.q_values = []

	def initialize_q_values(self):
		for state in self.get_states():
			for action in self.get_actions(state):
				self.q_values.append((state, action, 0))

	def get_q_value(self, state, action):
		for s, a, value in self.q_values:
			if state == s and action == a:
				return value

	def get_q_values(self):
		return self.q_values

	def update_q_value(self, state, action, value):
		index = 0
		for q_val in self.get_q_values():
			if state == q_val[0] and action == q_val[1]:
				lst = list(self.q_values[index])
				lst[2] = value
				self.q_values[index] = lst
			index += 1

	def get_states(self):
		#return all states
		return self.states

	def get_actions(self, state):
		#return all available actions given a state
		if state.is_goal_state(): return []        
		actions = []
		for index, pin in enumerate(state.pins):
			for action in filter(lambda x: x[0] == index and len(pin) > 0, self.actions):
				actions.append(action)
		return actions

	def get_reward(self, state, action):
		#return the reward for taking an action in a given state
		if state.move_disk(action).is_goal_state():
			return 100
		elif state.move_disk(action).bigger_disk_on_top():
			return -10
		else:
			return -1

	def update_policy(self, state, action):
		#update the policy for a state
		for index, s in enumerate(self.states):
			if s == state:
				self.policies[index] = action

	def get_policy_action(self, state):
		#get the action attached to the policy of the given state
		for index, s in enumerate(self.states):
			if s == state:
				return self.policies[index]
	
	def select_best_action(self, state):
		state_q_values = []
		state_actions = [action for action in self.get_actions(state)]
		for action in state_actions:
			state_q_values.append(self.get_q_value(state,action))
		max_index = state_q_values.index(max(state_q_values))
		best_action = state_actions[max_index]
		return best_action

	def epsilon_greedy_select_action(self, state, epsilon):
		z = np.random.random()
		if z > epsilon:
			return self.select_best_action(state)
		else:
			return random.choice(self.get_actions(state))

	def q_learning(self, learning_rate, num_episodes, starting_state):
		self.initialize_q_values()
		for episode in range(num_episodes):
			state = starting_state
			while True:
				#best_action = self.select_best_action(state)
				best_action = self.epsilon_greedy_select_action(state, 0.3)
				reward = self.get_reward(state, best_action)
				new_state = state.move_disk(best_action)

				state_q_value = self.get_q_value(state, best_action)
				next_state_q_values = []

				if new_state.is_goal_state():
					next_state_q_values.append(0)
				else:
					next_state_actions = [action for action in self.get_actions(new_state)]
					for action in next_state_actions:
						next_state_q_values.append(self.get_q_value(new_state,action))

				max_next_q_val = max(next_state_q_values)
				update = learning_rate * (reward + self.gamma * max_next_q_val - state_q_value)

				self.update_q_value(state, best_action, state_q_value + update)
				self.update_policy(state, best_action)

				state = new_state
				if new_state.is_goal_state():
					break
				
if __name__ == "__main__":

	mdp_hanoi = MDP()
	start = time.time()
	for state in mdp_hanoi.get_states():
		if not state.is_goal_state():
			mdp_hanoi.q_learning(0.1, 10000, state)
	end = time.time()
	print('Q-learning for all initial states took: ', end-start, ' seconds.')
	print('Average time for one run: ', (end-start)/len(mdp_hanoi.get_states()))
	
	q_values = mdp_hanoi.get_q_values()

	'''
	for state, action, value in q_values:
		print('\n***\n')
		print('State: ', state)
		print('Action: ', action)
		print('Q-Value: ', value)
		print('\n***\n')
	'''
	
	for state in mdp_hanoi.get_states():
		print('\n***\n')
		print('State: ', state)
		print('Optimal Policy: ', mdp_hanoi.get_policy_action(state))
		print('Q-Value: ', mdp_hanoi.get_q_value(state, mdp_hanoi.get_policy_action(state)))
		print('\n***\n')
