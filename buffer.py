import numpy as np 

class Buffer:
	def __init__(self, max_size, input_shape, observation_shape, number_of_agents):
		self.max_size= max_size
		self.mem_cntr = 0
		self.state_memory = np.zeros((self.max_size, input_shape), dtype=np.float32)
		self.observations_memory = np.zeros((self.max_size, number_of_agents, observation_shape), dtype=np.float32)
		self.action_memory = np.zeros((self.max_size, number_of_agents, 1), dtype=np.float32)
		self.reward_memory = np.zeros((self.max_size,1), dtype=np.float32)
		self.next_state_memory = np.zeros((self.max_size, input_shape), dtype=np.float32)
		self.next_observations_memory = np.zeros((self.max_size, number_of_agents, observation_shape), dtype=np.float32)
		self.done_memory = np.zeros((self.max_size,1), dtype=np.uint8)

	def store_transition(self, state, observations, actions, reward, next_state, next_observations, done):
		index = self.mem_cntr%self.max_size
		self.state_memory[index] = state
		self.observations_memory[index] = observations
		self.action_memory[index] = actions
		self.reward_memory[index] = reward
		self.next_state_memory[index] = next_state
		self.next_observations_memory[index] = next_observations
		self.done_memory[index] = done

		self.mem_cntr += 1 

	def sample_buffer(self, batch_size):
		max_mem = min(self.mem_cntr, self.max_size)
		batch = np.random.choice(max_mem, batch_size, replace=False)
		print(batch)

		states = self.state_memory[batch]
		observations = self.observations_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		next_states = self.next_state_memory[batch]
		next_observations = self.next_observations_memory[batch]
		dones = self.done_memory[batch]
		#print(observations[0], states[0])
		return states, observations, actions, rewards, next_states, next_observations, dones