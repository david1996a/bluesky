import numpy as np
from ddpg import DDPG
from buffer import Buffer

NUM_LEARN_STEPS_PER_ENV_STEP = 1
GAMMA = 0.9
BATCH_SIZE = 2

class MADDPG:
	def __init__(self, state_size, observation_size, action_size, n_agents, buffer_samples):
		self.state_size	= state_size
		self.observation_size = observation_size
		self.action_size = action_size
		self.n_agents = n_agents
		self.whole_action_size = self.action_size * self.n_agents
		self.buffer_samples = buffer_samples	#Number of transitions to be stored in the buffer.
		self.episodes_before_training = 2
		"""Initializing the DDPG network for each agent"""
		self.maddpg_agents = [DDPG(state_size = self.state_size, 
			observation_size = self.observation_size, action_size = self.action_size,
			num_agents=self.n_agents) for i in range(self.n_agents)]

		"""Creating the buffer to store the trasnitions."""
		self.memory = Buffer(max_size=self.buffer_samples, input_shape=self.state_size,
		 	observation_shape = self.observation_size, number_of_agents=self.n_agents)

	def step(self, i_episode, state, observations, actions, reward, next_state, next_observations, done):
		self.memory.store_transition(state, observations, actions, reward, next_state, next_observations, done)

		if self.memory.mem_cntr > BATCH_SIZE and i_episode > self.episodes_before_training:
			for _ in range(NUM_LEARN_STEPS_PER_ENV_STEP):
				samples = self.memory.sample()
				print(samples)
				self.learn(samples, agent_no, GAMMA)
			self.soft_update_all()

	def learn(self, sample, agent_no, gamma):
		states, observations, actions, rewards, next_states, next_observations, dones = sample
		full_next_actions = np.zeros((self.n_agents, self.observations_size))

		for index, agent in enumerate(self.maddpg_agents):
			full_next_actions[index,:] = agent.actor_target.forward(next_observations)
		full_next_actions = T.tensor(full_next_actions).to(DEVICE)
		
		for agent in self.maddpg_agents:
			agent.learn((state, observation, actions, reward, next_states, next_observation, next_actions_full), GAMMA)

	def sample_memory(self):
		state, observations, action, reward, next_state, next_observations, done = \
						self.memory.sample_buffer(self.batch_size)

		states = T.tensor(state).to(DEVICE)
		observations = T.tensor(state).to(DEVICE)
		rewards = T.tensor(reward).to(DEVICE)
		dones = T.tensor(done).to(DEVICE)
		actions = T.tensor(action).to(DEVICE)
		next_states = T.tensor(next_state).to(DEVICE)
		next_observations = T.tensor(next_observations).to(DEVICE)

		return states, observations, actions, rewards, next_states, next_observations, dones

	def act(self, observations, i_episode, add_noise=True):
		actions = []
		for index, agent in enumerate(self.maddpg_agents):
			action = agent.act(observations[index], i_episode)
			actions.append(action)

		return actions

