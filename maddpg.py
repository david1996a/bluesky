import numpy as np
from ddpg import DDPG
from buffer import Buffer
import torch as T

NUM_LEARN_STEPS_PER_ENV_STEP = 3
GAMMA = 0.9
BATCH_SIZE = 3000
SAMPLE_SIZE = 1000
DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")

class MADDPG:
	def __init__(self, state_size, observation_size, action_size, n_agents, buffer_samples):
		self.state_size	= state_size
		self.observation_size = observation_size
		self.action_size = action_size
		self.n_agents = n_agents
		self.whole_action_size = self.action_size * self.n_agents
		self.buffer_samples = buffer_samples	#Number of transitions to be stored in the buffer.
		self.episodes_before_training = 0
		"""Initializing the DDPG network for each agent"""
		self.maddpg_agents = [DDPG(state_size = self.state_size, 
			observation_size = self.observation_size, action_size = self.action_size,
			num_agents=self.n_agents) for i in range(self.n_agents)]

		"""Creating the buffer to store the trasnitions."""
		self.memory = Buffer(max_size=BATCH_SIZE , input_shape=self.state_size,
		 	observation_shape = self.observation_size, number_of_agents=self.n_agents)

	def step(self, i_episode, state, observations, actions, reward, next_state, next_observations, dones):
		self.memory.store_transition(state, observations, actions, reward, next_state, next_observations, dones)

		if self.memory.mem_cntr > BATCH_SIZE and i_episode > self.episodes_before_training:
			for _ in range(NUM_LEARN_STEPS_PER_ENV_STEP):
				for agent_no in range(self.n_agents):
					samples = self.sample_memory()
					self.learn(samples, agent_no, GAMMA)
			self.soft_update_all()

	def learn(self, sample, agent_no, gamma):
		states, observations, actions, rewards, next_states, next_observations, dones = sample

		full_next_actions = T.zeros((SAMPLE_SIZE, self.n_agents, self.action_size), dtype=T.float32, device=DEVICE)
		for index, agent in enumerate(self.maddpg_agents):
			full_next_actions[:,index,:] = agent.actor_target.forward(next_observations[:,index,:])
		agent_rewards = rewards[:,agent_no]
		agent_dones = dones[:,agent_no]
		agent.learn((states, observations, actions, agent_rewards, next_states, next_observations, full_next_actions, agent_dones), 
			GAMMA, agent_no, self.n_agents)

	def sample_memory(self):
		state, observations, action, reward, next_state, next_observations, done = \
						self.memory.sample_buffer(SAMPLE_SIZE)

		states = T.tensor(state, device=DEVICE)
		observations = T.tensor(observations, device=DEVICE)
		rewards = T.tensor(reward, device=DEVICE)
		dones = T.tensor(done, device=DEVICE)
		actions = T.tensor(action, device=DEVICE)
		next_states = T.tensor(next_state, device=DEVICE)
		next_observations = T.tensor(next_observations, device=DEVICE)

		return states, observations, actions, rewards, next_states, next_observations, dones

	def act(self, observations, i_episode, add_noise=True):
		actions = []
		for index, agent in enumerate(self.maddpg_agents):
			action = agent.act(observations[index], i_episode)
			actions.append(action)

		return actions

	def soft_update_all(self):
		for agent in self.maddpg_agents:
			agent.soft_update_all()

	def save_maddpg(self):
		for agent_id, agent in enumerate(self.maddpg_agents):
			T.save(agent.actor_local.state_dict(), 'models/checkpoint_actor_local_'+str(agent_id)+'.pth')
			T.save(agent.critic_local.state_dict(), 'models/checkpoint_critic_local_'+str(agent_id)+'.pth')

	def load_maddpg(self):
		for agent, agent in enumerate(self.maddpg_agents):
			agent.actor_local.load_state_dict(T.load('models/checkpoint_actor_local_'+str(agent_id)
				+'.pth',map_location=lambda storage, loc: storage))
			agent.critic_local.load_state_dict(T.load('models/checkpoint_critic_local_'+str(agent_id)
				+'.pth',map_location=lambda storage, loc: storage))

			agent.noise_scale = NOISE_END
			