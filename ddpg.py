from agentcritic import Actor, Critic
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np 

#Constants
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 30        # minibatch size
SAMPLE_SIZE = 20
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4 #3e-5 #1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4 #3e-5 #1e-4        # learning rate of the critic
WEIGHT_DECAY_actor = 0.0 #3e-4 #0        # L2 weight decay
WEIGHT_DECAY_critic = 0.0 #1e-6 #0        # L2 weight decay
#to decay exploration as it learns
NOISE_START=1.0
NOISE_END=0.1
NOISE_REDUCTION=0.999
EPISODES_BEFORE_TRAINING = 0
NUM_LEARN_STEPS_PER_ENV_STEP = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_ACTION = 60.0
MIN_ACTION = -60.0

class DDPG(object):

	def __init__(self, state_size, observation_size, action_size, num_agents):
		self.obs_size = observation_size
		self.action_size = action_size
		self.state_size = state_size
		self.noise_scale = NOISE_START
		"""Actor network"""
		self.actor_local = Actor(self.obs_size, self.action_size).to(DEVICE)
		self.actor_target = Actor(self.obs_size, self.action_size).to(DEVICE)
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY_actor)

		"""Critic network"""
		self.critic_local = Critic(self.state_size, num_agents*action_size).to(DEVICE)
		self.critic_target = Critic(self.state_size, num_agents*action_size).to(DEVICE)
		self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY_critic)

		# Make sure target is initialized with the same weight as the source (makes a big difference)
		self.hard_update(self.actor_target, self.actor_local)
		self.hard_update(self.critic_target, self.critic_local)

	def learn(self, experiences, gamma, agent_no, number_agents):
		state, observation, actions, reward, next_states, next_observation, next_actions_full, agent_dones = experiences
		
		"""Training the critic"""
		#next_actions = torch.tensor(next_actions_full.detach().cpu().data.numpy().reshape(
		#	SAMPLE_SIZE, number_agents)).to(DEVICE)

		next_actions = torch.zeros((SAMPLE_SIZE, number_agents), device=DEVICE)
		next_actions = torch.reshape(next_actions_full, (SAMPLE_SIZE, number_agents))
		#print(next_actions.device)
		Q_target_next = self.critic_target.forward(next_states, next_actions)
		Q_target = reward + gamma * Q_target_next * (1 - agent_dones)
		#next_actions2 = torch.tensor(actions.detach().cpu().data.numpy().reshape(SAMPLE_SIZE, number_agents)).to(DEVICE)
		next_actions = torch.reshape(actions, (SAMPLE_SIZE, number_agents))
		Q_expected = self.critic_local.forward(state, next_actions)
		critic_loss = F.mse_loss(input=Q_expected, target=Q_target)
		critic_loss.backward()
		self.critic_optimizer.zero_grad()

		"""Training the actor"""
		actions[:,agent_no,:] = self.actor_local.forward(observation[:,agent_no])
		#next_actions3 = torch.tensor(actions.detach().cpu().data.numpy().reshape(SAMPLE_SIZE, number_agents)).to(DEVICE)).mean()
		#next_actions3 = torch.reshape(actions, (SAMPLE_SIZE, number_agents), device=DEVICE)
		actor_loss = -self.critic_local.forward(state, next_actions).mean()
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

	def soft_update_all(self):
		self.soft_update(self.critic_local, self.critic_target, TAU)
		self.soft_update(self.actor_local, self.actor_target, TAU)

	def soft_update(self, model1, model2, tau):
		for target_param, local_param in zip(model2.parameters(), model1.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

	def hard_update(self, model1, model2):
		for target_param, local_param in zip(model2.parameters(), model1.parameters()):
			target_param.data.copy_(local_param.data)

	def act(self, observation, i_episode, add_noise=True):
		if i_episode > EPISODES_BEFORE_TRAINING and self.noise_scale > NOISE_END:
			self.noise_scale = NOISE_REDUCTION**(i_episode-EPISODES_BEFORE_TRAINING)

		if not add_noise:
			self.noise_scale = 0.0
		
		with torch.no_grad():
			action = self.actor_local.forward(observation).cpu().data.numpy()
		self.actor_local.train()
		
		action += self.noise_scale*self.add_noise()
		return np.clip(action, MIN_ACTION, MAX_ACTION)

	def add_noise(self):
		noise = 0.5*np.random.randn(1,self.action_size) #sigma of 0.5 as sigma of 1 will have alot of actions just clipped
		return np.squeeze(noise)







