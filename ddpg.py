from agentcritic import Agent, Critic
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np 

class DDPG(object):

	def __init__(self, state_size, observations_size, action_size, num_agents):
		self.obs_size = observation_size
		self.action_size = action_size
		self.state_size = state_size
		self.noise_scale = NOISE_START
		"""Actor network"""
		self.actor_local = Actor(obs_size, action_size).to(DEVICE)
		self.actor_target = Actor(obs_size, action_size).to(DEVICE)
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY_actor)

		"""Critic network"""
		self.critic_local = Critic(self.state_size, num_agents*action_size).to(DEVICE)
        self.critic_target = Critic(self.state_size, num_agents*action_size).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY_critic)

        # Make sure target is initialized with the same weight as the source (makes a big difference)
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

    def learn(self, experiences, gamma):
    	state, observation, actions, reward, next_states, next_observation, next_actions_full = experiences
    	actions_next = self.actor_local.forward(next_observations)

    	"""Training the critic"""
    	Q_target_next = self.critic_target.forward(next_states, actions_next)
    	Q_target = agent_rewards + gamma * Q_target_next * (1 - agent_dones)
    	Q_expected = self.critic_local.forward(state, actions)
    	critic_loss = F.mse_loss(input=Q_expected, target=Q_target)
    	critic_loss.backward()
    	self.critic_optimizer.zero_grad()

    	"""Training the actor"""
    	actor_loss = -self.critic_local(states, self.actor_local.forward(states)).mean()
    	self.actor_optimizer.zero_grad()
    	actor.loss.backward()
    	self.actor_optimizer.step()

    def soft_update_all(self):
    	self.soft_update(self.critic_local, self.critic_target, TAU)
    	self.soft_update(self.actor_local, self.actor_target, TAU)


    def soft_update(model1, model2, TAU):
    	for target_param, local_param in zip(model2.parameters(), model1.parameters()):
    		target_param.datacopy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(model1, model2):
    	for target_param, local_param in zip(model2.parameters(), model1.parameters()):
    		target_param.data.copy_(local_param.data)

    def act(self, observation, i_episode, add_noise=True):
    	if i_episode > EPISODE_BEFORE_TRAINING and self.noise_scale > NOISE_END:
    		self.noise_scale = NOISE_REDUCTION**(i_episode-EPISODES_BEFORE_TRAINING)

    	if not add_noise:
    		self.noise_scale = 0.0

    	with torch.no_grad():
    		action = self.actor_local.forward(observation).numpy()
    	action += self.noise_scale*self.add_noise()
    	return action

    def add_noise(self):
        noise = 0.5*np.random.randn(1,self.action_size) #sigma of 0.5 as sigma of 1 will have alot of actions just clipped
        return noise







