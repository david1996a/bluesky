class DDPG(object):

	def __init__(self, state_size, action_size, num_agents):
		self.state_size = state_size
		self.action_size = action_size

		#Actor network
		self.actor_local = Actor(state_size, action_size).to(DEVICE)
		self.actor_target = Actor(state_size, action_size).to(DEVICE)
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY_actor)

		#Critic network
		self.critic_local = Critic(num_agents*state_size, num_agents*action_size).to(DEVICE)
        self.critic_target = Critic(num_agents*state_size, num_agents*action_size).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY_critic)

        # Make sure target is initialized with the same weight as the source (makes a big difference)
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)