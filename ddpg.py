class DDPG(object):

	def __init__(self, state_size, action_size, num_agents):
		self.state_size = state_size
		self.action_size = action_size

		#Actor network
		self.actor_local = Actor(state_size, action_size).to(DEVICE)
		self.actor_target = Actor(state_size, action_size).to(DEVICE)
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY_actor)

		#Critic network
		