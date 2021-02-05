import bluesky as bs 
import numpy as np

class BSEnv:
	def __init__(self):
		super(BSEnv, self).__init__()
		bs.init("sim")
		bs.net.connect()
		self.update_interval = 25	#[s] time between each timestep from the reinforcement learing model
		self.max_ehading = 359 #[º] Max. relative degree heading. Used later to scale the state space.
		self.max_distance = bs.settings.asas * nm * 5 #[m] Maximum distance between planes that gets passed in the states
													  # Not sure if this has any effect.
		if altitude_change:
			self.n_actions = 7 #Taking changes of vertical speed as possible actions
		else:
			self.n_actions = 5 #Just horizontal heading changes

		self.action_space = np.arange(self.n_actions)	#Vector that will containt the possible options
		self.observation_space = np.array([])	#Vector that will contain the state space.

	def reset(self):
		#TO DO: Rewrite this function so it is easier to restart the scenario. It takes too long since it has to
		#restart all bluesky.
		bs.init("sim")
		bs.net.connect()
		return self.calculate_state()

	def calculate_state(self):
		for ac in range(bs.traf.ntraf):
			qdr, distinm = bs.tools.geo.qdrdist(bs.traf.lat, bs.traf.lon,
									np.ones(len(bs.traf.lat))*bs.traf.lat[ac], np.ones(len(bs.traf.lon))*bs.traf.lon[ac])
			dist = distinnm * nm
			planes = np.argsort(dist)[0:self.number_of_planes]

			heading, distance = bs.tools.geo.qdrdist(bs.traf.lat[ac], bs.traf.lon[ac], 
				bs.traf.ap.route[ac].wplat[-1], bs.traf.ap.route[ac].wplon[-1])
			state = np.array([], dtype=np.float32)

		return state

	def reward(self):




