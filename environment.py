import bluesky as bs 
import numpy as np
from bluesky.tools.aero import nm

class BSEnv:
	def __init__(self, altitude_change = False):
		super(BSEnv, self).__init__()
		bs.init("sim")
		bs.net.connect()
		self.update_interval = 25	#[s] time between each timestep from the reinforcement learing model
		self.max_heading = 359 #[º] Max. relative degree heading. Used later to scale the state space.
		self.max_distance = bs.settings.asas_pzr * nm * 5 #[m] Maximum distance between planes that gets passed in the states
													  # Not sure if this has any effect.
		if altitude_change:
			self.n_actions = 7 #Taking changes of vertical speed as possible actions
		else:
			self.n_actions = 5 #Just horizontal heading changes

		self.action_space = np.arange(self.n_actions)	#Vector that will containt the possible options
		self.observation_space = np.array([])	#Vector that will contain the state space.

	def reset(self):
		"""
		TO DO: Rewrite this function so it is easier to restart the scenario. It takes too long since it has to
		restart all bluesky.
		"""
		bs.init("sim")
		bs.net.connect()
		return self.calculate_state()

	def calculate_state(self):
		"""
		Function that returns the state for all the agents of the environment. It retuns it in the form of a 
		numpy matrix where each row will be the state for an aircraft.
		"""
		states = np.zeros(shape(bs.traf.ntraf, self.observation_space))
		for ac in range(bs.traf.ntraf):
			state = np.array([], dtype=np.float32)
			
			#Distance and heading relative to the other planes.
			qdr, distinm = bs.tools.geo.qdrdist(bs.traf.lat, bs.traf.lon,
									np.ones(len(bs.traf.lat))*bs.traf.lat[ac], np.ones(len(bs.traf.lon))*bs.traf.lon[ac])
			dist = distinnm * nm
			planes = np.argsort(dist)[0:self.number_of_planes]

			for plane in planes:
				np.append(state, [dist[plane], qdr[plane]])

			#Distance and heading from the plane to the destination.
			heading, distance = bs.tools.geo.qdrdist(bs.traf.lat[ac], bs.traf.lon[ac], 
				bs.traf.ap.route[ac].wplat[-1], bs.traf.ap.route[ac].wplon[-1])

			np.append(state, [heading, diatance])
			states[ac,:] = state

		return states

	def reward(self):
		"""
		Function that returns the reard for each state
		"""
		pass

	def step(self):
		bs.sim.step()
		bs.net.step()
		print(bs.sim.simt)

	def take_actions(self, actions):
		"""
		Function that takes actions for all the planes.
		0 -> keep going straight
		1 -> heading change of +30º
		2 -> heading change of -30º
		3 -> heading change of +60º
		4 -> heading change of -60º
		5 -> vertical speed change +100m/s
		6 -> vertical speed change -100m/s
		7 -> vertical speed change +200m/s
		8 -> vertical speed change -200m/s
		"""
		for index, actions in actions:
			if action == 1:
				bs.traf.ap.selhdgcmd(0, bs.traf.hdg[index]+30)
			elif action == 2:
				bs.traf.ap.selhdgcmd(0, bs.traf.hdg[index]-30)
			elif action == 3:
				bs.traf.ap.selhdgcmd(0, bs.traf.hdg[index]+60)
			elif action == 4:
				bs.traf.ap.selhdgcmd(0, bs.traf.hdg[index]-60)
			elif action == 5:
				bs.traf.ap.selvspdcmd(0, bs.traf.vs[index]+100)
			elif action == 6:
				bs.traf.ap.selvspdcmd(0, bs.traf.vs[index]-100)
			elif action == 7:
				bs.traf.ap.selvspdcmd(0, bs.traf.vs[index]+200)
			elif action == 8:
				bs.traf.ap.selvspdcmd(0, bs.traf.vs[index]-200)

		bs.sim.step()
		bs.net.step()







