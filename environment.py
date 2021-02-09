import bluesky as bs 
import numpy as np
from bluesky.tools.aero import nm
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class BSEnv:
	def __init__(self, altitude_change = False):
		super(BSEnv, self).__init__()
		bs.init("sim")
		bs.net.connect()
		self.initial_number_of_planes = 25
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
		self.number_of_planes = 5	#Number of surrounding planes that each agent takes into account when calculating individual states.
		self.time_step = 10	#[s] time between each timestep for the rl model.

	def reset(self):
		"""
		TO DO: Rewrite this function so it is easier to restart the scenario. It takes too long since it has to
		restart all bluesky.
		"""
		bs.sim.reset()
		bs.net.connect()
		states = self.calculate_states()
		state = self.calculate_general_state()
		return state, states

	def calculate_states(self):
		"""
		Function that returns the state for all the agents of the environment. It retuns it in the form of a 
		numpy matrix where each row will be the state for an aircraft.
		"""
		states = np.zeros(shape=(self.initial_number_of_planes, 12), dtype=np.float32)
		for ac in range(bs.traf.ntraf):
			state = np.zeros(12, dtype=np.float32)
			index = 0
			#Distance and heading relative to the other planes.
			qdr, distinm = bs.tools.geo.qdrdist(bs.traf.lat, bs.traf.lon,
									np.ones(len(bs.traf.lat))*bs.traf.lat[ac], np.ones(len(bs.traf.lon))*bs.traf.lon[ac])
			dist = distinm * nm
			planes = np.argsort(dist)[0:self.number_of_planes]

			for plane in planes:
				state[index:index+2] = [dist[plane],qdr[plane]]
				#state = np.append(state, [dist[plane], qdr[plane]])
				index+=2

			#Distance and heading from the plane to the destination.
			heading, distance = bs.tools.geo.qdrdist(bs.traf.lat[ac], bs.traf.lon[ac], 
				bs.traf.ap.route[ac].wplat[-1], bs.traf.ap.route[ac].wplon[-1])
			state[index:] = [heading, distance]
			#state = np.append(state, [heading, distance])
			states[ac,:] = state

		return states

	def calculate_general_state(self):
		"""
		Function that returns the global state in the form of a numpy array.
		"""
		state = np.zeros(self.initial_number_of_planes*4, dtype=np.float32)
		state[0:bs.traf.ntraf] = bs.traf.lat
		state[self.initial_number_of_planes:(self.initial_number_of_planes+bs.traf.ntraf)] = bs.traf.lon
		state[2*self.initial_number_of_planes:(2*self.initial_number_of_planes+bs.traf.ntraf)] = bs.traf.alt
		state[3*self.initial_number_of_planes:(3*self.initial_number_of_planes+bs.traf.ntraf)] = bs.traf.cas		
		return np.concatenate((bs.traf.lat, bs.traf.lon, bs.traf.alt, bs.traf.cas))
		
	def reward(self, w_time=1, w_fuel=1, w_conflicts=1, w_complexity=1):
		"""
		Function that returns the reward for each state
		"""
		return -w_time*self.time_step - w_fuel*1 - w_conflicts*1 - w_complexity*1

	def step(self):
		bs.sim.step()
		bs.net.step()
		
		state = self.calculate_general_state()
		states = self.calculate_states()
		done = self.check_episode_done()
		reward = self.reward()
		return state, states, reward, done


	def take_actions_continuous(self, actions):
		for index, action in enumerate(actions):
			bs.traf.ap.selhdgcmd(index, bs.traf.hdg[index]+action)

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

	def check_episode_done(self):
		if bs.traf.ntraf < self.number_of_planes:
			return True
		if bs.sim.simt > 250:
			return True

		return False







