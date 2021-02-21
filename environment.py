import bluesky as bs 
import numpy as np
from bluesky.tools.aero import nm
import time
import os
from routes import Route
import math
import copy

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
MU0 = 0.9087861752210031
MU0 = np.deg2rad(MU0)
R = 6378137
f_inv = 298.257224
f = 1.0 / f_inv

class BSEnv:
	def __init__(self, altitude_change = False):
		super(BSEnv, self).__init__()
		bs.init("sim-detached")
		#bs.net.connect()
		
		self.initial_number_of_planes = bs.traf.ntraf
		self.update_interval = 25	#[s] time between each timestep from the reinforcement learing model
		self.max_heading = 359 #[º] Max. relative degree heading. Used later to scale the state space.
		self.max_distance = bs.settings.asas_pzr * nm * 5 #[m] Maximum distance between planes that gets passed in the states
													  # Not sure if this has any effect.
		"""
		if altitude_change:
			self.n_actions = 7 #Taking changes of vertical speed as possible actions
		else:
			self.n_actions = 5 #Just horizontal heading changes
		"""

		#self.action_space = np.arange(self.n_actions)	#Vector that will containt the possible options
		self.observation_space = np.array([])	#Vector that will contain the state space.
		self.number_of_planes = 5	#Number of surrounding planes that each agent takes into account when calculating individual states.
		self.time_step = 5	#[s] time between each timestep for the rl model.
		self.agents_id_idx = {bs.traf.id[i]: i for i in range(bs.traf.ntraf)}
		self.routes = {}
		for i in range(bs.traf.ntraf):
			self.routes[bs.traf.id[i]] = Route(np.append(bs.traf.lat[i],bs.traf.ap.route[i].wplat), 
			np.append(bs.traf.lon[i],bs.traf.ap.route[i].wplon),
			np.append(bs.traf.alt[i],bs.traf.ap.route[i].wpalt), bs.traf.id[i])

		self.traffic = None
		self.simulation = None 
		self.screen = None
		self.net = None

	def reset(self, episode):
		tic = time.time()
		print("resetting...")
		bs.sim.reset()
		"""
		if episode%5 == 0:
			bs.sim.reset()
			self.traffic = copy.deepcopy(bs.traf)
			self.simulation = copy.deepcopy(bs.sim)
			self.screen = copy.deepcopy(bs.scr)
			self.net = copy.deepcopy(bs.net)
		else:
			print(bs.sim.simt, self.simulation.simt)
			print(id(bs.sim), id(bs.sim.simt), id(self.simulation), id(self.simulation.simt))
			bs.traf = self.traffic
			bs.sim = self.simulation
			bs.scr = self.screen
			bs.net = self.net
			print(bs.sim.simt)
		"""
		self.agents_id_idx = {bs.traf.id[i]: i for i in range(bs.traf.ntraf)}
		self.routes = {}
		for i in range(bs.traf.ntraf):
			self.routes[bs.traf.id[i]] = Route(np.append(bs.traf.lat[i],bs.traf.ap.route[i].wplat), 
			np.append(bs.traf.lon[i],bs.traf.ap.route[i].wplon),
			np.append(bs.traf.alt[i],bs.traf.ap.route[i].wpalt), bs.traf.id[i])
		#bs.net.connect()
		states = self.calculate_states()
		state = self.calculate_general_state()
		tac = time.time()
		print("Resetting took ",(tac-tic)," seconds")
		return state, states

	def calculate_states(self):
		"""
		Function that returns the state for all the agents of the environment. It retuns it in the form of a 
		numpy matrix where each row will be the state for an aircraft.
		"""
		states = np.ones(shape=(self.initial_number_of_planes, 18), dtype=np.float32)*-999
		for acid in bs.traf.id:
			state = np.zeros(18, dtype=np.float32)
			index = 0
			ac = bs.traf.id2idx(acid)
			#Distance and heading relative to the other planes.
			qdr, distinm = bs.tools.geo.qdrdist(bs.traf.lat, bs.traf.lon,
									np.ones(len(bs.traf.lat))*bs.traf.lat[ac], np.ones(len(bs.traf.lon))*bs.traf.lon[ac])
			dist = distinm * nm
			planes = np.argsort(dist)[0:self.number_of_planes]

			for plane in planes:
				state[index:index+2] = [dist[plane],qdr[plane]]
				index+=2

			wplats = np.array([], dtype=np.float32)
			wplons = np.array([], dtype=np.float32)

			for i in range(min(3, len(self.routes[acid].wplat)-self.routes[acid].current_tram)):
				wplats = np.append(wplats, self.routes[acid].wplat[self.routes[acid].current_tram+i])
				wplons = np.append(wplons, self.routes[acid].wplon[self.routes[acid].current_tram+i])

			wplats = padding_vector(wplats, 3)
			wplons = padding_vector(wplons, 3)

			#Distance and heading from the plane to the destination.
			heading, distance = bs.tools.geo.qdrdist(bs.traf.lat[ac], bs.traf.lon[ac], 
				bs.traf.ap.route[ac].wplat[-1], bs.traf.ap.route[ac].wplon[-1])
	
			state[(len(state)-2):] = [heading, distance]
			states[self.agents_id_idx[acid],:] = state
		return states

	def calculate_general_state(self):
		"""
		Function that returns the global state in the form of a numpy array.
		"""
		wplats = np.ones((self.initial_number_of_planes*3), dtype=np.float32) * -999
		wplons = np.ones((self.initial_number_of_planes*3), dtype=np.float32) * -999
		lats = np.ones((self.initial_number_of_planes), dtype=np.float32) * -999
		lons = np.ones((self.initial_number_of_planes), dtype=np.float32) * -999
		cass = np.ones((self.initial_number_of_planes), dtype=np.float32) * -999
		alts = np.ones((self.initial_number_of_planes), dtype=np.float32) * -999
		for acid in bs.traf.id:
			index = bs.traf.id2idx(acid)
			#print(len(self.routes[acid].wplat), self.routes[acid].current_tram, len(self.routes[acid].wplat)-self.routes[acid].current_tram)
			for i in range(min(3, len(self.routes[acid].wplat)-self.routes[acid].current_tram)):
				wplats[self.agents_id_idx[acid]*3+i] = self.routes[acid].wplat[self.routes[acid].current_tram+i]
				wplons[self.agents_id_idx[acid]*3+i] = self.routes[acid].wplon[self.routes[acid].current_tram+i]
			lats[self.agents_id_idx[acid]] =	bs.traf.lat[index]
			lons[self.agents_id_idx[acid]] = bs.traf.lon[index]
			alts[self.agents_id_idx[acid]] = bs.traf.alt[index]
			cass[self.agents_id_idx[acid]] = bs.traf.cas[index]
		"""
		print("lats: ",lats)
		print("lons: ",lons)
		print("alts: ",alts)
		print("cass: ",cass)
		print("wplats: ",wplats)
		print("wplons: ",wplons)
		"""
		state = np.concatenate((lats, lons, alts, cass, wplats, wplons))
		return state

	def reward(self, w_time=1, w_fuel=1, w_conflicts=1, w_complexity=1):
		"""
		Function that returns the reward for each state
		"""
		dest_lat = np.empty([bs.traf.ntraf], dtype=np.float32)
		dest_lon = np.empty([bs.traf.ntraf], dtype=np.float32)

		distances = bs.tools.geo.latlondist(bs.traf.lat, bs.traf.lon,
			dest_lat, dest_lon)
		reward = 0
		for ac in range(bs.traf.ntraf):
			dest_idx = bs.navdb.getaptidx(bs.traf.ap.dest[ac])
			dest_lat[ac] = bs.navdb.aptlat[dest_idx]
			dest_lon[ac] = bs.navdb.aptlon[dest_idx]
			reward += self.reward_individual(ac, distances[ac]/1000)
			"""
			dest_lat = np.hstack((dest_lat, np.array([bs.navdb.aptlat[dest_idx]])))
			dest_lat = np.hstack((dest_lon, np.array([bs.navdb.aptlon[dest_idx]])))
			"""
		rewards = np.zeros(self.initial_number_of_planes, dtype=np.float32)
		indexs = [self.agents_id_idx[acid] for acid in bs.traf.id]
		rewards[indexs] = reward/bs.traf.ntraf
		return rewards

	def reward_individual(self, ac, distance, w_time=1, w_fuel=1, w_conflicts=1, w_distance=2, w_offtrack=1):
		reward_time = w_time
		reward_fuel = w_fuel*bs.traf.perf.fuelflow[ac]
		reward_conflicts = w_conflicts*sum(bs.traf.id[ac] in conf for conf in bs.traf.cd.confpairs_unique)
		reward_distance = w_distance*distance/(100+abs(distance)) if math.isnan(w_distance*distance/(100+abs(distance))) == False else 0


		distance_track = self.routes[bs.traf.id[ac]].distance_to_tram(self.routes[bs.traf.id[ac]].current_tram, 
			bs.traf.lat[ac], bs.traf.lon[ac], bs.traf.alt[ac])
		distance_off_track =  w_distance*distance_track/(2+abs(distance_track)) if math.isnan(2+abs(distance_track)) == False else 0
		off_heading = self.routes[bs.traf.id[ac]].calculate_heading_off(self.routes[bs.traf.id[ac]].current_tram, 
			bs.traf.hdg[ac])*2*np.pi/360
		reward_offtrack = w_offtrack*(np.cos(off_heading)+distance_off_track)
		"""
		print("Reward time: ", reward_time)
		print("Reward fuel: ", reward_fuel)
		print("Reward conflicts: ",reward_conflicts)
		print("Reward distance: ", reward_distance)
		print("Reward offtrack: ", reward_offtrack)
		"""
		rewards = [reward_time, reward_fuel, reward_conflicts, reward_distance, reward_offtrack]
		for index,reward in enumerate(rewards):
			if math.isnan(reward) == True:
				print(reward, index)
		return -reward_time-reward_fuel-reward_conflicts-reward_distance-reward_offtrack

	def step(self):
		for i in range(self.time_step):
			bs.sim.step()
			bs.net.step()
			done = self.check_episode_done()
			if done:
				break

		dones = np.ones(self.initial_number_of_planes, dtype=np.uint8)
		for acid, route in self.routes.items():
			idx = bs.traf.id2idx(acid)
			route.calculate_current_tram(bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.alt[idx])
			if acid not in bs.traf.id:
				dones[self.agents_id_idx[acid]] = 1

		state = self.calculate_general_state()
		states = self.calculate_states()
		#done = self.check_episode_done()
		reward = self.reward()
		return state, states, reward, dones, done


	def take_actions_continuous(self, actions):
		for acid in bs.traf.id:
			idx = bs.traf.id2idx(acid)
			bs.traf.ap.selhdgcmd(idx, bs.traf.hdg[idx]+actions[self.agents_id_idx[acid]])

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
		if bs.traf.ntraf < 5:#self.number_of_planes:
			print("Resetting because the number of planes is ",bs.traf.ntraf, bs.sim.simt)
			return True
		if bs.sim.simt > 12000:
			print("Resetting because the simulation time is ", bs.sim.simt)
			return True

		return False


def padding_vector(vector, length, padding_num=0, dtype=np.float32):
	result = np.zeros(shape=(length), dtype=dtype) + padding_num
	result[0:len(vector)] = vector
	return result

class Saved_scenario():
    def __init__(self):
        self.traffic = None
        self.simulation = None 
        self.screen = None
        self.rand_zone = None
        self.rand_numb_ac = None
        self.rand_routecompleted = None
        self.rand_wpcalculated = None
        self.rand_createconfict = None
        self.rand_nwconflat = None
        self.rand_nwconflon = None
        self.net = None

    def save_scenario(self):
        self.traffic = copy.deepcopy(bs.traf)
        self.simulation = copy.deepcopy(bs.sim)
        self.screen = copy.deepcopy(bs.scr)
        self.net = copy.deepcopy(bs.net)
        self.save_random_class()

    def save_random_class(self):
        self.rand_zone = copy.deepcopy(bs.random.zone)
        self.rand_numb_ac = copy.deepcopy(bs.random.numb_ac)
        self.rand_routecompleted = copy.deepcopy(bs.random.routecompleted)
        self.rand_wpcalculated = copy.deepcopy(bs.random.wpcalculated)
        self.rand_createconfict = copy.deepcopy(bs.random.createconflict)
        self.rand_nwconflat = copy.deepcopy(bs.random.nwconflat)
        self.rand_nwconflon = copy.deepcopy(bs.random.nwconflon)

    def load_random_class(self):
        bs.random.zone = copy.deepcopy(self.rand_zone)
        bs.random.numb_ac = copy.deepcopy(self.rand_numb_ac)
        bs.random.routecompleted = copy.deepcopy(self.rand_routecompleted)
        bs.random.wpcalculated = copy.deepcopy(self.rand_wpcalculated)
        bs.random.createconflict = copy.deepcopy(self.rand_createconfict)
        bs.random.nwconflat = copy.deepcopy(self.rand_nwconflat)
        bs.random.nwconflon = copy.deepcopy(self.rand_nwconflon)

    def load_scenario(self):
        bs.traf = copy.deepcopy(self.traffic)
        bs.sim = copy.deepcopy(self.simulation)
        bs.scr = copy.deepcopy(self.screen)
        bs.net = copy.deepcopy(self.net)
        self.load_random_class()

