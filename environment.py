import bluesky as bs 
import numpy as np
from bluesky.tools.aero import nm
import time
import os
from vectors import *
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

		self.routes = {}
		for i in range(bs.traf.ntraf):
			self.routes[bs.traf.id[i]] = Route(np.append(bs.traf.lat[i],bs.traf.ap.route[i].wplat), 
			np.append(bs.traf.lon[i],bs.traf.ap.route[i].wplon),
			np.append(bs.traf.alt[i],bs.traf.ap.route[i].wpalt), bs.traf.id[i])

	def reset(self):
		"""
		TO DO: Rewrite this function so it is easier to restart the scenario. It takes too long since it has to
		restart all bluesky.
		"""
		tic = time.time()
		print("resetting...")
		bs.sim.reset()
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
		states = np.zeros(shape=(self.initial_number_of_planes, 18), dtype=np.float32)
		for ac in range(bs.traf.ntraf):
			state = np.zeros(18, dtype=np.float32)
			index = 0
			acid = bs.traf.id[ac]
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
			states[ac,:] = state

		return states

	def calculate_general_state(self):
		"""
		Function that returns the global state in the form of a numpy array.
		"""
		wplats = np.array([], dtype=np.float32)
		wplons = np.array([], dtype=np.float32)
		for ac in range(bs.traf.ntraf):
			acid = bs.traf.id[ac]
			for i in range(min(3, len(self.routes[acid].wplat)-self.routes[acid].current_tram)):
				wplats = np.append(wplats, self.routes[acid].wplat[self.routes[acid].current_tram+i])
				wplons = np.append(wplons, self.routes[acid].wplon[self.routes[acid].current_tram+i])

		wplats = padding_vector(wplats, 3*bs.traf.ntraf)
		wplons = padding_vector(wplons, 3*bs.traf.ntraf)

		state = np.zeros(self.initial_number_of_planes*4+self.initial_number_of_planes*6, dtype=np.float32)
		state[0:bs.traf.ntraf] = bs.traf.lat
		state[self.initial_number_of_planes:(self.initial_number_of_planes+bs.traf.ntraf)] = bs.traf.lon
		state[2*self.initial_number_of_planes:(2*self.initial_number_of_planes+bs.traf.ntraf)] = bs.traf.alt
		state[3*self.initial_number_of_planes:(3*self.initial_number_of_planes+bs.traf.ntraf)] = bs.traf.cas
		state[4*self.initial_number_of_planes:(4*self.initial_number_of_planes+bs.traf.ntraf*6)] = np.append(wplats, wplons)
		return state

	def reward(self, w_time=1, w_fuel=1, w_conflicts=1, w_complexity=1):
		"""
		Function that returns the reward for each state
		"""
		dest_lat = np.empty([bs.traf.ntraf], dtype=np.float32)
		dest_lon = np.empty([bs.traf.ntraf], dtype=np.float32)

		distances = bs.tools.geo.latlondist(bs.traf.lat, bs.traf.lon,
			dest_lat, dest_lon)

		for ac in range(bs.traf.ntraf):
			dest_idx = bs.navdb.getaptidx(bs.traf.ap.dest[ac])
			dest_lat[ac] = bs.navdb.aptlat[dest_idx]
			dest_lon[ac] = bs.navdb.aptlon[dest_idx]
			"""
			dest_lat = np.hstack((dest_lat, np.array([bs.navdb.aptlat[dest_idx]])))
			dest_lat = np.hstack((dest_lon, np.array([bs.navdb.aptlon[dest_idx]])))
			"""

		return -w_time*bs.traf.ntraf - w_fuel*1 - w_conflicts*1 

	def reward_individual(self, ac, distance, w_time=1, w_fuel=1, w_conflicts=1, w_distance=2, w_offtrack=1):
		reward_time = w_time
		reward_fuel = w_fuel*bs.traf.perf.fuelflow[ac]
		reward_conflicts = w_conflicts*bs.traf.cd.confpairs_unique.count(bs.traf.id[ac])
		reward_distance = w_distance*distance/(100+distance)

		dsitance_track = self.routes[bs.traf.ac[ac]].distance_to_tram(self.routes[bs.traf.ac[ac]].current_tram, 
			bs.traf.lat[ac], bs.traf.lon[ac], bs.traf.alt[ac])
		distance_off_track =  w_distance*distance_track/(2+distance_track)
		off_heading = self.routes[bs.traf.ac[ac]].calculate_heading_off(self.routes[bs.traf.ac[ac]].current_tram, 
			bs.traf.hdg[ac]) *2*np.pi
		reward_offtrack = w_offtrack*(np.cos(off_heading)+distance_off_track)

		return -reward_time-reward_fuel-rewar_conflicts-reward_distance-reward_offtrack

	def step(self):
		for i in range(self.time_step):
			bs.sim.step()
			bs.net.step()
			done = self.check_episode_done()
			if done:
				break
		
		for acid, route in self.routes.items():
			idx = bs.traf.id2idx(acid)
			route.calculate_current_tram(bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.alt[idx])

		state = self.calculate_general_state()
		states = self.calculate_states()
		done = self.check_episode_done()
		reward = self.reward()
		return state, states, reward, done


	def take_actions_continuous(self, actions):
		for index in range(bs.traf.ntraf):
			bs.traf.ap.selhdgcmd(index, bs.traf.hdg[index]+actions[index])

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
			return True
		if bs.sim.simt > 7200:
			return True

		return False

class Route:
	def __init__(self, wplats, wplons, wpalts, acid):
		self.wplat = wplats
		self.wplon = wplons
		self.wpalt = wpalts
		self.number_of_trams = len(self.wplat)-1
		self.current_tram = 0
		self.acid = acid

	def calculate_current_tram(self, lat, lon, alt):
		for tram in range(self.number_of_trams):
			distance_to_tram = self.distance_to_tram(tram, lat, lon, alt)
			distance_to_next_wp = bs.tools.geo.latlondist(lat, lon, self.wplat[tram+1], self.wplon[tram+1])

			if distance_to_tram < distance_to_next_wp:
				self.current_tram = tram
				return tram, distance_to_tram


	def calculate_heading_off(self, tram, heading):
		desired_heading, _ = qdrdist(self.wplat[tram], self.wplon[tram],
			self.wplat[tram+1], self.wplon[tram+1])
		return desired_heading - heading

	def distance_to_tram(self, tram, lat, lon, alt):
		x1, y1, z1 = lla2ecef((self.wplat[tram], self.wplon[tram], self.wpalt[tram]))
		x2, y2, z2 = lla2ecef((self.wplat[tram+1], self.wplon[tram+1], self.wpalt[tram+1]))
		xp, yp, zp = lla2ecef((lat, lon, alt))

		A = np.array([x1, y1, z1])
		B = np.array([x2, y2, z2])
		P = np.array([xp, yp, zp])

		distance = distance_numpy(A, B, P)/1000
		return distance
		#print(alt, self.wpalt[tram], self.wpalt[tram+1])
		#print(lat, lon, alt)
		"""
		line_vector = (x2-x1, y2-y1, z2-z1)
		point_vector = (xp, yp, zp)

		print(line_vector, point_vector)
		line_len = length(line_vector)
		line_unit_vector = unit(line_vector)
		ponit_vector_scaled = scale(point_vector, 1.0/line_len)
		t = dot(line_unit_vector, ponit_vector_scaled)

		if t < 0.0:
			t = 0.0
		elif t > 1.0:
			t = 1.0

		nearest = scale(line_vector, t)
		dist = distance(nearest, point_vector)
		#print(dist)
		return dist/1000
		"""

from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm
import math

# from: https://gist.github.com/nim65s/5e9902cd67f094ce65b0
def distance_numpy(A, B, P):
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(A == B):
    	return math.dist(A, P)
    if all(A == P) or all(B == P):
        return 0
    if arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > pi / 2:
        return norm(P - A)
    if arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2:
        return norm(P - B)
    return norm(cross(A-B, A-P))/norm(B-A)

def lla2ecef(lla):
	latitude, longitude, altitude = lla

	coslat = np.cos(latitude * np.pi / 180)
	sinlat = np.sin(latitude * np.pi / 180)

	coslon = np.cos(longitude * np.pi / 180)
	sinlon = np.sin(longitude * np.pi / 180)

	c = 1 / np.sqrt(coslat * coslat + (1 - f) * (1 - f) * sinlat * sinlat)
	s = (1 - f) * (1 - f) * c

	x = (R*c + altitude) * coslat * coslon
	y = (R*c + altitude) * coslat * sinlon
	z = (R*s + altitude) * sinlat
	return x, y, z


def padding_vector(vector, length, padding_num=0, dtype=np.float32):
	result = np.zeros(shape=(length), dtype=dtype) + padding_num
	result[0:len(vector)] = vector
	return result



