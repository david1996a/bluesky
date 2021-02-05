import numpy as np
import gym
import bluesky as bs
from bluesky.network.client	import Client
from bluesky.tools.aero import nm
import numpy as np
from bluesky.ui import qtgl
import MapPlot as mapplt
from ecosystems import *

class BlueSkyEnv(gym.Env):

	def __init__(self):
		super(BlueSkyEnv, self).__init__()
		bs.init('sim')
		bs.net.connect()
		#bs.settings.set_variable_defaults(asas_pzr=5.0, asas_pzh=1000.0,
        #                          asas_dtlookahead=300.0)
		self.action_space = gym.spaces.Discrete(9)
		self.number_of_planes = 3
		self.update_interval = 15
		self.max_number_conflicts = 15 
		self.max_heading = 359
		self.max_speed = 300
		self.max_distance = bs.settings.asas_pzr * nm * 5
		self.acumulated_fuelflow = 0
		self.observation_space = gym.spaces.Box(low = np.zeros((self.number_of_planes * 4 + 3,1),dtype=np.float32), high =  np.ones((self.number_of_planes * 4 + 3,1),dtype=np.float32))
		print("End of __init__ of the environment")
		print(bs.traf.lat)

	def reset(self):
		bs.init("sim")
		bs.net.connect()
		state = self.calculate_state()
		return state

	def step(self, action):

		if bs.traf.ap.route[0].flag_landed_runway == True:
			episode_over = True
		else:
			episode_over = False

		for i in range(self.update_interval):
			#print(bs.traf.lat, i)
			bs.net.step()
			bs.sim.step()
			bs.scr.step()
			if i == 50:
				bs.traf.creconfs(str(bs.traf.ntraf+1), 'A388', 0, 100, 0, 0,0,0)
			if bs.traf.ap.route[0].flag_landed_runway == True:
				episode_over = True
				break
			else:
				episode_over = False

		while bs.traf.cd.inconf[0] == False:
			bs.net.step()
			bs.sim.step()
			bs.scr.step()
			#print(bs.traf.cd.inconf, bs.sim.simt)
			if bs.traf.ap.route[0].flag_landed_runway == True:
				episode_over = True
				break
			else:
				episode_over = False

		self.take_action(action)
		#print("taking action")
		#Selecting the 15 closest planes to the ac
		
		state = self.calculate_state()
		reward = self.evaluate_state(1,2,3,5,state)

		return state, reward, episode_over, {}

	def evaluate_state(self, a, b, c, d, state):
		self.acumulated_fuelflow = self.acumulated_fuelflow + state[2]*self.update_interval
		return -self.update_interval*a - state[0]*b - state[1]*self.update_interval*c - state[2]*d

	def take_action(self, action):
		"""
		if self.redirected == True:
			self.previous_track = bs.traf.trk[0]
			self.previous_heading = bs.traf.hdg[0]
		"""
		if action == 1:
			bs.traf.ap.selhdgcmd(0, bs.traf.hdg[0]+30)
		elif action == 2:
			bs.traf.ap.selhdgcmd(0, bs.traf.hdg[0]-30)
		elif action == 3:
			bs.traf.ap.selhdgcmd(0, bs.traf.hdg[0]+60)
		elif action == 4:
			bs.traf.ap.selhdgcmd(0, bs.traf.hdg[0]-60)
		elif action == 5:
			bs.traf.ap.selvspdcmd(0, bs.traf.vs[0]+100)
		elif action == 6:
			bs.traf.ap.selvspdcmd(0, bs.traf.vs[0]-100)
		elif action == 7:
			bs.traf.ap.selvspdcmd(0, bs.traf.vs[0]+200)
		elif action == 8:
			bs.traf.ap.selvspdcmd(0, bs.traf.vs[0]-200)
		bs.sim.step()
		bs.net.step()

	def render(self):
		pass

	def calculate_state(self):
		qdr, distinnm = bs.tools.geo.qdrdist(bs.traf.lat, bs.traf.lon,
									np.ones(len(bs.traf.lat))*bs.traf.lat[0], np.ones(len(bs.traf.lon))*bs.traf.lon[0])
		distinnm = distinnm * nm
		planes = np.argsort(distinnm)[0:self.number_of_planes]

		heading, distance = bs.tools.geo.qdrdist(bs.traf.lat[0], bs.traf.lon[0], bs.traf.ap.route[0].wplat[-1], bs.traf.ap.route[0].wplon[-1])
		state = np.array([],dtype=np.int64)
		number_conflicts = len(bs.traf.cd.confpairs_unique)
		state = np.append(state, [distance/(400 * nm), bs.traf.perf.fuelflow[0]/100, number_conflicts/self.max_number_conflicts])
		for ac in planes:
			if bs.traf.cd.inconf[ac] == True :
				inconflict = 1
			else:
				inconflict = 0
			relative_heading, distance = qdr[ac], distinnm[ac]
			distance = distance * nm
			state = np.append(state, [inconflict, relative_heading/self.max_heading, distance/self.max_distance, bs.traf.cas[ac]/self.max_speed])
		return state


