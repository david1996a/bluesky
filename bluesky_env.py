import numpy as np
import gym
import bluesky as bs
from bluesky.network.client	import Client
import numpy as np

class BlueSkyEnv(gym.Env):

	def __init__(self):
		super(BlueSkyEnv, self).__init__()
		self.action_space = gym.spaces.Discrete(9)
        self.number_of_planes = 15
        self.update_interval = 60
        self.max_number_conflicts  = 15 
        self.max_heading = 359
        self.max_speed = 300
        self.max_distance = bs.settings.asas_pzr * nm * 5
		self.observation_space = gym.spaces.Tuple((gym.spaces.Box(low=[0, 0, 0], high=[400 * nm, 100, self.number_conflicts], shape(3,1), dtype = float32)), 
            gym.spaces.Box(low = [0, 0, 0], high = [1.0, self.max_heading, self.max_distance], shape(3,self.number_of_planes), dtype = float32))
        

	def reset(self):
		bs.sim.reset()

	def step(self, action):

        if bs.traf.route[0].flag_landed_runway == True:
            episode_over = True
        else:
            episode_over = False

        for i in range(self.update_interval):
            bs.sim.step()
            bs.net.step()
            if bs.traf.route[0].flag_landed_runway == True:
                episode_over = True
                break
            else:
                episode_over = False

        while bs.traf.cd.inconf[ac] == False:
            bs.sim.step()
            bs.net.step()
            if bs.traf.route[0].flag_landed_runway == True:
                episode_over = True
                break
            else:
                episode_over = False

		self.take_action(action)
        #Selecting the 15 closest planes to the ac
        qdr, distinnm = geo.qdrdist(bs.traf.lat, bs.traf.lon,
                                    np.ones(len(bs.traf.lat))*bs.traf.lat[0], np.ones(len(bs.traf.lon))*bs.traf.lon[0])
        distinnm = distinnm * nm
        planes = np.argsort(distinnm)[0:self.number_of_planes]

		heading, distance = qdrdist(bs.traf.lat[0], bs.traf.lon[0], bs.traf.ap.route[0].wplat[-1], bs.traf.ap.route[0].wplon[-1])
        state = np.array([],dtype=np.int64)
        number_conflicts = len(bs.traf.cd.confpairs_unique)
        state = np.append(self.state, [distance, bs.traf.perf.fuelflow[0], self.number_conflicts])
        for ac in planes:
            if bs.traf.cd.inconf[ac] == True :
                inconflict = 1
            else:
                inconflict = 0
            relative_heading, distance = qdr[ac], distinm[ac]
            distance = distance * nm
            state = np.append(self.state, [inconflict, relative_heading, distance, bs.traf.cas[ac]])
		observation = state

        

		reward = self.evaluate_state(1,2,3,60,state)

		return state, reward, episode_over, {}

	def evaluate_state(self, a, b, c, d, state):

        acumulated_fuelflow = acumulated_fuelflow + state[2]*update_interval
        return -update_interval*a - state[0]*b - state[1]*self.update_interval*c - state[2]*d 

    def take_action(self, action):
    	if self.redirected == True:
    		self.previous_track = bs.traf.trk[0]
    		self.previous_heading = bs.traf.hdg[0]
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
        print("rendering")


