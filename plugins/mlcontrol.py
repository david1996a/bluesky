""" External control plugin for Machine Learning applications. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, sim, traf #, settings, navdb, traf, sim, scr, tools
import bluesky as bs
import copy
import numpy as np
from bluesky.tools.geo import *
import random

myclientrte = None

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():

    # Addtional initilisation code
    global scenario, state, agent
    state = State()
    scenario = Saved_scenario()
    agent = Agent()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'MLCONTROL',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': state.update_interval,

        'update':          update,
        'preupdate':       preupdate,

        # If your plugin has a state, you will probably need a reset function to
        # clear the state in between simulations.
        'reset':         reset
        }

    stackfunctions = {
        # The command name for your function
        'MLSTEP': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'MLSTEP',

            # A list of the argument types your function accepts. For a description of this, see ...
            '',

            # The name of your function in this plugin
            mlstep,

            # a longer help text of your function.
            'Simulate one MLCONTROL time interval.']
    }

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin


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

class State():
    def __init__(self):
        self.state = None
        self.latitude = None
        self.longitude = None
        self.altitude = None
        self.speed =  None
        self.vspeed = None
        self.number_conflicts = None
        self.heading = None
        self.acumulated_fuelflow = 0
        self.update_interval = 25
        self.w = None


    def update(self):
        """
        if any(bs.random.cdetect.inconf):
            print('Confpairs:   ', bs.random.cdetect.confpairs_unique, '\n')
            print('Lospairs:    ', bs.random.cdetect.lospairs_unique, '\n')
            print('Inconf:   ', bs.random.cdetect.inconf, '\n')
            print('Tcpamax:    ', bs.random.cdetect.tcpamax, '\n')
            print('Qdr:   ', bs.random.cdetect.qdr, '\n')
            print('Dist:    ', bs.random.cdetect.dist, '\n')
            print('Dcpa:   ', bs.random.cdetect.dcpa, '\n')
            print('Tcpa:    ', bs.random.cdetect.tcpa, '\n')
            print('TLOS:    ', bs.random.cdetect.tLOS, '\n')
        """
        heading, distance = qdrdist(bs.traf.lat[0], bs.traf.lon[0], bs.traf.ap.route[0].wplat[-1], bs.traf.ap.route[0].wplon[-1])
        self.state = np.array([],dtype=np.int64)
        self.number_conflicts = len(bs.traf.cd.confpairs_unique)
        self.state = np.append(self.state, [distance, bs.traf.perf.fuelflow[0], self.number_conflicts])
        for ac in range(1, bs.traf.ntraf):
            if bs.traf.cd.inconf[ac] == True :
                inconflict = 1
            else:
                inconflict = 0
            relative_heading, distance = qdrdist(bs.traf.lat[0], bs.traf.lon[0], bs.traf.lat[ac], bs.traf.lon[ac])
            self.state = np.append(self.state, [inconflict, relative_heading, distance, bs.traf.cas[ac]])
        if self.w is None:
            self.w = np.ones(len(self.state)+1, dtype=np.int64)*-20.0
        
        print(self.w)


    def evaluate_state(self, a, b, c, d):

        self.acumulated_fuelflow = self.acumulated_fuelflow + self.state[2]*self.update_interval
        return -self.update_interval*a - self.state[0]*b - self.state[1]*self.update_interval*c - self.state[2]*d 

        
class Agent():
    def __init__(self):
        self.previous_track = None
        self.previous_heading = None
        self.redirected = True
        self.action_list = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.choosing_action = False
        self.action = 0
        self.action_values = np.array([-125.0,-100.0,-100.0,-100.0,-100.0,-100.0,-100.0,-100.0,-100.0])
        self.current_state_value = None
        self.previous_state_value = None
        self.gamma = 0.6
        self.epsilon = 0.1

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
        self.redirected = False

    def redirect(self):

        bs.traf.ap.selhedg(0, self.previous_heading)
        self.redirected = True

        if abs(bs.traf.trk[0] - self.previous_track) < 5:
            bs.traf.swlnav[0] = True
            bs.traf.swvnav[0] = True
            bs.traf.swvnavspd[0] = True
        
def update():
   pass

def preupdate():
    state.update()
    if agent.action == 0:
        scenario.save_scenario()
        agent.current_state_value =  state.w.dot(np.append(state.state, np.argmax(agent.action_values)))
    if agent.choosing_action == True:
        if agent.action != 0:
            agent.action_values[agent.action-1] = state.w.dot(np.append(state.state, agent.action-1))
        scenario.load_scenario()
        agent.take_action(agent.action)
        agent.action += 1
        if agent.action == 10:
            agent.choosing_action = False

    else:
        if random.randint(0,100)<agent.epsilon:
            agent.take_action(random.randint(0,8))
        else:
            agent.take_action(np.argmax(agent.action_values))
        agent.choosing_action = True
        agent.action = 0
        agent.previous_state_value = agent.current_state_value
        reward = state.evaluate_state(1,1,1,150)
        state.w = gradient_descent(agent.previous_state_value + reward, state.w, state.state, 0.1 ,np.argmax(agent.action_values))

def reset():
    pass

def mlstep():
    global myclientrte
    myclientrte = stack.routetosender()
    sim.op()

def gradient_descent(g_t, w, state, step, action):
    current_value = w.dot(np.append(state, action))
    w = w + step * (g_t - current_value)*(np.append(state, action))
    return w

def SARSA_step():
    action_list = np.numpy([0,1,2,3,4,5,6,7,8])
    action = SARSA_choose_action()

    while espisode:
        agent.take_action(action)
