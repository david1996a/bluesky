""" External control plugin for Machine Learning applications. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, sim, traf  #, settings, navdb, traf, sim, scr, tools
import bluesky as bs
from bluesky.tools.areafilter import Circle, Box
import random
from bluesky.tools.geo import *
import numpy as np
from bluesky.traffic.asas import *
import itertools
from ecosystems import *
from bluesky.tools.aero import ft, nm
from bluesky.tools import geo, areafilter, plotter
myclientrte = None
import numpy as np
from environment import padding_vector 
from agentcritic import *
import torch as T 
from routes import Route

MIN_ACTION = -60.0
MAX_ACTION = 60.0

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():

    # Addtional initilisation code
    global actor, routes, number_of_planes
    actor = Actor(18, 1)
    load_model(actor)
    routes = {bs.traf.id[i]: Route(np.append(bs.traf.lat[i],bs.traf.ap.route[i].wplat), 
            np.append(bs.traf.lon[i],bs.traf.ap.route[i].wplon),
            np.append(bs.traf.alt[i],bs.traf.ap.route[i].wpalt), bs.traf.id[i]) for i in range(bs.traf.ntraf)}
    number_of_planes = 5
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'TEST_MODELS',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 5.0,

        'update':          update,

        'preupdate':        preupdate,

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

def preupdate():
    pass

def update():
    states = np.array([], dtype=np.float32)
    for ac in range(bs.traf.ntraf):
        routes[bs.traf.id[ac]].calculate_current_tram(bs.traf.lat[ac], bs.traf.lon[ac], bs.traf.alt[ac])
        observation = calculate_state(ac)
        observation = T.tensor(observation)
        action = actor.forward(observation)
        print(action)
        action = np.clip(action.detach().numpy(), MIN_ACTION, MAX_ACTION)
        #print(action)
        bs.traf.ap.selhdgcmd(ac, bs.traf.hdg[ac]+action)


def calculate_state(ac):
    state = np.zeros(18, dtype=np.float32)
    index = 0
    acid = bs.traf.id[ac]
    #Distance and heading relative to the other planes.
    qdr, distinm = bs.tools.geo.qdrdist(bs.traf.lat, bs.traf.lon,
                            np.ones(len(bs.traf.lat))*bs.traf.lat[ac], np.ones(len(bs.traf.lon))*bs.traf.lon[ac])
    dist = distinm * nm
    planes = np.argsort(dist)[0:number_of_planes]

    for plane in planes:
        state[index:index+2] = [dist[plane],qdr[plane]]
        index+=2

    wplats = np.array([], dtype=np.float32)
    wplons = np.array([], dtype=np.float32)

    for i in range(min(3, len(routes[acid].wplat)-routes[acid].current_tram)):
        wplats = np.append(wplats, routes[acid].wplat[routes[acid].current_tram+i])
        wplons = np.append(wplons, routes[acid].wplon[routes[acid].current_tram+i])

    wplats = padding_vector(wplats, 3)
    wplons = padding_vector(wplons, 3)

            #Distance and heading from the plane to the destination.
    heading, distance = bs.tools.geo.qdrdist(bs.traf.lat[ac], bs.traf.lon[ac], 
        bs.traf.ap.route[ac].wplat[-1], bs.traf.ap.route[ac].wplon[-1])
    
    state[(len(state)-2):] = [heading, distance]
    return state

def load_model(actor):
    actor.load_state_dict(T.load('models/checkpoint_actor_local_1'
                +'.pth',map_location=lambda storage, loc: storage))

def reset():
    pass


def mlstep():
    pass