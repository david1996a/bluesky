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

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():

    # Addtional initilisation code
    global random_scenario
    random_scenario = RandomScenario()
    random_scenario.start()
    bs.random = random_scenario

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'SCENARIO',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 1.0,

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
    random_scenario.update()

def update():
    pass

def reset():
    pass

def mlstep():
    pass

class RandomScenario:
    def __init__(self):
        self.zone = Circle('zone', (random.uniform(50,40), random.uniform(0,15), 100))
        self.numb_ac = 20
        self.ap_inside = bs.navdb.getapinside(40, 50, 0, 15)
        self.routecompleted = np.full((self.numb_ac), False)
        self.wpcalculated = np.full((self.numb_ac), False)
        self.createconflict = 1
        self.nwconflat = 0
        self.nwconflon = 0

    def start(self):
        for ac in range(self.numb_ac):
            self.createplane(ac)
        bs.scr.showroute(bs.traf.id[0])
        bs.sim.ffmode = True
        

    def getnextwp(self, lat, lon, idx, criteria = 'd', range_look = 1):
        dest = bs.traf.ap.dest[idx]
        dest_idx = bs.navdb.getaptidx(dest)
        dest_lat = bs.navdb.aptlat[dest_idx]
        dest_lon = bs.navdb.aptlon[dest_idx]
        #We select waypoints surrounding the aircraft.
        list_next_wp = bs.navdb.getwpinside(lat-range_look , lat+range_look , lon-range_look , lon+range_look)
        min_value = np.inf
        weight1 = 10
        for waypoint in list_next_wp:
            qdrwp, dwp = qdrdist(lat, lon, bs.navdb.wplat[waypoint], bs.navdb.wplon[waypoint])
            qdrdest, ddest = qdrdist(bs.navdb.wplat[waypoint], bs.navdb.wplon[waypoint], dest_lat, dest_lon)
            value = (abs(qdrwp - qdrdest)/360.0*weight1 + (ddest - dwp)/ddest)
            if min_value>value:
                final_idx = waypoint
                min_value = value
        name = bs.navdb.wpid[final_idx]
        wptype = bs.navdb.wptype[final_idx]
        lat2 =  bs.navdb.wplat[final_idx]
        lon2 = bs.navdb.wplon[final_idx]
        bs.traf.ap.route[idx].addwpt(idx, name, 0, lat2, lon2)
        return final_idx

    def update(self):
        """
        if random.randint(0,10000)< self.createconflict and bs.sim.simt > 5:
            print("creating conflict")
            bs.traf.creconfs(str(bs.traf.ntraf+1), actype = 'A388', 0, 100, 20, 600)
        
        matrix = np.zeros(shape=(bs.traf.ntraf, bs.traf.ntraf), dtype=bool)
        areas = []
        combinations = list(itertools.combinations(range(bs.traf.ntraf),2))
        #bs.scr.objdel()
        for ac in range(bs.traf.ntraf):
            areas.append(EcosystemAreas(lat = bs.traf.lat[ac], lon = bs.traf.lon[ac], v = bs.traf.cas[ac], alpha_max = 60,alt=bs.traf.alt[ac]))
            #bs.scr.objappend('POLY', 'object'+str(ac), np.dstack((areas[-1].latitude, areas[-1].longitude)).flatten())

            areafilter.defineArea('test', "POLY", np.dstack((areas[-1].latitude, areas[-1].longitude)).flatten())
            areas[-1].calculate()

        for combination in combinations:
            result = calculate_tlosh(areas[combination[0]], areas[combination[1]], bs.settings.asas_pzr * nm)
            matrix[combination[0], combination[1]] = result
            matrix[combination[1], combination[0]] = result
        """
        for ac in range(bs.traf.ntraf):
            dest = bs.traf.ap.dest[ac]
            dest_idx = bs.navdb.getaptidx(dest)
            dest_lat = bs.navdb.aptlat[dest_idx]
            dest_lon = bs.navdb.aptlon[dest_idx]
            qdrwp, dwp = qdrdist(bs.traf.lat[ac], bs.traf.lon[ac], dest_lat, dest_lon)
            if dwp*nm<100:
                bs.traf.delete(ac)
                break
            if self.zone.checkInside(bs.traf.lat[ac], bs.traf.lon[ac], bs.traf.alt[ac]) == False:
                bs.traf.delete(ac)
                break
                

        self.fix_index_bug()
        
        

    def addrwy(self, idx):
        rwykeys = bs.navdb.rwythresholds[bs.traf.ap.dest[idx]]
        rwykeys = list(rwykeys.keys())
        rwyid = str(random.choice(rwykeys))
        name = bs.traf.ap.dest[idx]+"RWYx"+rwyid+"000"
        lat = bs.navdb.rwythresholds[bs.traf.ap.dest[idx]][rwyid][0]
        lon = bs.navdb.rwythresholds[bs.traf.ap.dest[idx]][rwyid][1]
        bs.traf.ap.route[idx].addwpt(idx, name, 5, lat, lon)

    def createplane(self, ac):
        #Generate a random distance and a heading vector. With that, we can generate a randome lat, lon point using qdrpos geo function.
        rand_distance = random.randint(0, self.zone.r)
        heading_vector = random.randint(0,360)
        lat, lon = qdrpos(latd1 = self.zone.clat, lond1 = self.zone.clon, qdr = heading_vector, dist = rand_distance)
        #runwayerror = True
        list_next_wp = self.ap_inside
        waypoint = random.choice(list_next_wp)
        dest_lat = bs.navdb.aptlat[waypoint]
        dest_lon = bs.navdb.aptlon[waypoint]
        dest_id = bs.navdb.aptid[waypoint]
        qdr, d = qdrdist(lat, lon, dest_lat, dest_lon)
        #Airport this aircraft will be heading.
        """
        while runwayerror == True:
            print("looking 4 an airport...")
            airport_destination_idx = random.choice(self.ap_inside)
            try:
                airport_destination_id = bs.navdb.aptid[airport_destination_idx]
                if bs.navdb.rwythresholds[airport_destination_id] == {}:
                    runwayerror = True
                else:
                    runwayerror = False
            except:
                pass
        
        airport_destination_lat = bs.navdb.aptlat[airport_destination_idx]
        airport_destination_lon = bs.navdb.aptlon[airport_destination_idx]
        airport_destination_name = bs.navdb.aptname[airport_destination_idx]
        qdr, d = qdrdist(lat, lon, airport_destination_lat, airport_destination_lon)
        """
        #Creating the plane.
        acid = 'AC'+str(ac)
        bs.traf.cre(acid = acid, actype = 'A388', aclat=lat, aclon = lon, achdg = qdr, acalt=10000, acspd=200)
        bs.traf.ap.dest[ac] = dest_id
        bs.traf.ap.setdestorig("DEST", ac, dest_id)

        #The first waypoint is added.
        #waypoint = self.getnextwp(lat, lon ,ac)
        self.calc_route(ac)
        name = bs.traf.ap.route[ac].wpname[0]
        #The plane is directed to that route. The point is set to an acivewaypoint.
        bs.traf.ap.route[ac].direct(ac, name)

    def nextwp(self, idx, dist2wp, dist2dest):
        """
            for idx in [i for i in range(len(dist2wp)) if dist2wp[i] < 25000 and self.wpcalculated[i] == True and self.routecompleted[i] == False] :
                bs.wpcalculated[idx] = False
            for idx in [i for i in range(len(dist2wp)) if dist2wp[i] < 18000 and self.wpcalculated[i] == False and self.routecompleted[i] == False] :
                self.getnextwp(idx, bs.traf, range = 2)
                self.wpcalculated[idx] = True

            # FMS route update and possibly waypoint shift. Note: qdr, dist2wp will be updated accordingly in case of wp switch
            for idx in [i for i in range(len(dist2dest)) if dist2dest[i] < 100000 and self.routecompleted[i] == False]:
                self.addrwy(idx)
                self.routecompleted[idx] = True
        """
        pass
    def calc_route(self, idx):
        dest = bs.traf.ap.dest[idx]
        dest_idx = bs.navdb.getaptidx(dest)
        dest_lat = bs.navdb.aptlat[dest_idx]
        dest_lon = bs.navdb.aptlon[dest_idx]
        lat = bs.traf.lat[idx]
        lon = bs.traf.lon[idx]
        distance = 100000
        route_calculation_on = True
        rlook=1
        counter = 0
        #While the plane is not close enough to the destination
        while route_calculation_on:
            counter += 1.5
            qdrwp, dwp = qdrdist(lat, lon, dest_lat, dest_lon)
            current_distance = dwp*nm
            if current_distance<distance:
                route_calculation_on = False
                break
            wp_idx = self.getnextwp(lat, lon, idx, range_look = rlook)
            lat = bs.navdb.wplat[wp_idx]
            lon = bs.navdb.wplon[wp_idx]
            if counter>10:
                rlook += 0.2
                distance -= 1000

        #self.addrwy(idx)

    def fix_index_bug(self):
        for ac in range(bs.traf.ntraf):
            bs.traf.ap.route[ac].iac = ac

def reset():
    random_scenario.zone = Circle('zone', (random.uniform(50,40), random.uniform(0,15), 400))
    random_scenario.numb_ac = 20
    random_scenario.routecompleted = np.full((random_scenario.numb_ac), False)
    random_scenario.wpcalculated = np.full((random_scenario.numb_ac), False)
    random_scenario.createconflict = 1
    random_scenario.nwconflat = 0
    random_scenario.nwconflon = 0
    random_scenario.start()


