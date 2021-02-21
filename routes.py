import bluesky as bs 
import numpy as np
MU0 = 0.9087861752210031
MU0 = np.deg2rad(MU0)
R = 6378137
f_inv = 298.257224
f = 1.0 / f_inv

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
		desired_heading, _ = bs.tools.geo.qdrdist(self.wplat[tram], self.wplon[tram],
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