import numpy as np
import bluesky as bs 
from bluesky.tools.aero import ft, nm
MU0 = 0.9087861752210031
MU0 = np.deg2rad(MU0)
R = 6378137
f_inv = 298.257224
f = 1.0 / f_inv

class EcosystemAreas:

	def __init__(self, lat, lon, v, alpha_max, alt, time_horizon = 60, time_steps = 10):
		self.v = v
		self.lat_o = lat
		self.lon_o = lon
		self.alt_o = alt
		self.latitude = np.array([])
		self.longitude = np.array([])
		self.altitude = np.array([])
		self.x_enu = np.array([0])
		self.y_enu = np.array([0])
		self.z_enu = np.array([0])
		self.xecef = np.array([])
		self.yecef = np.array([])
		self.zecef = np.array([])
		self.time_horizon = time_horizon #in seconds
		self.time_steps = time_steps
		self.alpha_max = alpha_max

	def calculate(self):

		x_final_point = self.time_horizon * self.v

		for alpha in range(-self.alpha_max, self.alpha_max, 5):
			x = (x_final_point - self.x_enu[0]) * np.cos(np.deg2rad(alpha))
			y = (x_final_point - self.y_enu[0]) * np.sin(np.deg2rad(alpha))
			alpha_relative = alpha - np.arctan(y/x)
			self.x_enu = np.append(self.x_enu, [np.sqrt(x**2 + y**2) * np.cos(alpha_relative)])
			self.y_enu = np.append(self.y_enu, [np.sqrt(x**2 + y**2) * np.sin(alpha_relative)])

			lat, lon, alt = getLatLong((self.x_enu[-1], self.y_enu[-1], 0), (self.lat_o, self.lon_o), alpha, self.alt_o)
			self.latitude = np.append(self.latitude, [lat])
			self.longitude = np.append(self.longitude, [lon])
			self.altitude = np.append(self.altitude, [alt])

			xecef, yecef, zecef = lla2ecef((lat, lon, alt))
			self.xecef = np.append(self.xecef, xecef)
			self.yecef = np.append(self.yecef, yecef)
			self.zecef = np.append(self.zecef, zecef)
			
def getLatLong(xyz, lat_long_orig, rotation, href):

	psi = np.deg2rad(rotation)

	x, y, z = xyz
	lat_orig, lon_orig = lat_long_orig

	m_per_deg_lat = 111132.92 - 559.82 * np.cos(2*MU0) + 1.175 * np.cos(4*MU0) -0.0023 * np.cos(6*MU0)
	m_per_deg_long = 111412.84 * np.cos(MU0) -93.5* np.cos(3*MU0) + 0.118 * np.cos(5*MU0)
	
	N = np.cos(psi) * y - np.sin(psi) * x
	E = np.sin(psi) * y + np.cos(psi) * x

	Ndeg = N / m_per_deg_lat
	Edeg = E / m_per_deg_long

	lat = lat_orig + Ndeg
	lon = lon_orig + Edeg
	alt = href + z

	return lat, lon, alt

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

def calculate_tlosh(area1, area2, dlosh):
	for index in range(len(area1.xecef)):
		if index == len(area1.xecef)-1:
			e1 = np.array([area1.xecef[0], area1.yecef[0], area1.zecef[0]]) - np.array([area1.xecef[index], area1.yecef[index], area1.zecef[index]])
		else:
			e1 = np.array([area1.xecef[index+1], area1.yecef[index+1], area1.zecef[index+1]]) - np.array([area1.xecef[index], area1.yecef[index], area1.zecef[index]])
		r1 = np.array([area1.xecef[index], area1.yecef[index], area1.zecef[index]])
		for index2 in range(len(area2.xecef)):
			r2 = np.array([area2.xecef[index2], area2.yecef[index2], area2.zecef[index2]])
			if index2 == len(area2.xecef)-1:
				e2 = np.array([area2.xecef[0], area2.yecef[0], area2.zecef[0]]) - np.array([area2.xecef[index2], area2.yecef[index2], area2.zecef[index2]])
			else:
				e2 = np.array([area2.xecef[index2+1], area2.yecef[index2+1], area2.zecef[index2+1]]) - np.array([area2.xecef[index2], area2.yecef[index2], area2.zecef[index2]])

		d = distance_from_two_lines(e1, e2, r1, r2)
		if abs(d)<=dlosh:
			return True
	return False


def distance_from_two_lines(e1, e2, r1, r2):
	# e1, e2 = Direction vector
    # r1, r2 = Point where the line passes through

    # Find the unit vector perpendicular to both lines
    n = np.cross(e1, e2)
    n /= np.linalg.norm(n)

    # Calculate distance
    d = np.dot(n, r1 - r2)

    return d


			

