import bluesky as bs
import gym
class BlueSkyEnv(gym.env):
	def __init__(self):
		self.traffic = bs.Traffic()
		self.simulation = bs.Simulation()