import bluesky as bs 
from environment import BSEnv
import time
from maddpg import MADDPG
import torch
from collections import namedtuple, deque
import numpy as np
import time
import profile

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The device is: ",DEVICE)

env = BSEnv()
resets = 0
state_size = bs.traf.ntraf*4+6*bs.traf.ntraf
observation_size = 18
action_size = 1
n_agents = 20
n_episodes = 3
buffer_samples = 200
maddpg = MADDPG(state_size, observation_size, action_size, n_agents, buffer_samples)
scores_deque = deque(maxlen=100)
scores_list = []
scores_list_100_avg = []

def main():
	for i in range(n_episodes):
		tic_e = time.time()
		state, observations = env.reset()
		scores = 0
		num_steps = 0
		done = False
		i_i = 0
		#print("Starting episode... ",i)
		while done == False:
			actions = maddpg.act(torch.from_numpy(observations).to(DEVICE), i)
			env.take_actions_continuous(actions)
			next_state, next_observations, reward, done = env.step()
			maddpg.step(i, state, observations, actions, reward, next_state, next_observations, done)
			scores += reward
			state = next_state
			num_steps += 1
			observations = next_observations
			i_i += 1
			
			if i_i % 20 == 0:
				print("Step: 	",i_i)
			
		scores_deque.append(scores)
		scores_list.append(scores)
		scores_list_100_avg.append(np.mean(scores_deque))
		print('Episode {}\tAverage Score: {:.2f}\tCurrent Score: {}'.format(i, np.mean(scores_deque), np.max(scores)))
		print('Noise Scaling: {}, Memory size: {} and Num Steps: {}'.format(maddpg.maddpg_agents[0].noise_scale, 
			len(maddpg.memory.state_memory), num_steps))

		if i % 50 == 0:
			maddpg.save_maddpg()
			print("Saved Model. Episode {}\tAverage Score: {:.2f}".format(i, np.mean(scores_deque)))
		tac_e = time.time()
		print("L'episodi ",i," ha tardat ",(tac_e-tic_e)," segons.")

profile.run('main()')