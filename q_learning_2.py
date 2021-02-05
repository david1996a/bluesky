import gym
from sklearn.linear_model import SGDRegressor
import bluesky as bs
import numpy as np
#from bluesky_env import *
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing

if "../" not in sys.path:
  sys.path.append("../") 

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

env = gym.envs.make('BlueSkyEnv:BlueSkyEnv-v0')
#matplotlib.style.use('ggplot')
#env = gym.envs.make("MountainCar-v0")
print(env.observation_space.sample() for x in range(10))
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
observation_examples = np.reshape(observation_examples, (10000, env.number_of_planes * 4 + 3))
print(observation_examples.shape)
scaler = sklearn.preprocessing.StandardScaler()
print(scaler)
scaler.fit(observation_examples)
print("before featurizer")
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))

class FunctionEstimator():

	def __init__(self):
		self.models = []
		for _ in range(env.action_space.n):
			model = SGDRegressor(learning_rate = "constant")
			model.partial_fit([self.featurize_state(env.reset())], [0])
			self.models.append(model)

	def featurize_state(self, state):
		scaled = scaler.transform([state])
		featurized = featurizer.transform(scaled)
		return featurized[0]

	def predict(self, s, a=None):
		features = self.featurize_state(s)
		print(features)
		if not a:
			return np.array([m.predict([features])[0] for m in self.models])
		else:
			return self.models[a].predict([features])[0]

	def update(self, s, a, y):
		features = self.featurize.state(s)
		self.models[a].partial_fit([features], [y])


def epsilon_greedy_policy(estimator, epsilon, nA):
	def policy_fn(observation):
		A = np.ones(nA, dtype=float) * epsilon / nA
		q_values = estimator.predict(observation)
		print(q_values)
		best_action = np.argmax(q_values)
		A[best_action] += 1.0 - epsilon
		return A
	return policy_fn

class Episode_Stats():
	def __init__(self, ep_lengths, ep_rewards):
		self.episode_lengths = ep_lengths
		self.episode_rewards = ep_rewards

def q_learning(env, estimator, num_episodes, discount_factor = 1.0, epsilon = 0.1, epsilon_decay = 1.0):
	
	print("inside q_learning")
	stats = Episode_Stats(
		ep_lengths = np.zeros(num_episodes),
		ep_rewards = np.zeros(num_episodes))
	
	for i_episode in range(num_episodes):
		print("Episode:	", i_episode)
		policy = epsilon_greedy_policy(
			estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)

		last_reward = stats.episode_rewards[i_episode -1]
		sys.stdout.flush()

		state = env.reset()

		next_action = None

		for t in itertools.count():
			if next_action is None:
				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			next_state, reward, done, _ = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			q_values_next = estimator.predict(next_state)
			td_tarfet = reward + discount_factor * np.max(q_values_next)

			estimator.update(state, action, td_target)

			if done:
				break

			state = next_state

	return stats
print("before estimator")
estimator = FunctionEstimator()
print("before q_learning")
q_learning(env, estimator, 2)

