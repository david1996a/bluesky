from gym.envs.registration import register

register(
    id='BlueSkyEnv-v0',
    entry_point='blueskyenv.envs:BlueSkyEnv'
)