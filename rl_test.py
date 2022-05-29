from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import numpy as np
import cv2
import gym
import numpy as np
from WrappedEnv import RobotEnv
import itertools

from stable_baselines3 import SAC

# env = gym.make("Pendulum-v1")

env = CogEnvDecoder(env_name="../mac_confrontation_v2/cog_confrontation_env.app", no_graphics=False, time_scale=1, worker_id=1) 
# rl_env = RobotEnv("../mac_confrontation_v2/cog_confrontation_env.app")

# model = SAC("MlpPolicy", rl_env, verbose=1)
# model.learn(total_timesteps=1000, log_interval=4)
# model.save("sac_pendulum")
# 
# del model # remove to demonstrate saving and loading
 
model = SAC.load("sac_pendulum")
 
obs_ = env.reset()

while True:
    obs = np.zeros(61 + 28)
    vec_state = obs_['vector']
    vec_state[2] = [vec_state[2]]
    obs[:61] = obs_['laser']
    obs[61:] = np.array(list(itertools.chain.from_iterable(vec_state)))

    action, _states = model.predict(obs, deterministic=True)
    obs_, reward, done, info = env.step(action)

    if done:
        obs = env.reset()

