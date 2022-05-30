from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import numpy as np
import cv2
import gym
import numpy as np
from WrappedEnv import RobotEnv

from stable_baselines3 import SAC

# env = gym.make("Pendulum-v1")

# env = CogEnvDecoder(env_name="../mac_confrontation_v2/cog_confrontation_env.app", no_graphics=False, time_scale=1, worker_id=1) 
rl_env = RobotEnv("../../mac_confrontation_v2/cog_confrontation_env.app")

model = SAC.load("sac_pendulum", env=rl_env, learning_rate=0.001)

# model = SAC("MlpPolicy", rl_env, verbose=1)
model.learn(total_timesteps=100000, log_interval=4, eval_log_path="./log")
model.save("sac_pendulum")

# train for 100000 steps
# train for 200000 steps
# train for 300000 steps

# del model # remove to demonstrate saving and loading
# 
# 
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     # env.render()
#     if done:
#         obs = env.reset()

