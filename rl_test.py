from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import numpy as np
import cv2
import gym
import numpy as np
from WrappedEnv import RobotEnv

from stable_baselines3 import SAC

# env = gym.make("Pendulum-v1")

env = CogEnvDecoder(env_name="../mac_confrontation_v2/cog_confrontation_env.app", no_graphics=False, time_scale=1, worker_id=1) 
rl_env = RobotEnv("../mac_confrontation_v2/cog_confrontation_env.app")

model = SAC("Mlppolicy", rl_env, verbose=1)
model.learn(total_timesteps=10, log_interval=4)
model.save("sac_pendulum")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # env.render()
    if done:
      obs = env.reset()

# # env = CogEnvDecoder(env_name="mac_v2/cog_sim2real_env.app", no_graphics=False, time_scale=1, worker_id=1) 
# print(">>>", env._env.reward_range, "<<<")
# # env = CogEnvDecoder(env_name="win_V1/RealGame.exe", no_graphics=False, time_scale=1, worker_id=1) # windows os
# # env_name: path of the simulator
# # no_graphics: should use headless mode [Warning: if no_graphics is True, image if invalid!]
# # time_scale: useful for speedup collecting data during training, max value is 100
# # worker_id: socket port offset, useful for multi-thread training
# num_episodes = 10
# num_steps_per_episode = 500 # max: 1500
# for i in range(num_episodes):
#     #every time call the env.reset() will reset the envinronment
#     observation = env.reset()
#     
#     for j in range(num_steps_per_episode):
#         # action = env.action_space.sample()
#         action = [0, 0, 0.1, 1]  # [vx, vy, vw, fire]; vx: the velocity at which the vehicle moves forward, vy: the velocity at which the vehicle moves to the left, vw: Angular speed of the vehicle counterclockwise rotation, fire: Shoot or not
#         obs, reward, done, info = env.step(action)
#         cv2.imshow("color_image", obs["color_image"])
#         cv2.waitKey(1)
#         check_state(obs, info)
#         print("===", reward, "===")
