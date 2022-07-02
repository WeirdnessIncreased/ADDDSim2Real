import cv2
import math
import numpy as np
from Robot1 import Robot
import PathPlanner as PathPlanner
from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder

num_episodes = 10
num_steps_per_episode = int(1e4)
env = CogEnvDecoder(env_name="../mac_v2/cog_sim2real_env.app", no_graphics=False, time_scale=1, worker_id=1) 

robot = None
goal_prec = 0.50 # [m]

for i in range(num_episodes):
    obs = env.reset()
    robot = Robot(obs)

    last_activation_tar = -1

    for j in range(num_steps_per_episode):

        print(f"=== Current state: x={obs['vector'][0][0]} y={obs['vector'][0][1]} theta={obs['vector'][0][2]}")

        activation_tar = robot.check_activation(obs)
        action = [0, 0, 0, 0]
        if activation_tar != -1:
            if activation_tar != last_activation_tar:
                robot.update_activation_path(obs, activation_tar)
                last_activation_tar = activation_tar
            if math.hypot(obs["vector"][0][0] - obs["vector"][5 + activation_tar][0], obs["vector"][0][1] - obs["vector"][5 + activation_tar][1]) > goal_prec:
                action = robot.get_activation_action()
            rotation = robot.get_activation_rotation(obs, activation_tar)
            action[2] = rotation
        else:
            pass

        # Rotation matrix (yin wei che de chao xiang wen ti
        theta = obs["vector"][0][2]
        vx, vy = action[0], action[1]
        action[0] = math.cos(-theta) * vx - math.sin(-theta) * vy
        action[1] = math.sin(-theta) * vx + math.cos(-theta) * vy

        print(f"Next action: {action}")
        action[0] *= 1
        action[1] *= 1
        obs, reward, done, info = env.step(action)

        robot.update_state(obs)

        # cv2.imshow("color_image", obs["color_image"])
        # cv2.waitKey(1)
        # print(reward, done)
