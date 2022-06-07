import cv2
import math
import numpy as np
from modules.Robot3 import Robot
from modules import PathPlanner as PathPlanner
from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
from modules import extended_kalman_filter as ex

num_episodes = 10
num_steps_per_episode = int(5e8)
env = CogEnvDecoder(env_name="../mac_v2/cog_sim2real_env.app", no_graphics=False, time_scale=1, worker_id=1) 

robot = None
goal_prec = 0.5
np.random.seed(19260817)

bias_x = np.random.uniform(-0.5, 0.5, 1)[0]
bias_y = np.random.uniform(-0.5, 0.5, 1)[0]

bias_x = 0
bias_y = 0

map_size = [8.08, 4.48]

for i in range(num_episodes):

    obs = env.reset()
    robot = Robot(obs)

    # obs["vector"][0][0] += np.random.uniform(-0.1, 0.1, 1)[0] + bias_x
    # obs["vector"][0][1] += np.random.uniform(-0.1, 0.1, 1)[0] + bias_y

    # x, y = obs["vector"][0][0], obs["vector"][0][1]
    # obs["vector"][0][0] = min(max(x, 0), map_size[0])
    # obs["vector"][0][1] = min(max(y, 0), map_size[1])

    last_activation_tar = -1
    t1, t2 = 0.04, 0.04

    for j in range(num_steps_per_episode):
        activation_tar = robot.check_activation(obs)
        cu_x, cu_y = obs["vector"][0][0], obs["vector"][0][1]
        print(f'=== Current position: x={cu_x}, y={cu_y}')
        if activation_tar != -1:
            if activation_tar != last_activation_tar:
                robot.update_activation_path(obs, activation_tar)
                last_activation_tar = activation_tar
            action = robot.get_activation_action(cu_x, cu_y)
            action[2] = robot.get_activation_rotation(obs, activation_tar)
        else:
            action = [0, 0, 0, 0]
            break

        # Rotation matrix 
        theta = obs["vector"][0][2] + action[2] * 0.0001
        vx = action[0]
        vy = action[1]

        action[0] = math.cos(-theta) * vx - math.sin(-theta) * vy
        action[1] = math.sin(-theta) * vx + math.cos(-theta) * vy

        la_x, la_y = obs["vector"][0][0], obs["vector"][0][1]

        print(f"Next action: {action}")
        obs, reward, done, info = env.step(action)


        # if robot.tar_v_x != 0 and obs["vector"][0][0] != last_x:
        #     t1 = (obs["vector"][0][0] - last_x) / robot.tar_v_x
        # if robot.tar_v_y != 0 and obs["vector"][0][1] != last_y:
        #     t2 = (obs["vector"][0][1] - last_y) / robot.tar_v_y

        
        # cu_x, cu_y = obs["vector"][0][0], obs["vector"][0][1]
        # t1 = (cu_x - la_x) / vx
        # t2 = (cu_y - la_y) / vy
        # print(f"t1: {t1}     t2: {t2}")

        # obs["vector"][0][0] += np.random.uniform(-0.1, 0.1, 1)[0] + bias_x
        # obs["vector"][0][1] += np.random.uniform(-0.1, 0.1, 1)[0] + bias_y
        # x, y = obs["vector"][0][0], obs["vector"][0][1]

        # ideal_x = robot.cx[robot.target_idx]
        # ideal_y = robot.cy[robot.target_idx]
        # vt = robot.state.v
        # yaw = robot.state.yaw
        # xxx = obs["vector"][0][0]
        # yyy = obs["vector"][0][1]
        # xEst = ex.noise_reduce(xxx, yyy, yaw, vt, ideal_x, ideal_y)
        # obs["vector"][0][0] = xEst[0, 0]
        # obs["vector"][0][1] = xEst[1, 0]

        # x, y = obs["vector"][0][0], obs["vector"][0][1]
        # obs["vector"][0][0] = min(max(x, 0), map_size[0])
        # obs["vector"][0][1] = min(max(y, 0), map_size[1])
