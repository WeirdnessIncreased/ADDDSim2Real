import cv2
import math
import numpy as np
from modules.Robot3 import Robot
from modules import PathPlanner as PathPlanner
from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder

num_episodes = 10
num_steps_per_episode = int(1e4)
env = CogEnvDecoder(env_name="../mac_v2/cog_sim2real_env.app", no_graphics=False, time_scale=1, worker_id=1) 

robot = None
goal_prec = 0.5
np.random.seed(19260817)
prex = np.random.uniform(-0.5, 0.5, 1)[0]
prey = np.random.uniform(-0.5, 0.5, 1)[0]
for i in range(num_episodes):
    obs = env.reset()
    # obs["vector"][0][0] += np.random.uniform(-0.1, 0.1, 1)[0] + prex
    # obs["vector"][0][1] += np.random.uniform(-0.1, 0.1, 1)[0] + prey
    robot = Robot(obs)

    last_activation_tar = -1
    t1, t2 = 0.04, 0.04
    for j in range(num_steps_per_episode):
        activation_tar = robot.check_activation(obs)
        cum_rotation = 0
        if activation_tar != -1:
            if activation_tar != last_activation_tar:
                robot.update_activation_path(obs, activation_tar)
                last_activation_tar = activation_tar
            if math.hypot(obs["vector"][0][0] - obs["vector"][5 + activation_tar][0], obs["vector"][0][1] - obs["vector"][5 + activation_tar][1]) > goal_prec:
                action = robot.get_activation_action(obs['vector'][0][0], obs['vector'][0][1])
            elif True:
                rotation = robot.get_activation_rotation(obs, activation_tar)
                action[2] = rotation
        else:
            pass

        # Rotation matrix 
        theta = obs["vector"][0][2]
        vx, vy = action[0], action[1]
        action[0] = math.cos(-theta) * vx - math.sin(-theta) * vy
        action[1] = math.sin(-theta) * vx + math.cos(-theta) * vy

        la_x, la_y = obs["vector"][0][0], obs["vector"][0][1]

        print(f"Next action: {action}")
        obs, reward, done, info = env.step(action)

        # if robot.tar_v_x != 0 and obs["vector"][0][0] != last_x:
        #     t1 = (obs["vector"][0][0] - last_x) / robot.tar_v_x
        # if robot.tar_v_y != 0 and obs["vector"][0][1] != last_y:
        #     t2 = (obs["vector"][0][1] - last_y) / robot.tar_v_y

        
        cu_x, cu_y = obs["vector"][0][0], obs["vector"][0][1]
        t1 = (cu_x - la_x) / vx
        t2 = (cu_y - la_y) / vy
        print(f"t1: {t1}     t2: {t2}")

        # obs["vector"][0][0] += np.random.uniform(-0.1, 0.1, 1)[0] + prex
        # obs["vector"][0][1] += np.random.uniform(-0.1, 0.1, 1)[0] + prey

        robot.update_state(obs)

        # cv2.imshow("color_image", obs["color_image"])
        # cv2.waitKey(1)
        # print(reward, done)
