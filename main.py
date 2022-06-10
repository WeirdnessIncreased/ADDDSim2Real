import cv2
import math
import numpy as np
from modules.Robot import Robot
from modules import PathPlanner as PathPlanner
from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
from modules import extended_kalman_filter as ex
import matplotlib.pyplot as plt
import history.kalman as his_kal
from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
from modules.lidar_data_mapping import lidar_mapping
from modules.lidar_data_mapping import update as lidar_update

num_episodes = 20
num_steps_per_episode = int(5e8)
# env = CogEnvDecoder(env_name="../mac_confrontation_v2/cog_confrontation_env.app", no_graphics=False, time_scale=1, worker_id=1) 
env = CogEnvDecoder(env_name="../mac_v2/cog_sim2real_env.app", no_graphics=False, time_scale=1, worker_id=2) 

robot = None
goal_prec = 0.5
np.random.seed(19260817)

map_size = [8.08, 4.48]

##### history for noise reduction #####
x_perf = []
y_perf = []
x_esti = []
y_esti = []
x_real = []
y_real = []
x_fake = []
y_fake = []
##### history for noise reduction #####

activation_step_control = 0 # 是潜在未知 bug 的对应略策

for i in range(num_episodes):
    bias_x = np.random.uniform(-0.5, 0.5, 1)[0]
    bias_y = np.random.uniform(-0.5, 0.5, 1)[0]

    obs = env.reset()
    robot = Robot(obs)

    ##### noise #####
    noise_x = np.random.uniform(-0.1, 0.1, 1)[0]
    noise_y = np.random.uniform(-0.1, 0.1, 1)[0]
    obs["vector"][0][0] += noise_x + bias_x
    obs["vector"][0][1] += noise_y + bias_y
    ##### noise #####

    x, y = obs["vector"][0][0], obs["vector"][0][1]
    obs["vector"][0][0] = min(max(x, 0.05), map_size[0] - 0.05)
    obs["vector"][0][1] = min(max(y, 0.05), map_size[1] - 0.05)

    last_activation_tar = -1
    t1, t2 = 0.04, 0.04

    ex.clear(obs['vector'][0][0], obs['vector'][0][1])
    x_perf = []
    y_perf = []
    x_esti = []
    y_esti = []
    x_real = []
    y_real = []
    x_fake = []
    y_fake = []

    just_started_conf = True

    vector_data = obs['vector']
    dynamic_obstacles = [vector_data[5][:2], vector_data[6][:2], vector_data[7][:2], vector_data[8][:2], vector_data[9][:2]]
    lidar_update(dynamic_obstacles)

    laser_data = np.array(obs["laser"])
    debias_sum_x = 0
    debias_sum_y = 0
    debias_steps = 0
    laser_data = np.array(obs["laser"])
    x, y = lidar_mapping(obs['vector'], laser_data)
    if not (abs(obs['vector'][0][0] - x) > 0.6 or abs(obs['vector'][0][1] - y) > 0.6):
        debias_sum_x += obs['vector'][0][0] - x
        debias_sum_y += obs['vector'][0][1] - y
        debias_steps += 1

    la_fight_tar = np.array([None, None])
    fight_step_control = 0

    for j in range(num_steps_per_episode):

        if debias_steps != 0:
            obs["vector"][0][0] -= debias_sum_x / debias_steps
            obs["vector"][0][1] -= debias_sum_y / debias_steps

        activation_tar = robot.check_activation(obs)
        cu_x, cu_y = obs["vector"][0][0], obs["vector"][0][1]
        # print(f'=== Current position: x={cu_x}, y={cu_y}')
        if activation_tar != -1:
            if activation_tar != last_activation_tar or activation_step_control >= 128:
                activation_step_control = 0
                robot.update_activation_path(obs, activation_tar)
                last_activation_tar = activation_tar
            activation_step_control += 1
            action = robot.get_activation_action(cu_x, cu_y)
            action[2] = robot.get_activation_rotation(obs, activation_tar)

            laser_data = np.array(obs["laser"])
            x, y = lidar_mapping(obs['vector'], laser_data)
            if not (abs(obs['vector'][0][0] - x) > 0.6 or abs(obs['vector'][0][1] - y) > 0.6):
                debias_sum_x += obs['vector'][0][0] - x
                debias_sum_y += obs['vector'][0][1] - y
                debias_steps += 1
        else:
            if (la_fight_tar == robot.random_tar).all():
                fight_step_control += 1
            else:
                la_fight_tar = robot.random_tar
                fight_step_control = 1
            if fight_step_control >= 30:
                robot.random_tar = np.array([None, None])
            action = robot.get_fight_action(obs)

        # Rotation matrix 
        theta = obs["vector"][0][2] + action[2] * 0.0001
        vx = action[0]
        vy = action[1]

        action[0] = math.cos(-theta) * vx - math.sin(-theta) * vy
        action[1] = math.sin(-theta) * vx + math.cos(-theta) * vy

        la_x, la_y = obs["vector"][0][0], obs["vector"][0][1]

        # print(f"=== Next action: {action}")
        # print(f"=== Step control: {activation_step_control}")
        obs, reward, done, info = env.step(action)

        if done: break

        # if robot.tar_v_x != 0 and obs["vector"][0][0] != last_x:
        #     t1 = (obs["vector"][0][0] - last_x) / robot.tar_v_x
        # if robot.tar_v_y != 0 and obs["vector"][0][1] != last_y:
        #     t2 = (obs["vector"][0][1] - last_y) / robot.tar_v_y

        
        # cu_x, cu_y = obs["vector"][0][0], obs["vector"][0][1]
        # t1 = (cu_x - la_x) / vx
        # t2 = (cu_y - la_y) / vy
        # print(f"t1: {t1}     t2: {t2}")

        x, y = obs["vector"][0][0], obs["vector"][0][1]

        ##### noise #####
        noise_x = np.random.uniform(-0.1, 0.1, 1)[0]
        noise_y = np.random.uniform(-0.1, 0.1, 1)[0]
        obs["vector"][0][0] += noise_x + bias_x
        obs["vector"][0][1] += noise_y + bias_y
        ##### noise #####

        ##### kalman filter for localization #####
        xxx = obs['vector'][0][0]
        yyy = obs['vector'][0][1]
        yaw = obs['vector'][0][2]
        xEst = ex.noise_reduce(xxx, yyy, yaw, math.hypot(vx, vy), action[2])

        # ideal_x = robot.cx[robot.target_idx]
        # ideal_y = robot.cy[robot.target_idx]
        # vt = robot.state.v
        # yaw = robot.state.yaw
        # xEst = his_kal.noise_reduce(xxx, yyy, yaw, vt, ideal_x, ideal_y)

        obs["vector"][0][0] = xEst[0, 0]
        obs["vector"][0][1] = xEst[1, 0]

        # x_esti.append(xEst[0, 0])
        # y_esti.append(xEst[1, 0])
        # x_real.append(x)
        # y_real.append(y)
        # x_fake.append(xxx)
        # y_fake.append(yyy)

        x, y = obs["vector"][0][0], obs["vector"][0][1]
        obs["vector"][0][0] = min(max(x, 0.05), map_size[0] - 0.05)
        obs["vector"][0][1] = min(max(y, 0.05), map_size[1] - 0.05)

    
        print(info)

        # x_perf.append((xxx-x)/noise_x)  
        # y_perf.append((yyy-y)/noise_y)  
        ##### kalman filter for localization #####

        # print(f'=== Noise: x={noise_x} y={noise_y} === Fix: x={x-xxx}, y={y-yyy} === performance: x={(xxx-x)/noise_x} y={(yyy-y)/noise_y}')

        ##### check rotation #####
        # plt.clf()
        # w = obs['vector'][0][2]
        # x1, y1 = obs['vector'][0][0], obs['vector'][0][1]
        # x2, y2 = obs['vector'][3][0], obs['vector'][3][1]
        # plt.plot([x1], [y1], 'xb')
        # plt.plot([x2], [y2], 'xr')
        # plt.plot([x1, x2], [y1, y2], '-', color='0.9')
        # plt.plot([0, 0, 8, 8], [0, 4, 0, 4], 'x', color='0')
        # b = y1 - np.tan(w) * x1
        # xx = np.arange(x1-2, x1+2, 0.3)
        # yy = xx * np.tan(w) + b
        # plt.plot(xx, yy, '-g')
        # plt.pause(0.001)
        # plt.show(block=False)
        ##### check rotation #####

    # plt.plot(x_real, y_real, "-r", label="real")
    # plt.plot(x_fake, y_fake, ".b", label="fake")
    # plt.plot(x_esti, y_esti, "-g", label="estimation")
    # plt.title(f'average performance: x={np.mean(x_perf)}, y={np.mean(y_perf)}')
    # plt.grid(True)
    # plt.show(block=False)
    # plt.pause(2)

