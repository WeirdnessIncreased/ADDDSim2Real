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
# from modules.lidar_data_mapping import lidar_mapping
# from modules.lidar_data_mapping import update as lidar_update
import modules.lidar_data_mapping as lidar
lidar_mapping = lidar.lidar_mapping
lidar_update  = lidar.update

np.random.seed(19260817)

goal_prec = 0.5

map_size = [8.08, 4.48]

activation_step_control = 0 

debias_sum_x = 0
debias_sum_y = 0
debias_steps = 0

last_activation_tar = -1

la_fight_tar = None
fight_step_control = 0

action = [1, 1, -0.1, 0]

class Agent:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.cleaned = False
        self.robot = None
        self.la_v_x = 0
        self.la_v_y = 0
        self.la_w = 0
        self.his = []
        self.emergency = False

    def clear(self, obs):
        ex.clear(obs['vector'][0][0], obs['vector'][0][1])

        self.robot = Robot(obs)
        self.la_v_x = 0
        self.la_v_y = 0
        self.la_w = 0
        self.his = []
        self.emergency = False

        vector_data = obs['vector']
        dynamic_obstacles = [vector_data[5][:2], vector_data[6][:2], vector_data[7][:2], vector_data[8][:2], vector_data[9][:2]]
        lidar_update(dynamic_obstacles)

        global debias_sum_x
        global debias_sum_y
        global debias_steps
        global last_activation_tar
        global la_fight_tar
        global fight_step_control
        global activation_step_control
        global action

        debias_sum_x = 0
        debias_sum_y = 0
        debias_steps = 0
        last_activation_tar = -1
        la_fight_tar = None
        fight_step_control = 0
        activation_step_control = 0
        action = [1, 1, -0.1, 0]

    
    def agent_control(self, obs, done, info):

        global debias_sum_x
        global debias_sum_y
        global debias_steps
        global last_activation_tar
        global la_fight_tar
        global fight_step_control
        global activation_step_control

        ###### please don't stuck there ######
        self.his.append((obs['vector'][0][0], obs['vector'][0][1]))
        if len(self.his) >= 10:
            self.his = self.his[-10:]
            mean_x = np.array(self.his)[:, 0].mean()
            mean_y = np.array(self.his)[:, 1].mean()
            rmse = np.mean([math.hypot(x - mean_x, y - mean_y) for x, y in self.his])
            # print('rmse', rmse)
            # print('his', self.his)
            if rmse <= 0.03:
                if not self.emergency:
                    self.emergency = True
                    global action
                    action = [-x for x in action]
            else:
                self.emergency = False
        ###### please don't stuck there ######

        ########## reset each epoch ##########
        if obs['vector'][5][2] == False and self.cleaned == False:
            self.clear(obs)
            self.cleaned = True
        elif obs['vector'][5][2] == True:
            self.cleaned = False
        ########## reset each epoch ##########

        ########## noise reduction ##########
        xxx = obs['vector'][0][0]
        yyy = obs['vector'][0][1]
        yaw = obs['vector'][0][2]
        xEst = ex.noise_reduce(xxx, yyy, yaw, math.hypot(self.la_v_x,self.la_v_y), self.la_w)

        obs["vector"][0][0] = xEst[0, 0]
        obs["vector"][0][1] = xEst[1, 0]

        x, y = obs["vector"][0][0], obs["vector"][0][1]
        obs["vector"][0][0] = min(max(x, 0.05), map_size[0] - 0.05)
        obs["vector"][0][1] = min(max(y, 0.05), map_size[1] - 0.05)

        laser_data = np.array(obs["laser"])
        try:
            x, y, check = lidar_mapping(obs['vector'], laser_data)
            if not (abs(obs['vector'][0][0] - x) > 0.6 or abs(obs['vector'][0][1] - y) > 0.6):
                if( check ):
                    debias_sum_x += obs['vector'][0][0] - x
                    debias_sum_y += obs['vector'][0][1] - y
                    debias_steps += 1
        except:
            pass

        if debias_steps != 0:
            obs["vector"][0][0] -= debias_sum_x / debias_steps
            obs["vector"][0][1] -= debias_sum_y / debias_steps

        print('fixed coordinate', obs['vector'][0])
        print('lidar\'s status_x & status_y', lidar.status_x, lidar.status_y)
        ########## noise reduction ##########
        
        ############# get action #############
        # global action

        activation_tar = self.robot.check_activation(obs)
        cu_x, cu_y = obs["vector"][0][0], obs["vector"][0][1]
        if self.emergency:
            action = action
            last_activation_tar = -1
            self.robot.random_tar = None
        elif activation_tar != -1:
            # print(activation_step_control)
            if activation_tar != last_activation_tar or activation_step_control >= 100:
                activation_step_control = 0
                self.robot.update_activation_path(obs, activation_tar)
                last_activation_tar = activation_tar
            activation_step_control += 1
            action = self.robot.get_activation_action(cu_x, cu_y)
            action[2] = self.robot.get_activation_rotation(obs, activation_tar)
        else:
            # print(fight_step_control)
            if la_fight_tar == self.robot.random_tar:
                fight_step_control += 1
            else:
                la_fight_tar = self.robot.random_tar
                fight_step_control = 1
            if fight_step_control >= 50:
                self.robot.random_tar = None
                fight_step_control = 0
            action = self.robot.get_fight_action(obs)

        theta = obs["vector"][0][2] + action[2] * 0.0001
        vx = action[0]
        vy = action[1]

        action[0] = math.cos(-theta) * vx - math.sin(-theta) * vy
        action[1] = math.sin(-theta) * vx + math.cos(-theta) * vy
        ############# get action #############

        ############ remember me #############
        self.la_v_x = action[0]
        self.la_v_y = action[1]
        self.la_w = action[2]
        ############ remember me #############

        return action

