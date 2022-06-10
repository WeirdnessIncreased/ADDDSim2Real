import math
from modules import Controller3, PathPlanner
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import os
from modules import cubic_spline_planner

############ Points for Random Walk ##############

critical_points = pickle.load(open('./pathlet/critical_points', 'rb'))

################ Obstacle & Map ##################

ox, oy = [], [] # store solid fixed obstacles and map borders
conf_ox, conf_oy = [], []
map_size = [8.08, 4.48] # [m]
obstacle_prec = 0.02 # [m]

fixed_obstacles = {
    'B1': [ 7.08, 1.00, 8.08, 1.2],
    'B2': [ 5.78, 2.14 ,6.58 , 2.34], 
    'B3': [ 6.58, 3.48, 6.78, 4.48],
    'B4': [ 3.54, 0.935, 4.45 , 1.135], 
    'B5': [ 3.864, 2.064, 4.216, 2.416], 
    'B6': [ 3.54, 3.345, 4.45 , 3.545],
    'B7': [ 1.5, 0, 1.7, 1],
    'B8': [ 1.5, 2.14, 2.3, 2.34], 
    'B9': [ 0, 3.28, 1, 3.48]
}

for name in fixed_obstacles:
    pos = fixed_obstacles[name]
    for x in np.arange(pos[0], pos[2], obstacle_prec):
        for y in np.arange(pos[1], pos[3], obstacle_prec):
            ox.append(x)
            oy.append(y)
            if name not in ['B2', 'B5', 'B8']:
                conf_ox.append(x)
                conf_oy.append(y)

for x in np.arange(0, map_size[0], obstacle_prec):
    ox.append(x)
    oy.append(0)
    ox.append(x)
    oy.append(map_size[1])

for y in np.arange(0, map_size[1], obstacle_prec):
    ox.append(0)
    oy.append(y)
    ox.append(map_size[0])
    oy.append(y)


################# Hyperparameters #################

k = 5  # control gain
Kp = 0.03  # speed proportional gain
dt = 0.04  # [s] time difference
L = 0.18  # [m] Wheel base of vehicle
max_steer = math.pi / 6  # [rad] max steering angle

show_animation = False

show_speed = False
show_pos = False
show_spline = False
print_path = False

target_speed = 2  #[m/s]

################# The Robot Agent #################

class Robot:
    def __init__(self, obs):
        global critical_points
        self.ox = ox[:] # obstacle x for path planning
        self.oy = oy[:] # obstacle y for path planning
        self.conf_ox = conf_ox[:]
        self.conf_oy = conf_oy[:]
        self.cx = None
        self.cy = None
        self.cyaw = None
        self.target_idx = None
        self.state = None
        self.tar_v_x = None
        self.tar_v_y = None
        self.random_tar = np.array([None, None])
        self.la_en_b = 24

        self.ob_for_dis_x = [] 
        self.ob_for_dis_y = [] 

        vector_data = obs["vector"]
        sx, sy = vector_data[0][0], vector_data[0][1]
        dynamic_obstacles = [vector_data[5][:2], vector_data[6][:2], vector_data[7][:2], vector_data[8][:2], vector_data[9][:2]]

        for ob in dynamic_obstacles:
            for x in np.arange(ob[0] - 0.15, ob[0] + 0.15, obstacle_prec):
                for y in np.arange(ob[1] - 0.15, ob[1] + 0.15, obstacle_prec):
                    self.ox.append(x)
                    self.oy.append(y)
                    self.conf_ox.append(x)
                    self.conf_oy.append(y)
                
        for ob in dynamic_obstacles:
            self.ob_for_dis_x.append((ob[0]))
            self.ob_for_dis_y.append((ob[1]))

            
        PathPlanner.set_planner(self.ox, self.oy)

        obst_control = lambda x: all((abs(x[0] - self.ob_for_dis_x[i]) > 0.15 + 0.4 or abs(x[1] - self.ob_for_dis_y[i]) > 0.15 + 0.4) for i in range(len(self.ob_for_dis_x)))
        critical_points = list(filter(obst_control, critical_points))

    def update_state(self, obs):
        vector_data = obs["vector"]
        self.state.x, self.state.y = vector_data[0][0], vector_data[0][1]
        self.ang = vector_data[0][2]

    def check_activation(self, obs):
        vector_data = obs["vector"]
        tar = -1
        for i in range(5):
            if vector_data[5 + i][2] == False:
                tar = i
                break
        # print(f"=== Next goal: [{tar}]")
        return tar

    def update_activation_path(self, obs, tar):
        vector_data = obs["vector"]

        sx, sy = vector_data[0][0], vector_data[0][1]
        # print(f"sx: {sx}")
        # print(f"sy: {sy}")
        gx, gy = vector_data[5 + tar][0], vector_data[5 + tar][1]        
        # print(f"gx: {gx}")
        # print(f"gy: {gy}")

        path_x, path_y = PathPlanner.get_path(sx, sy, gx, gy, self.ox, self.oy)

        if print_path:
            print(path_x)
            print(path_y)

        lx = len(path_x)
        origin_begin_x = path_x[-1]
        origin_end_x = path_x[0]
        path_x = path_x[min(-lx//6, -1)::min(-lx//10, -1)]

        ly = len(path_y)
        origin_begin_y = path_y[-1]
        origin_end_y = path_y[0]
        path_y = path_y[min(-ly//6, -1)::min(-ly//10, -1)]

        if origin_begin_x not in path_x:
            path_x = [origin_begin_x] + path_x
            path_y = [origin_begin_y] + path_y
        if origin_end_x not in path_x:
            path_x = path_x + [origin_end_x]
            path_y = path_y + [origin_end_y]

        path_x += [gx]
        path_y += [gy]
        
        if print_path:
            print(path_x)
            print(path_y)

        self.cx, self.cy, self.cyaw, ck, s = cubic_spline_planner.calc_spline_course(path_x, path_y, ds=0.10)

        if show_spline:
            plt.plot(path_x, path_y, "xb", label="input")
            plt.plot(self.cx, self.cy, "-r", label="spline")
            plt.grid(True)
            plt.axis("equal")
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.legend()
            plt.show()

        fake_yaw = np.arctan2(path_y[1] - path_y[0], path_x[1] - path_x[0])
        self.state = Controller3.State(x=sx, y=sy, yaw=fake_yaw, v=2.0)
        self.target_idx, _ = Controller3.calc_target_index(self.state, self.cx, self.cy)

        if show_animation:
            self.simulation(obs, self.cx, self.cy, self.cyaw, ck, fake_yaw)

        # print(f"=== Updated path for goal [{tar + 1}]")

    def get_activation_action(self, sx, sy):
        ai = Controller3.pid_control(target_speed, self.state.v)
        di, self.target_idx = Controller3.stanley_control(self.state, self.cx, self.cy, self.cyaw, self.target_idx)

        last_x, last_y = self.state.x, self.state.y

        self.state.update(ai, di, sx, sy)

        print('==', self.state.v)

        self.tar_v_x = self.state.v * np.cos(self.state.yaw)
        self.tar_v_y = self.state.v * np.sin(self.state.yaw)

        if show_speed:
            print(f"vx: {self.tar_v_x}     vy: {self.tar_v_y}")
        if show_pos:
            print(f"x: {self.state.x}      y: {self.state.y}")

        return [self.tar_v_x, self.tar_v_y, 0, 0]

    def get_activation_rotation(self, obs, tar):
        vector_data = obs["vector"]

        sx, sy, stheta = vector_data[0][0], vector_data[0][1], vector_data[0][2]
        gx, gy = vector_data[5 + tar][0], vector_data[5 + tar][1]        

        tar_theta = np.arctan((gy - sy) / (gx - sx))
        if np.tan(tar_theta) * (gy - sy) < 0:
            tar_theta += math.pi

        if abs(tar_theta - stheta) > math.pi:
            w = - (tar_theta - stheta) / dt
        else:
            w = + (tar_theta - stheta) / dt

        return w

    def get_fight_action(self, obs):
        vec = obs['vector']

        cu_x, cu_y, cu_w = vec[0]
        en_x, en_y, en_w = vec[3]

        cu_h, cu_b = obs['vector'][1]
        en_h, en_b = obs['vector'][4]

        if en_b == 0:
            print('嘿嘿嘿你没子弹了吧～')
            action = [0, 0, 100, 0]
        else:
            action = [0, 0, 0, 0]

            # shoot or not
            ob_to_line_prec = 0.03
            ob = list(zip(self.conf_ox, self.conf_oy))
            a, b = np.linalg.solve([[cu_x, 1], [en_x, 1]], [[cu_y], [en_y]]) # y = ax + b
            dist_control = lambda x: np.abs(a * x[0] - x[1] + b) / np.sqrt(a ** 2 + 1) <= ob_to_line_prec
            cros_control = lambda x: np.sign(x[0] - en_x) == np.sign(cu_x - x[0]) and np.sign(en_y - x[1]) == np.sign(x[1] - cu_y)
            # plt.clf()
            # plt.plot([cu_x], [cu_y], 'xb')
            # plt.plot([en_x], [en_y], 'xr')
            # plt.plot(self.ox, self.oy, '.', 'grey')
            # plt.plot([i[0] for i in ob], [i[1] for i in ob], '.r')
            # plt.pause(0.001)
            # plt.show(block=True)
            ob = list(filter(dist_control, ob))
            # plt.clf()
            # plt.plot([cu_x], [cu_y], 'xb')
            # plt.plot([en_x], [en_y], 'xr')
            # plt.plot(self.ox, self.oy, '.', 'grey')
            # plt.plot([i[0] for i in ob], [i[1] for i in ob], '.r')
            # plt.pause(0.001)
            # plt.show(block=True)
            ob = list(filter(cros_control, ob))
            # plt.clf()
            # plt.plot([cu_x], [cu_y], 'xb')
            # plt.plot([en_x], [en_y], 'xr')
            # plt.plot(self.ox, self.oy, '.', 'grey')
            # plt.plot([i[0] for i in ob], [i[1] for i in ob], '.r')
            # plt.pause(0.001)
            # plt.show(block=True)

            if len(ob) <= 0 and cu_b > 0:
                action[3] = 1
            else:
                action[3] = 0

            

            # random walk
            if (self.random_tar == None).all() or math.hypot(self.random_tar[0] - cu_x, self.random_tar[1] - cu_y) < 0.5:
            # if self.check_tar(cu_x, cu_y, en_x, en_y) == False:
                self.random_tar = self.get_fight_route(cu_x, cu_y, en_x, en_y)
            action[0], action[1] = self.get_fight_velocity(cu_x, cu_y)
            # action[0] = np.random.rand()
            # action[1] = np.random.rand()

            future_x, future_y = cu_x - action[0] * dt, cu_y - action[1] * dt

            # rotation
            # tar = np.arctan2(en_y - cu_y, en_x - cu_x)
            # tar = np.arctan((en_y - cu_y) / (en_x - cu_x))
            tar = np.arctan((en_y - future_y) / (en_x - future_x))
            if np.tan(tar) * (en_y - cu_y) < 0:
                tar += math.pi
            if abs(tar - cu_w) > math.pi:
                action[2] = - (tar - cu_w) / dt * 1.2
            else:
                action[2] = + (tar - cu_w) / dt * 1.2

            if abs(tar - cu_w) > math.pi / 12 or math.hypot(cu_x - en_x, cu_y - en_y) > 3:
                action[3] = 0

            # if action[3]:
            #     action[0], action[1] = 0, 0

            # if action[1] > 0.1:
            #     action[2] -= +1
            # elif action[1] < 0.1:
            #     action[2] += -1

        return action

    def check_tar(self, sx, sy, ex, ey):
        cand = critical_points[:]

        dist_control = lambda x: math.hypot(x[0] - ex, x[1] - ey) >= 2.5 and math.hypot(x[0] - sx, x[1] - sy) >= 0.5
        angl_control = lambda x: np.arccos(((x[0] - ex) * (sx - ex) + (x[1] - ey) * (sy - ey)) / math.hypot(x[0] - ex, x[1] - ey) / math.hypot(sx - ex, sy - ey)) > math.pi / 10
        # cros_control = lambda x: not (np.sign(x[0] - ex) == np.sign(ex - sx) or np.sign(x[1] - ey) == np.sign(ey - sy))
        cros_control = lambda x: math.hypot(x[0] - ex, x[1] - ey) >= math.hypot(x[0] - sx, x[1] - sy)

        cand = list(filter(dist_control, cand))
        cand = list(filter(angl_control, cand))
        cand = list(filter(cros_control, cand))

        if self.random_tar not in cand or math.hypot(self.random_tar[0] - sx, self.random_tar[1] - sy) < 0.15:
            return False

        return True
    
    def get_fight_route(self, sx, sy, ex, ey):

        cand = critical_points[:]

        dist_control = lambda x: math.hypot(x[0] - ex, x[1] - ey) >= 2 and math.hypot(x[0] - sx, x[1] - sy) >= 0.5
        angl_control = lambda x: np.arccos(((x[0] - ex) * (sx - ex) + (x[1] - ey) * (sy - ey)) / math.hypot(x[0] - ex, x[1] - ey) / math.hypot(sx - ex, sy - ey)) > math.pi / 10
        # cros_control = lambda x: not (np.sign(x[0] - ex) == np.sign(ex - sx) or np.sign(x[1] - ey) == np.sign(ey - sy))
        cros_control = lambda x: math.hypot(x[0] - ex, x[1] - ey) >= math.hypot(x[0] - sx, x[1] - sy)

        cand = list(filter(dist_control, cand))
        cand = list(filter(angl_control, cand))
        cand = list(filter(cros_control, cand))

        # tar = cand[np.random.choice(np.arange(len(cand)))]
        try:
            tar = cand[np.random.choice(np.arange(len(cand)))]
        except:
            tar = critical_points[np.random.choice(np.arange(len(critical_points)))] 
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print('cand:',cand)
        # print('tar:', tar)
        # print('obstacles: ', list(zip(self.ob_for_dis_x, self.ob_for_dis_y)))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # plt.clf()
        # plt.plot([sx], [sy], 'xb')
        # plt.plot([ex], [ey], 'xr')
        # plt.plot([i[0] for i in critical_points], [i[1] for i in critical_points], '.', color='0.9')
        # plt.plot(self.ox, self.oy, '.')
        # plt.plot([i[0] for i in cand], [i[1] for i in cand], '.g')
        # plt.plot([tar[0]], [tar[1]], 'ob')
        # plt.pause(0.001)
        # plt.show(block=False)

        path_x, path_y = PathPlanner.get_path(sx, sy, tar[0], tar[1], self.ox, self.oy)

        lx = len(path_x)
        origin_begin_x = path_x[-1]
        origin_end_x = path_x[0]
        path_x = path_x[min(-lx//6, -1)::min(-lx//10, -1)]

        ly = len(path_y)
        origin_begin_y = path_y[-1]
        origin_end_y = path_y[0]
        path_y = path_y[min(-ly//6, -1)::min(-ly//10, -1)]

        if origin_begin_x not in path_x:
            path_x = [origin_begin_x] + path_x
            path_y = [origin_begin_y] + path_y
        if origin_end_x not in path_x:
            path_x = path_x + [origin_end_x]
            path_y = path_y + [origin_end_y]

        path_x += [tar[0]]
        path_y += [tar[1]]
         
        self.cx, self.cy, self.cyaw, ck, s = cubic_spline_planner.calc_spline_course(path_x, path_y, ds=0.10)

        fake_yaw = np.arctan2(path_y[1] - path_y[0], path_x[1] - path_x[0])
        self.state = Controller3.State(x=sx, y=sy, yaw=fake_yaw, v=2)
        self.target_idx, _ = Controller3.calc_target_index(self.state, self.cx, self.cy)
        print("random_tar", tar)

        return tar 

    def get_fight_velocity(self, sx, sy):
        ai = Controller3.pid_control(target_speed, self.state.v)
        di, self.target_idx = Controller3.stanley_control(self.state, self.cx, self.cy, self.cyaw, self.target_idx)

        last_x, last_y = self.state.x, self.state.y

        self.state.update(ai, di, sx, sy)

        self.tar_v_x = self.state.v * np.cos(self.state.yaw)
        self.tar_v_y = self.state.v * np.sin(self.state.yaw)

        return self.tar_v_x, self.tar_v_y 

    def simulation(self, obs, cx, cy, cyaw, ck, yaw):
        vector_data = obs["vector"]
        sx, sy = vector_data[0][0], vector_data[0][1]
        state = Controller3.State(x=sx, y=sy, yaw=yaw, v=0.0)
        max_simulation_time = 100000.0
        last_idx = len(cx) - 1

        time = 0.0
        x = [state.x]
        y = [state.y]
        yaw = [state.yaw]
        v = [state.v]
        t = [0.0]
        target_idx, _ = Controller3.calc_target_index(state, cx, cy)

        while max_simulation_time >= time and last_idx > target_idx:
            ai = Controller3.pid_control(target_speed, state.v)
            di, target_idx = Controller3.stanley_control(state, cx, cy, cyaw, target_idx)
            di = np.clip(di, -max_steer, max_steer)

            state.x += state.v * np.cos(state.yaw) * dt
            state.y += state.v * np.sin(state.yaw) * dt
            state.yaw += state.v / L * np.tan(di) * dt
            state.yaw = Controller3.normalize_angle(state.yaw)
            state.v += ai * dt
            time += dt

            x.append(state.x)
            y.append(state.y)
            yaw.append(state.yaw)
            v.append(state.v)
            t.append(time)

            if show_animation:
                plt.cla()
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(cx, cy, ".r", label="course")
                plt.plot(x, y, "-b", label="trajectory")
                plt.plot(cx[target_idx], cy[target_idx], "xg", label="target")
                plt.axis("equal")
                plt.grid(True)
                plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
                plt.pause(0.001)
        plt.close('all')
