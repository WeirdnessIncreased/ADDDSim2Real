import math
from modules import Controller3, PathPlanner
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from modules import cubic_spline_planner

################# Obstacle & Map #################

ox, oy = [], [] # store solid fixed obstacles and map borders
map_size = [8.08, 4.48] # [m]
obstacle_prec = 0.15 # [m]

fixed_obstacle = {
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

for name in fixed_obstacle:
    pos = fixed_obstacle[name]
    for x in np.arange(pos[0], pos[2], obstacle_prec):
        for y in np.arange(pos[1], pos[3], obstacle_prec):
            ox.append(x)
            oy.append(y)

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

show_animation = True

show_speed = False
show_pos = False
show_spline = False
print_path = False

target_speed = 24  #[m/s]

################# The robot agent #################

class Robot:
    def __init__(self, obs):
        self.ox = ox[:] # obstacle x for path planning
        self.oy = oy[:] # obstacle y for path planning
        self.cx = None
        self.cy = None
        self.cyaw = None
        self.target_idx = None
        self.state = None
        self.tar_v_x = None
        self.tar_v_y = None

        vector_data = obs["vector"]
        sx, sy = vector_data[0][0], vector_data[0][1]
        dynamic_obstacles = [vector_data[5][:2], vector_data[6][:2], vector_data[7][:2], vector_data[8][:2], vector_data[9][:2]]

        for ob in dynamic_obstacles:
            for x in np.arange(ob[0] - 0.15, ob[0] + 0.15, obstacle_prec):
                for y in np.arange(ob[1] - 0.15, ob[1] + 0.15, obstacle_prec):
                    self.ox.append(x)
                    self.oy.append(y)
                    # print(x, y)

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
        print(f"=== Next goal: [{tar}]")
        return tar

    def update_activation_path(self, obs, tar):
        vector_data = obs["vector"]

        sx, sy = vector_data[0][0], vector_data[0][1]
        print(f"sx: {sx}")
        print(f"sy: {sy}")
        gx, gy = vector_data[5 + tar][0], vector_data[5 + tar][1]        
        print(f"gx: {gx}")
        print(f"gy: {gy}")

        # gy -= 0.3
        # if math.floor(gy) <= 1:
        #     gy = vector_data[5 + tar][1] + 0.3
        # mx, my = map_size[0], map_size[1]

        path_x, path_y = PathPlanner.get_path(sx, sy, gx, gy, self.ox, self.oy)
        # path_x, path_y = path_x[-1::-2], path_y[-1::-2]

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

        # self.path = MotionController.CubicSplinePath(path_x, path_y)

        # print(path_x, path_y)

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
        self.state = Controller3.State(x=sx, y=sy, yaw=fake_yaw, v=0.0)
        self.target_idx, _ = Controller3.calc_target_index(self.state, self.cx, self.cy)

        if show_animation:
            self.simulation(obs, self.cx, self.cy, self.cyaw, ck, fake_yaw)

        print(f"=== Updated path for goal [{tar + 1}]")

    def get_activation_action(self, sx, sy):
        ai = Controller3.pid_control(target_speed, self.state.v)
        di, self.target_idx = Controller3.stanley_control(self.state, self.cx, self.cy, self.cyaw, self.target_idx)

        last_x, last_y = self.state.x, self.state.y

        self.state.update(ai, di, sx, sy)

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
            action = [0, 0, 100, 0]
        else:
            action = [0, 0, 0, 0]

            # rotation
            tar = np.arctan2(en_y - cu_y, en_x - cu_x)
            if abs(tar - cu_w) > math.pi:
                action[2] = - (tar - cu_W) / dt
            else:
                action[2] = + (tar - cu_w) / dt

            # shoot or not
            ob = list(zip(self.ox, self.oy))
            # close_ob = filter(lmbda x: math.hypot(x[0] - wsm))

            # random walk
            action[0] = np.random.rand() * 192608 % 17
            action[1] = np.random.rand() * 192608 % 17


        return action

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

            if show_animation:  # pragma: no cover
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(cx, cy, ".r", label="course")
                plt.plot(x, y, "-b", label="trajectory")
                plt.plot(cx[target_idx], cy[target_idx], "xg", label="target")
                plt.axis("equal")
                plt.grid(True)
                plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
                plt.pause(0.001)
        plt.close('all')

        # if show_animation:  # pragma: no cover
        #     plt.plot(cx, cy, ".r", label="course")
        #     plt.plot(x, y, "-b", label="trajectory")
        #     plt.legend()
        #     plt.xlabel("x[m]")
        #     plt.ylabel("y[m]")
        #     plt.axis("equal")
        #     plt.grid(True)

        #     plt.subplots(1)
        #     plt.plot(t, [iv * 3.6 for iv in v], "-r")
        #     plt.xlabel("Time[s]")
        #     plt.ylabel("Speed[km/h]")
        #     plt.grid(True)
        #     plt.show()

