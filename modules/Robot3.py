import math
from modules import Controller3, PathPlanner
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from modules import cubic_spline_planner

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

map_size = [8.08, 4.48]

k = 5  # control gain
Kp = 0.05  # speed proportional gain
dt = 0.1  # [s] time difference
L = 0.19  # [m] Wheel base of vehicle
max_steer = np.radians(30)  # [rad] max steering angle

show_animation = True 

target_speed = 30.0 / 3.6  #[m/s]

ox1, oy1, ox2, oy2 = [], [], [], []

for name in fixed_obstacle:
    ox1.append(fixed_obstacle[name][0])
    oy1.append(fixed_obstacle[name][1])
    ox2.append(fixed_obstacle[name][2])
    oy2.append(fixed_obstacle[name][3])


class Robot:
    def __init__(self, obs):
        self.cx = None
        self.cy = None
        self.cyaw = None
        self.target_idx = None

        vector_data = obs["vector"]
        sx, sy = vector_data[0][0], vector_data[0][1]
        self.ang = vector_data[0][2]
        self.state = Controller3.State(x=sx, y=sy, yaw=np.radians(20.0), v=0.0)

        dynamic_obstacles = [vector_data[5][:2], vector_data[6][:2], vector_data[7][:2], vector_data[8][:2], vector_data[9][:2]]
        
        for ob in dynamic_obstacles:
            x = ob[0]
            y = ob[1]
            ox1.append(x - 0.15)
            oy1.append(y - 0.15)
            ox2.append(x + 0.15)
            oy2.append(y + 0.15)

    def update_state(self, obs):
        # note that velocity and yaw are updated in get_action()
        vector_data = obs["vector"]
        self.x, self.y = vector_data[0][0], vector_data[0][1]
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

        gy -= 0.5
        mx, my = map_size[0], map_size[1]
        obstacles = list(zip(ox1, oy1, ox2, oy2))

        path_x, path_y = PathPlanner.get_path(sx, sy, gx, gy, mx, my, obstacles)
        # for i in range(len(path_x)):
            # path_x[i] = path_x[i] * 10
        print(path_x)
        print(path_y)
        lx = len(path_x)
        path_x = [path_x[-1]] + path_x[min(-lx//6, -1)::min(-lx//10, -1)]
        # for i in range(len(path_y)):
            # path_y[i] = path_y[i] * 10
        ly = len(path_y)
        path_y = [path_y[-1]] + path_y[min(-ly//6, -1)::min(-ly//10, -1)]
        print(path_x)
        print(path_y)
        # path_x = [path_x[-1]] + path_x[-3::-6]
        # path_y = [path_y[-1]] + path_y[-3::-6]
        # self.path = MotionController.CubicSplinePath(path_x, path_y)
        self.cx, self.cy, self.cyaw, ck, s = cubic_spline_planner.calc_spline_course(path_x, path_y, ds=0.1)
        self.target_idx, _ = Controller3.calc_target_index(self.state, self.cx, self.cy)
        self.simulation(obs, self.cx, self.cy, self.cyaw, ck)

        # Controller2.visualize(path_x, path_y, sx, sy)
        # MotionController.visualize(path_x, path_y, sx, sy)

        print(f"=== Updated path for goal [{tar + 1}]")

    def get_activation_action(self, ang):
        ai = Controller3.pid_control(target_speed, self.state.v)
        di, self.target_idx = Controller3.stanley_control(self.state, self.cx, self.cy, self.cyaw, self.target_idx)

        ang = self.ang - ang
        di = np.clip(di, -max_steer, max_steer)

        self.state.yaw += self.state.v / L * np.tan(di) * dt
        self.state.yaw = Controller3.normalize_angle(self.state.yaw)
        self.state.v += ai * dt
        vx = self.state.v * np.cos(self.state.yaw)
        vy = self.state.v * np.sin(self.state.yaw)
        tar_v_x = math.cos(-ang) * vx - math.sin(-ang) * vy 
        tar_v_y = math.sin(-ang) * vx + math.cos(-ang) * vy

        return [tar_v_x, tar_v_y, 0, 0]

    def get_activation_rotation(self, obs, tar):
        vector_data = obs["vector"]

        sx, sy, stheta = vector_data[0][0], vector_data[0][1], vector_data[0][2]
        gx, gy = vector_data[5 + tar][0], vector_data[5 + tar][1]        

        tar_theta = np.arctan((gy - sy) / (gx - sx))
        if stheta < 0: stheta = 2 * math.pi + stheta

        if gx > sx and gy > sy:
            w = (tar_theta - stheta) / dt
            print("1")
        elif gx < sx and gy > sy:
            w = (math.pi + tar_theta - stheta) / dt
            tar_theta = math.pi + tar_theta
            print("2")
        elif gx < sx and gy < sy:
            w = (math.pi + tar_theta - stheta) / dt
            print("3")
        elif gx > sx and gy < sy:
            w = (2 * math.pi + tar_theta - stheta) / dt
            print("4")

        return w
        

    def simulation(self, obs, cx, cy, cyaw, ck):
        vector_data = obs["vector"]
        sx, sy = vector_data[0][0], vector_data[0][1]
        state = Controller3.State(x=sx, y=sy, yaw=np.radians(20.0), v=0.0)
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

        if show_animation:  # pragma: no cover
            plt.plot(cx, cy, ".r", label="course")
            plt.plot(x, y, "-b", label="trajectory")
            plt.legend()
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.axis("equal")
            plt.grid(True)

            plt.subplots(1)
            plt.plot(t, [iv * 3.6 for iv in v], "-r")
            plt.xlabel("Time[s]")
            plt.ylabel("Speed[km/h]")
            plt.grid(True)
            plt.show()
