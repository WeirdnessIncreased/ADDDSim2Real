import math
import pickle
from modules import PathPlanner
import numpy as np
from modules import Controller1

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

ox1, oy1, ox2, oy2 = [], [], [], []

for name in fixed_obstacle:
    ox1.append(fixed_obstacle[name][0])
    oy1.append(fixed_obstacle[name][1])
    ox2.append(fixed_obstacle[name][2])
    oy2.append(fixed_obstacle[name][3])

dt = 0.01 # [s]
L = 2 # [m] ? 
prec = 0.10

class Robot:
    def __init__(self, obs):
        self.path = None

        self.pathlets = pickle.load(open("pathlet/pathlets", "rb"))
        self.critical = pickle.load(open("pathlet/critical_points", "rb"))

        vector_data = obs["vector"]
        sx, sy = vector_data[0][0], vector_data[0][1]
        self.x = sx
        self.y = sy

        dynamic_obstacles = [vector_data[5][:2], vector_data[6][:2], vector_data[7][:2], vector_data[8][:2], vector_data[9][:2]]
        
        for ob in dynamic_obstacles:
            x = ob[0]
            y = ob[1]
            ox1.append(x - 0.15)
            oy1.append(y - 0.15)
            ox2.append(x + 0.15)
            oy2.append(y + 0.15)

        def is_in_obstacle(x, y):
            prec = 0.30
            for idx in range(len(ox1)):
                ob = [ox1[idx], oy1[idx], ox2[idx], oy2[idx]]
                if x >= ob[0] and x <= ob[2] and y >= ob[1] and y <= ob[3]:
                    return True
                if math.sqrt(math.hypot(x - ob[0], y - ob[1])) > prec or math.sqrt(math.hypot(x - ob[2], y - ob[3])) > prec:
                    return True
            return False

        for idx, pos in enumerate(self.critical):
            if is_in_obstacle(pos[0], pos[1]):
                del self.critical[idx]

    def update_state(self, obs):
        # note that velocity and yaw are updated in get_action()
        vector_data = obs["vector"]
        self.x, self.y = vector_data[0][0], vector_data[0][1]

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
        gx, gy = vector_data[5 + tar][0], vector_data[5 + tar][1]        

        mx, my = map_size[0], map_size[1]
        obstacles = list(zip(ox1, oy1, ox2, oy2))


        cand = self.critical
        cand = sorted(cand, key=lambda x: math.hypot(x[0] - sx, x[1] - sy))
        x1, y1 = cand[0]

        cand = self.critical
        cand = sorted(cand, key=lambda x: math.hypot(x[0] - gx, x[1] - gy))
        x2, y2 = cand[0]

        path_x1, path_y1 = PathPlanner.get_path(sx, sy, x1, y1, mx, my, obstacles)
        path_x2, path_y2 = PathPlanner.get_path(x2, y2, gx, gy, mx, my, obstacles)

        if (x1, y1, x2, y2) in self.pathlets:
            path_x, path_y = self.pathlets[(x1, y1, x2, y2)]
        else:
            path_x, path_y = self.pathlets[(x2, y2, x1, y1)]

        path_x = path_x[::-1]
        path_y = path_y[::-1]

        path_x = path_x1[::-1] + path_x + path_x2[::-1]
        path_y = path_y1[::-1] + path_y + path_y2[::-1]

        self.path = list(zip(path_x, path_y))

        print(f"=== Updated path for goal [{tar + 1}]")

    def get_activation_action(self):
        sx, sy = self.x, self.y

        tar = self.path[0]
        while math.hypot(tar[0] - sx, tar[1] - sy) <= prec:
            self.path = self.path[1:]
            tar = self.path[0]

        vx = (tar[0] - sx) / dt
        vy = (tar[1] - sy) / dt

        return [vx, vy, 0, 0]

    def get_activation_rotation(self, obs, tar):
        vector_data = obs["vector"]

        sx, sy, stheta = vector_data[0][0], vector_data[0][1], vector_data[0][2]
        gx, gy = vector_data[5 + tar][0], vector_data[5 + tar][1]        

        tar_theta = np.arctan((gy - sy) / (gx - sx))
        if np.tan(tar_theta) * (gy - sy) < 0:
            tar_theta += math.pi
        w = (tar_theta - stheta) / dt

        return w

