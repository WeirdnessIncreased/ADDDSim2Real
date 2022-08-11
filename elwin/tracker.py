import math
import copy
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

try:
    from params import args
    from costmap import CostMap
except:
    from elwin.params import args
    from elwin.costmap import CostMap


########## hyperparameters ##########
MAX_V = 2.5 # maxinum linear speed
MIN_V = 1.0 # minimum linear speed
MAX_W = math.pi / 4 # angular speed
MAX_A = 0 # acceleration
SAVGOL_WINDOW_SIZE = 13
LOOK_AHEAD = 1 # number of look ahead points
CURVATURE_THRESHOLD = 0.003 # curvature threshold for speed control
########## hyperparameters ##########


class Tracker:
    def __init__(self):
        self.path = None
        self.curv = None
        self.tarv = None

    def update_path(self, rx, ry, sv, gv):
        # 算 (x2, y2) 的曲率
        # (x1, y1) (x2, y2) (x3, y3)
        # (x - x1, y - y1) * (x2 - x1, y2 - y1) = 0
        # (x - x3, y - y3) * (x3 - x2, y3 - y2) = 0
        # (x2 - x1) * x - (x2 - x1) * x1 + (y2 - y1) * y - (y2 - y1) * y1 = 0
        # (x3 - x2) * x - (x3 - x2) * x3 + (y3 - y2) * y - (y3 - y2) * y3 = 0
        # a1 * x + b1 * y = c1
        # a2 * x + b2 * y = c2
        # A * (x, y) = B
        rx = np.array(rx)
        ry = np.array(ry)
        a1 = rx[1:-1] - rx[:-2:]
        a2 = rx[2::1] - rx[1:-1]
        b1 = ry[1:-1] - ry[:-2:]
        b2 = ry[2::1] - ry[1:-1]
        c1 = (rx[1:-1] - rx[:-2:]) * rx[:-2:] + (ry[1:-1] - ry[:-2:]) * ry[:-2:]
        c2 = (rx[2::1] - rx[1:-1]) * rx[2::1] + (ry[2::1] - ry[2::1]) * ry[2::1]
        a1 = np.expand_dims(np.expand_dims(a1, axis=-1), axis=-1)
        a2 = np.expand_dims(np.expand_dims(a2, axis=-1), axis=-1)
        b1 = np.expand_dims(np.expand_dims(b1, axis=-1), axis=-1)
        b2 = np.expand_dims(np.expand_dims(b2, axis=-1), axis=-1)
        c1 = np.expand_dims(c1, axis=-1)
        c2 = np.expand_dims(c2, axis=-1)
        r1 = np.concatenate([a1, b1], axis=-1)
        r2 = np.concatenate([a2, b2], axis=-1)
        aa = np.concatenate([r1, r2], axis=+1)
        bb = np.concatenate([c1, c2], axis=-1)

        xy = []
        for i in range(aa.shape[0]):
            A = aa[i]
            B = bb[i]
            if abs((ry[i + 1] - ry[i] / rx[i + 1] - rx[i]) - (ry[i + 2] - ry[i + 1] / rx[i + 2] - rx[i + 1])) < 1e-5:
                xy.append([-888, -888])
            else:
                xy.append(np.linalg.solve(A, B).tolist())
        xy = np.array(xy)
        
        self.path = np.array(list(zip(rx, ry)))
        self.curv = (1 / (np.hypot((xy - np.array(self.path[1:-1]) + 1e-8)[:,0], \
                (xy - np.array(self.path[1:-1]) + 1e-8)[:,1] + 1e-8))).tolist()
        self.tarv = [sv] + ((1 - np.array(self.curv) / CURVATURE_THRESHOLD) * MAX_V).tolist() + [gv]

        if args.anime_curv:
            plt.clf()
            plt.subplot(1, 4, 1)
            plt.scatter(rx, ry, c=self.tarv, cmap='plasma')
            plt.subplot(1, 4, 3)
            plt.plot(np.arange(0, len(self.tarv)), self.tarv, '-')
            plt.plot(np.arange(0, len(self.tarv)), self.tarv, 'ob')
            plt.subplot(1, 4, 2)

        self.tarv = savgol_filter(self.tarv, min(SAVGOL_WINDOW_SIZE, \
                len(self.tarv) if len(self.tarv) % 2 else len(self.tarv) \
                - 1), 4) # window size, polynomial order
        self.tarv[np.argwhere(self.tarv > MAX_V)] = MAX_V
        self.tarv[np.argwhere(self.tarv < MIN_V)] = MIN_V
        self.tarv[-1] = gv # 强制最后一个点的目标速度

        if args.anime_curv:
            plt.scatter(rx, ry, c=self.tarv, cmap='plasma')
            plt.subplot(1, 4, 4)
            plt.plot(np.arange(0, len(self.tarv)), self.tarv, '-')
            plt.plot(np.arange(0, len(self.tarv)), self.tarv, 'ob')
            plt.pause(0.001)
            plt.show(block=False)
            plt.pause(5)


        self.last = 0

    def get_action(self, x, y):
        # 找最近点，每次重找以防坐标误差
        cx, cy = x * 100, y * 100
        cur_id = np.argmin(np.hypot(self.path[:,0] - cx, self.path[:,1] - cy))
        tar_id = min(cur_id + LOOK_AHEAD, len(self.path) - 1)
        tx, ty = self.path[tar_id]
        tar_ve = self.tarv[cur_id] # 应该用当前的目标速度走吧
        tar_vx, tar_vy = (np.array([tx - cx, ty - cy]) / ((np.linalg.norm([tx - cx, ty - cy]) + 1e-32))) * tar_ve
        print('check tracker', cur_id, tar_id, cx, cy, tx, ty, tar_ve)
        return tar_vx, tar_vy


if __name__ == '__main__':
    tracker = Tracker()
