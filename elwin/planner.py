import math
import copy
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fclusterdata

try:
    from params import args
    from costmap import CostMap
    from tracker import Tracker
except:
    from elwin.params import args
    from elwin.costmap import CostMap
    from elwin.tracker import Tracker


########## hyperparameters ##########
HEURISTIC_WEIGHT = 1.2
STEP_SIZE = 10 # [cm]
DEFAULT_GOAL_PRECISION = 0.10 # [m]
WAYPOINT_INTERVAL = 5 # [cm]
PATIENCE = 7
########## hyperparameters ##########


class Planner:
    def __init__(self):
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 808, 448
        self.x_width, self.y_width = 808, 448 
        self.motion = self.get_motion_model()
        self.obstacle_map = None 

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost # [cm]
            self.parent_index = parent_index

    def planning(self, sx, sy, gx, gy, cost_map, goal_prec):
        """
        input:
            sx: start x position [m]
            sy: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        self.obstacle_map = cost_map

        sx = int(sx * 100)
        sy = int(sy * 100)
        gx = int(gx * 100)
        gy = int(gy * 100)

        sx = min(807, max(0, sx))
        sy = min(447, max(0, sy))
        gx = min(807, max(0, gx))
        gy = min(447, max(0, gy))

        s_node = self.Node(sx, sy, 0.0, -1)
        g_node = self.Node(gx, gy, 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.get_index(s_node)] = s_node

        patience = -1

        while True:
            if len(open_set) == 0:
                print("[Error] Open set is empty!")
                break

            c_id = min(open_set, key=lambda x: open_set[x].cost + \
                    self.calc_heuristic(g_node, open_set[x]))
            c_node = open_set[c_id]

            if args.anime_plan:
                plt.plot(c_node.x, c_node.y, '.', color='0.8')
                plt.gcf().canvas.mpl_connect('key_release_event', \
                        lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if patience != -1:
                patience += 1
                if patience >= PATIENCE:
                    break

            del open_set[c_id]
            closed_set[c_id] = c_node

            if math.hypot(c_node.x - g_node.x, c_node.y - g_node.y) * 0.01 < goal_prec:
                g_node.parent_index = self.get_index(c_node)
                g_node.cost = c_node.cost
                print('Found a path.')
                if patience == -1:
                    patience = 0 
                elif math.hypot(c_node.x - g_node.x, c_node.y - g_node.y) < \
                        math.hypot(closed_set[g_node.parent_index].x - g_node.x, \
                        closed_set[g_node.parent_index].y - g_node.y):
                    g_node.parent_index = self.get_index(c_node)
                    g_node.cost = c_node.cost
                    print('found a better path')

            if math.hypot(c_node.x - g_node.x, c_node.y - g_node.y) * 0.01 < DEFAULT_GOAL_PRECISION:
                g_node.parent_index = self.get_index(c_node)
                g_node.cost = c_node.cost
                print('Found a best path.')
                break

            for i, _ in enumerate(self.motion):
                node = self.Node(c_node.x + self.motion[i][0],
                                 c_node.y + self.motion[i][1],
                                 c_node.cost + self.motion[i][2], c_id)
                n_id = self.get_index(node) # n_id 即 next_id

                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(g_node, closed_set)
        rx, ry = rx[::-1], ry[::-1]

        if not math.hypot(c_node.x - g_node.x, c_node.y - g_node.y) * 0.01 < DEFAULT_GOAL_PRECISION:
            rx, ry = rx[:-1], ry[:-1]

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        rx, ry = [goal_node.x], [goal_node.y]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(n.x)
            ry.append(n.y)
            parent_index = n.parent_index
            la_x, la_y = n.x, n.y
        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = HEURISTIC_WEIGHT
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y) # [cm]
        return d

    def get_index(self, node):
        return node.x * self.y_width + node.y

    def verify_node(self, node):
        px = node.x
        py = node.y

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        if self.obstacle_map[node.x][node.y] != 0:
            return False

        return True

    @staticmethod
    def get_motion_model():
        x = STEP_SIZE
        motion = [[x,  0, x],
                  [0,  x, x],
                  [-x, 0, x],
                  [0, -x, x],
                  [-x,-x, math.sqrt(2 * x ** 2)],
                  [-x, x, math.sqrt(2 * x ** 2)],
                  [x, -x, math.sqrt(2 * x ** 2)],
                  [x,  x, math.sqrt(2 * x ** 2)]]

        return motion

    def is_clear_path(self, sx, sy, gx, gy):
        cx, cy = sx, sy
        while True:
            flg = False
            if cx == gx and cy == gy:
                return True
            nx, ny = cx, cy
            for dx, dy, dis in self.motion:
                dx /= STEP_SIZE
                dy /= STEP_SIZE
                xx, yy = cx + dx, cy + dy
                if self.obstacle_map[int(xx)][int(yy)] != 0:
                    continue
                if math.hypot(gx - xx, gy - yy) < math.hypot(gx - nx, gy - ny):
                    nx, ny = xx, yy
                    flg = True
            if not flg:
                return False
            cx, cy = nx, ny

    def smoothen(self, rx, ry):
        rx, ry = np.array(rx), np.array(ry)
        tck, u = scipy.interpolate.splprep([rx, ry], s=10)
        dis = np.sum(np.hypot(rx[1:] - rx[:-1], ry[1:] - ry[:-1]))
        num = math.ceil(dis / WAYPOINT_INTERVAL)
        print(dis, num)
        u = np.linspace(min(u), max(u), num)
        nx, ny = scipy.interpolate.splev(u, tck)
        return nx, ny

    def simplify(self, rx, ry):
        """
        找到路径中的拐点，然后对拐点进行聚类，同簇的尝试用直线相连。
        """
        rx = np.array(rx)
        ry = np.array(ry)
        nx = copy.deepcopy(rx)
        ny = copy.deepcopy(ry)
        slope = (ry[1:] - ry[:-1]) / ((rx[1:] - rx[:-1]) + 1e-8)
        sdiff = slope[1:] - slope[:-1]
        iflct = np.argwhere(np.abs(sdiff) > 0).flatten() + 1 # 拐点下标 (inflection points)

        if len(iflct) <= 2:
            return rx, ry, [0], [0]
        else:
            pairs = np.hstack([rx.reshape((-1,1)), ry.reshape((-1,1))])[iflct]
            clusters = fclusterdata(pairs, 32, criterion="distance")
            unq, idx = np.unique(clusters, return_index=True)
            idx = np.array(sorted(idx))
        if len(idx) == 1:
            st, ed = iflct[0], iflct[-1]
            sx = rx[st]
            sy = ry[st]
            gx = rx[ed]
            gy = ry[ed]
            if self.is_clear_path(sx, sy, gx, gy):
                print('this is clear')
                linspace_len = len(nx[iflct[0] + 1:iflct[-1]]) + 2
                nx[iflct[0] + 1:iflct[-1]] = \
                        np.linspace(sx, gx, linspace_len).astype(int)[1:-1]
                ny[iflct[0] + 1:iflct[-1]] = \
                        np.linspace(sy, gy, linspace_len).astype(int)[1:-1]
        else:
            for i in range(len(idx) - 1):
                if idx[i + 1] - idx[i] > 2:
                    sx = rx[iflct[idx[i]]]
                    sy = ry[iflct[idx[i]]]
                    gx = rx[iflct[idx[i + 1] - 1]]
                    gy = ry[iflct[idx[i + 1] - 1]]
                    if self.is_clear_path(sx, sy, gx, gy):
                        linspace_len = len(nx[iflct[idx[i]] + 1:iflct[idx[i + 1] - 1]]) + 2
                        nx[iflct[idx[i]] + 1:iflct[idx[i + 1] - 1]] = \
                                np.linspace(sx, gx, linspace_len).astype(int)[1:-1]
                        ny[iflct[idx[i]] + 1:iflct[idx[i + 1] - 1]] = \
                                np.linspace(sy, gy, linspace_len).astype(int)[1:-1]
        return nx, ny, rx[iflct], ry[iflct]
    
    def get_path(self, sx, sy, gx, gy, cost_map, goal_prec=DEFAULT_GOAL_PRECISION):
        if args.anime_plan:
            plt.clf()
            plt.plot(np.argwhere(cost_map != 0)[:,0], np.argwhere(cost_map != 0)[:,1], ".k")
            plt.plot(sx * 100, sy * 100, "*g")
            plt.plot(gx * 100, gy * 100, "*b")
            plt.grid(True)
            plt.axis("equal")

        rx0, ry0 = self.planning(sx, sy, gx, gy, cost_map, goal_prec)
        try:
            rx1, ry1, ix, iy = self.simplify(rx0, ry0)
            rx2, ry2 = self.smoothen(rx1, ry1)
        except:
            plt.plot(np.argwhere(cost_map != 0)[:,0], np.argwhere(cost_map != 0)[:,1], ".k")
            plt.plot(sx * 100, sy * 100, "*g")
            plt.plot(gx * 100, gy * 100, "*b")
            plt.grid(True)
            plt.axis("equal")
            plt.plot(rx0, ry0, 'xg')
            plt.pause(0.001)
            plt.show(block=True)
            plt.pause(5)

        if args.anime_plan:
            plt.plot(rx0, ry0, 'xg')
            plt.plot(ix, iy, "+r")
            plt.plot(rx1, ry1, "2b")
            plt.plot(rx2, ry2, "-y")
            plt.pause(0.001)
            plt.show(block=True)
            plt.pause(5)

        return rx2, ry2

if __name__ == '__main__':
    costmap = CostMap()
    planner = Planner()
    tracker = Tracker()
    rx, ry = planner.get_path(0.40, 0.40, 7.60, 4.18, costmap.map, 0.50)
    tracker.update_path(rx, ry, 0, 0)
    rx, ry = planner.get_path(6.30, 1.30, 1.60, 1.40, costmap.map, 0.50)
    tracker.update_path(rx, ry, 0, 0)
