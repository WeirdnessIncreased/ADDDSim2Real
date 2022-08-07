import math
import numpy as np
from costmap import CostMap
import matplotlib.pyplot as plt


HEURISTIC_WEIGHT = 1.2
STEP_SIZE = 10 # [cm]


show_animation = True


class AStarPlanner:
    def __init__(self):
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 448, 808
        self.x_width, self.y_width = 448, 808
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

        sx = min(447, max(0, sx))
        sy = min(807, max(0, sy))
        gx = min(447, max(0, gx))
        gy = min(807, max(0, gy))

        s_node = self.Node(sx, sy, 0.0, -1)
        g_node = self.Node(gx, gy, 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.get_index(s_node)] = s_node

        while True:
            if len(open_set) == 0:
                print("[Error] Open set is empty!")
                break

            c_id = min(open_set, key=lambda x: open_set[x].cost + \
                    self.calc_heuristic(g_node, open_set[x]))
            c_node = open_set[c_id]

            if show_animation:
                plt.plot(c_node.x, c_node.y, "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event', \
                        lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if math.hypot(c_node.x - g_node.x, c_node.y - g_node.y) * 0.01 < goal_prec:
                g_node.parent_index = c_node.parent_index
                g_node.cost = c_node.cost
                print('Found a path.')
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = c_node

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(c_node.x + self.motion[i][0],
                                 c_node.y + self.motion[i][1],
                                 c_node.cost + self.motion[i][2], c_id)
                n_id = self.get_index(node)

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

    def get_path(self, sx, sy, gx, gy, cost_map, goal_prec=0.10):

        if show_animation:
            plt.plot(np.argwhere(cost_map != 0)[:,0], np.argwhere(cost_map != 0)[:,1], ".k")
            plt.plot(sx, sy, "og")
            plt.plot(gx, gy, "xb")
            plt.grid(True)
            plt.axis("equal")

        rx, ry = self.planning(sx, sy, gx, gy, cost_map, goal_prec)

        if show_animation:
            plt.plot(rx, ry, "-r")
            plt.pause(0.001)
            plt.show(block=False)
            plt.pause(2)

        return rx, ry

if __name__ == '__main__':
    cost_map = CostMap()
    planner = AStarPlanner()
    planner.get_path(0.30, 0.30, 4.18, 7.60, cost_map.map, 0.10)
    pass
