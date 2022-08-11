import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk, binary_dilation

try:
    from params import args
except:
    from elwin.params import args


########## hyperparameters ##########
INFLATION_RADIUS = 23 # size of RoboMaster EP Robot: 32×24×27 [cm]
########## hyperparameters ##########


def generate_static_map():
    static_map = np.zeros((808, 448))
    fixed_obstacles = {
        'B1': [7.08, 1.00, 8.08, 1.2],
        'B2': [5.78, 2.14 ,6.58 , 2.34],
        'B3': [6.58, 3.48, 6.78, 4.48],
        'B4': [3.54, 0.935, 4.54 , 1.135],
        # 'B5': [3.864, 2.064, 4.216, 2.416],
        'B6': [3.54, 3.345, 4.54, 3.545],
        'B7': [1.5, 0, 1.7, 1],
        'B8': [1.5, 2.14, 2.3, 2.34],
        'B9': [0, 3.28, 1, 3.48]
    }

    ob_img = cv2.imread('../files/border_and_center.jpg') # B5 and border

    for i in range(448):
        for j in range(808):
            if ob_img[i][j][0] < 200:
                static_map[j][i] = 1

    for name in fixed_obstacles:
        x1, y1, x2, y2 = fixed_obstacles[name]
        x1 = int(np.floor(x1 * 100))
        y1 = int(np.floor(y1 * 100))
        x2 = int(np.ceil(x2 * 100))
        y2 = int(np.ceil(y2 * 100))
        for i in range(x1, min(x2, 808)):
            for j in range(y1, min(y2, 448)):
                static_map[i][j] = 1

    img_save = static_map * 255
    cv2.imwrite('../files/static_map.bmp', img_save) # 注意保存后是翻转的
        

class CostMap:
    def __init__(self):
        try:
            self.base_map = (cv2.imread('../files/static_map.bmp')[:,:,0] != 0).astype(np.float32)
        except:
            self.base_map = (cv2.imread('./files/static_map.bmp')[:,:,0] != 0).astype(np.float32)
        self.dyob_map = np.zeros((808, 448))
        self.base_map = self.inflate(self.base_map) # 膨胀层
        self.update()
        # self.show()
        
    def update(self):
        self.map = ((self.base_map + self.dyob_map) != 0).astype(np.float32)
        # todo: inflation layer

    def show(self):
        plt.clf()
        plt.imshow(self.map, cmap='gray_r', origin='lower') # gray_r 是 reversed grayscale
        plt.show()

    def inflate(self, ori_map, obs=None):
        """
        ori_map: the map to be inflated
        obs: (num_obs, 2), a list of coordinates (x, y) [cm]
        """
        ret = binary_dilation((ori_map != 0), disk(INFLATION_RADIUS, dtype=bool)).astype(np.float32) # 这也太快了吧...
        return ret
        # rad = INFLATION_RADIUS
        # ret = ori_map # not deep copy
        # if obs == None:
        #     obs = np.argwhere(ori_map != 0)
        # obs = obs.tolist()
        # directions = [[0, 1, 1.0], [0, -1, 1.0],
        #               [1, 1, 1.4], [1, -1, 1.4],
        #               [-1,1, 1.4], [-1,-1, 1.4],
        #               [1, 0, 1.0], [-1, 0, 1.0]]
        # num_cal = 0
        # for x0, y0 in obs: # obs 在动态变化
        #     for dx, dy, dis in directions:
        #         num_cal += 1
        #         x1, y1 = x0 + dx, y0 + dy
        #         if x1 < 0 or x1 >= 808 or y1 < 0 or y1 >= 448:
        #             continue
        #         if ret[x1][y1] == 1 or ret[x1][y1] + dis > rad:
        #             continue
        #         elif (ret[x1][y1] == 0 and ret[x0][y0] + dis <= rad) \
        #                 or ret[x1][y1] > ret[x0][y0] + dis:
        #             ret[x1][y1] = ret[x0][y0] + dis
        #             obs.append([x1, y1])
        # print(num_cal)
        # return (ret != 0).astype(np.float32)    

    def update_dynamic_obstacle(self, obs):
        """
        obs: (num_obs, 2), a list of coordinates (x, y) [m]
        """
        self.dyob_map = np.zeros((808, 448))
        obs_for_inflation = []
        for ob in obs:
            cx, cy = int(ob[0] * 100), int(ob[1] * 100)
            # print('dyob', ob, cx, cy)
            # obs_for_inflation.append([cx, cy])
            xx = np.arange(max(0, cx - 10), min(cx + 10, 808))
            yy = np.arange(max(0, cy - 10), min(cy + 10, 448))
            xx, yy = np.repeat(xx, len(yy)), yy.tolist() * len(xx)
            self.dyob_map[xx, yy] = 1
        self.dyob_map = self.inflate(self.dyob_map) # , np.array(obs_for_inflation))
        if args.anime_dyob:
            plt.clf()
            plt.imshow(self.dyob_map, cmap='gray_r', origin='lower')
            plt.show()
        self.update()



if __name__ == '__main__':
    generate_static_map()
    costmap = CostMap()
    costmap.show()
