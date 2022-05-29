import pickle
import sys
import numpy as np
import PathPlanner
import matplotlib.pyplot as plt

def get_obstacle():
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

    obstacles = list(zip(ox1, oy1, ox2, oy2))

    prec = 0.05

    ox = []
    oy = []

    for pos in obstacles:
        for x in np.arange(pos[0], pos[2], prec):
            for y in np.arange(pos[1], pos[3], prec):
                ox.append(x)
                oy.append(y)

    for x in np.arange(0, map_size[0], prec):
        ox.append(x)
        oy.append(0)
        ox.append(x)
        oy.append(map_size[1])

    for y in np.arange(0, map_size[1], prec):
        ox.append(0)
        oy.append(y)
        ox.append(map_size[0])
        oy.append(y)

    return ox, oy, obstacles

def get_critical():
    map_size = [8.08, 4.48]
    
    ox, oy, obstacle = get_obstacle()

    xx, yy = [], []
    xx_ = np.arange(0, map_size[0], map_size[0] / 20)
    yy_ = np.arange(0, map_size[1], map_size[1] / 10)

    def is_in_obstacle(x, y):
        for ob in obstacle:
            if x >= ob[0] and x <= ob[2] and y >= ob[1] and y <= ob[3]:
                return True
        return False

    for x in xx_:
        for y in yy_:
            if x == 0 or y == 0 or is_in_obstacle(x, y): continue
            xx.append(x)
            yy.append(y)

    return xx, yy

if __name__ == '__main__':
    mx, my = 8.08, 4.48

    ox, oy, obstacles = get_obstacle()
    xx, yy = get_critical()

    # print(xx)
    # plt.plot(ox, oy, ".k")
    # plt.plot(xx, yy, "og")
    # plt.grid(True)
    # plt.axis("equal")
    # plt.show()

    cnt = 1
    total = (len(xx) * len(yy) - len(xx)) / 2

    pathlets = {}

    for idx1, pos1 in enumerate(list(zip(xx, yy))):
        x1, y1 = pos1[0], pos1[1]
        for idx2, pos2 in enumerate(list(zip(xx, yy))):
            if idx2 <= idx1: continue
            print(f"Processing {cnt}/{total}", end="\r")
            sys.stdout.flush()
            x2, y2 = pos2[0], pos2[1]
            path_x, path_y = PathPlanner.get_path(x1, y1, x2, y2, mx, my, ox, oy)
            pathlets[(x1, y1, x2, y2)] = (path_x, path_y)
            cnt += 1

    with open("pathlets", "wb") as fs:
        pickle.dump(pathlets, fs)

    print("Finished (=ﾟωﾟ)ﾉ        ")
