import numpy as np
import pickle
import matplotlib.pyplot as plt

critical_points = pickle.load(open('./critical_points', 'rb'))
fixed_obstacles = {
    'B1': [ 7.08, 1.00, 8.08, 1.2],
    'B2': [ 5.78, 2.14 ,6.58 , 2.34], 
    'B3': [ 6.58, 3.48, 6.78, 4.48],
    'B4': [ 3.54, 0.935, 4.45 , 1.135], 
    'B5': [ 3.864, 2.064, 4.216, 2.416], 
    'B6': [ 3.54, 3.345, 4.45 , 3.545],
    'B7': [ 1.5, 0, 1.7, 1],
    'B8': [ 1.5, 2.14, 2.3, 2.34], 
    'B9': [ 0, 3.28, 1, 3.48],
#     'B10': [ 0, 0, 8.08, 0.02],
#     'B11': [ 0, 4.48, 8.08, 4.50],
#     'B12': [ 0, 0, 0.02, 4.48],
#     'B13': [ 8.08, 0, 8.10, 4.48]
}
ob_for_dis_x = []
ob_for_dis_y = []
ob_for_dis_w = []
for name in fixed_obstacles:
    if name not in ['B2', 'B5', 'B8']:
        ob_for_dis_x.append((fixed_obstacles[name][0] + fixed_obstacles[name][2]) / 2)
        ob_for_dis_y.append((fixed_obstacles[name][1] + fixed_obstacles[name][3]) / 2)
        ob_for_dis_w.append([(fixed_obstacles[name][2] - fixed_obstacles[name][0]) / 2, (fixed_obstacles[name][3] - fixed_obstacles[name][1]) / 2])
        for i in np.arange(fixed_obstacles[name][0], fixed_obstacles[name][2], 0.02):
            for j in np.arange(fixed_obstacles[name][1], fixed_obstacles[name][3], 0.02):
                plt.plot(i, j, '.r')

for i in np.arange(0, 8.10, 0.02):
    for j in np.arange(0, 0.04, 0.02):
        plt.plot(i, j, '.r')
        plt.plot(i, j + 4.48, '.r')

for i in np.arange(0, 0.04, 0.02):
    for j in np.arange(0, 4.50, 0.02):
        plt.plot(i, j, '.r')
        plt.plot(i + 8.08, j, '.r')
obst_control = lambda x: all((abs(x[0] - ob_for_dis_x[i]) > ob_for_dis_w[i][0] + 0.2 or abs(x[1] - ob_for_dis_y[i]) > ob_for_dis_w[i][1] + 0.2) for i in range(len(ob_for_dis_x)))
edge_control = lambda x: x[0] > 0.3 and x[0] < 8.05 and x[1] > 0.3 and x[1] < 4.45

print(len(critical_points))
# critical_points = list(filter(obst_control, critical_points))
# critical_points = list(filter(edge_control, critical_points))
print(len(critical_points))


for i in critical_points:
    plt.plot(i[0], i[1], '.b')
plt.pause(0.001)
plt.show()
