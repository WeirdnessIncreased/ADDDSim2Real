import pickle
import numpy as np

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
}

ob_for_dis_x = []
ob_for_dis_y = []
ob_for_dis_w = []
for name in fixed_obstacles:
    if name not in ['B2', 'B5', 'B8']:
        ob_for_dis_x.append((fixed_obstacles[name][0] + fixed_obstacles[name][2]) / 2)
        ob_for_dis_y.append((fixed_obstacles[name][1] + fixed_obstacles[name][3]) / 2)
        ob_for_dis_w.append([(fixed_obstacles[name][2] - fixed_obstacles[name][0]) / 2, (fixed_obstacles[name][3] - fixed_obstacles[name][1]) / 2])

map_size = [8.08, 4.48]
critical_points = []
for i in np.arange(0, map_size[0], map_size[0] / 40):
    for j in np.arange(0, map_size[1], map_size[1] / 20):
        width_x = map_size[0] / 50
        width_y = map_size[1] / 50
        critical_points.append([i + width_x / 2, j + width_y / 2])

obst_control = lambda x: all((abs(x[0] - ob_for_dis_x[i]) > ob_for_dis_w[i][0] + 0.2 or abs(x[1] - ob_for_dis_y[i]) > ob_for_dis_w[i][1] + 0.2) for i in range(len(ob_for_dis_x)))
edge_control = lambda x: x[0] > 0.3 and x[0] < 8.05 and x[1] > 0.3 and x[1] < 4.45
critical_points = list(filter(obst_control, critical_points))
critical_points = list(filter(edge_control, critical_points))

critical_points = np.array(critical_points)
pickle.dump(critical_points, open('./critical_points', 'wb')) 




