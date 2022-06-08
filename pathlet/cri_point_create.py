import pickle
import numpy as np

map_size = [8.08, 4.48]
critical_points = []
for i in np.arange(0, map_size[0], map_size[0] / 40):
    for j in np.arange(0, map_size[1], map_size[1] / 20):
        width_x = map_size[0] / 50
        width_y = map_size[1] / 50
        critical_points.append([i + width_x / 2, j + width_y / 2])
critical_points = np.array(critical_points)
pickle.dump(critical_points, open('./critical_points', 'wb')) 




