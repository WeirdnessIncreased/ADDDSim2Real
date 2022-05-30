import numpy as np

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

    ox1, oy1, ox2, oy2 = [], [], [], []

    for name in fixed_obstacle:
        ox1.append( fixed_obstacle[name][0] / 0.02 )
        oy1.append( fixed_obstacle[name][1] / 0.02 )
        ox2.append( fixed_obstacle[name][2] / 0.02 )
        oy2.append( fixed_obstacle[name][3] / 0.02 )

    obstacles = list(zip(ox1, oy1, ox2, oy2))

    obstacle_map = int(np.zeros(( 8.08 / 0.02 , 4.48 / 0.02 )))

    for pos in obstacles:
        for x in np.arange(pos[0], pos[2]):
            for y in np.arange(pos[1], pos[3]):    
                obstacle_map[x][y] = 1

    return obstacle_map


def numpy_conv(inputs, filter, padding="VALID"):
    H, W = inputs.shape
    filter_size = filter.shape[0]
    # default np.floor
    # filter_center = int(filter_size / 2.0)
    # filter_center_ceil = int(np.ceil(filter_size / 2.0))

    result = np.zeros((H - filter_size + 1, W - filter_size + 1))
    H, W = inputs.shape
    # print("new size",H,W)
    for r in range(0, H - filter_size + 1):
        for c in range(0, W - filter_size + 1):
            cur_input = inputs[r : r + filter_size, c : c + filter_size]
            cur_output = cur_input * filter
            conv_sum = np.sum(cur_output)
            result[r, c] = conv_sum
    return result