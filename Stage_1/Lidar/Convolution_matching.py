import numpy as np
import matplotlib.pyplot as plt

def get_obstacle( vector_data ):
    fixed_obstacle = {
        'B1': [ 7.08, 1.00, 8.08, 1.2],
        'B2': [ 5.78, 2.14 ,6.58 , 2.34], 
        'B3': [ 6.58, 3.48, 6.78, 4.48],
        'B4': [ 3.54, 0.935, 4.45 , 1.135], 
        'B5': [ 3.864, 2.064, 4.216, 2.416], 
        'B6': [ 3.54, 3.345, 4.45 , 3.545],
        'B7': [ 1.5, 0, 1.7, 1],
        'B8': [ 1.5, 2.14, 2.3, 2.34], 
        'B9': [ 0, 3.28, 1, 3.48],
        'B10': [ 0, 0, 0.02, 4.48 ],
        'B11': [ 8.08, 0, 8.1, 4.48],
        'B12': [ 0, 0, 8.1, 0.02 ],
        'B13': [ 0, 4.48, 8.08, 4.50 ]

    }

    ox1, oy1, ox2, oy2 = [], [], [], []

    for name in fixed_obstacle:
        ox1.append( fixed_obstacle[name][0] / 0.02 )
        oy1.append( fixed_obstacle[name][1] / 0.02 )
        ox2.append( fixed_obstacle[name][2] / 0.02 )
        oy2.append( fixed_obstacle[name][3] / 0.02 )

    obstacles = list(zip(ox1, oy1, ox2, oy2))
    # 8.08 / 0.02 = 404
    # 4.48 / 0.02 = 224
    obstacle_map = np.zeros( ( 705, 525 ) , dtype=int)
    
    for pos in obstacles:
        for x in np.arange(pos[0], pos[2]):
            for y in np.arange(pos[1], pos[3]):    
                obstacle_map[ (int)( x + 150 ), (int)( y + 150 ) ] = 1

    # plt.figure(1, figsize=(10, 4))
    # plt.subplot(122)
    # plt.imshow(obstacle_map)
    x = int(vector_data[0] / 0.02) + 150
    y = int(vector_data[1] / 0.02) + 150
    # print( "x,y ", x - 150 - 150, 224 + 150 - y - 150 )
    obstacle_map = obstacle_map[ x - 150: x + 150, y - 150: y + 150 ]
    # print( size_x_left - size_x_right, size_y_down - size_y_up )
    return obstacle_map, x - 150 - 150, 224 - y


def numpy_conv(inputs, filter, padding="VALID"):
    H, W = inputs.shape
    filter_size_x, filter_size_y = filter.shape
    # print( "size", H, W, filter_size_x, filter_size_y)
    # default np.floor
    # filter_center = int(filter_size / 2.0)
    # filter_center_ceil = int(np.ceil(filter_size / 2.0))
    result = np.zeros((H - filter_size_x + 1, W - filter_size_y + 1))
    H, W = inputs.shape
    Max_num = 0.0
    tx, ty = 0, 0
    # print("new size",H,W)
    for r in range(0, H - filter_size_x + 1):
        for c in range(0, W - filter_size_y + 1):
            cur_input = inputs[r : r + filter_size_x, c : c + filter_size_y ]
            cur_output = cur_input * filter
            conv_sum = np.sum(cur_output)
            if( conv_sum > Max_num ):
                Max_num = conv_sum
                tx = r
                ty = c
                # print( "1", tx, ty, Max_num )
            result[r, c] = conv_sum
    return result, tx + 75, ty + 75