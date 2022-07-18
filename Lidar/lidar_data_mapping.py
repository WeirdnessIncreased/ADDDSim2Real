import math
import matplotlib.pyplot as plt
import numpy as np
from cmath import pi

def bresenham( start, end ):
    # en.wikipedia.org/wiki/Bresenham's_line_algorithm
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx) # determine how steep the line is

    if is_steep: # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    swapped = False  # swap start and end points if necessary and store swap state
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1 # iterate over bounding box generating points between start and end

    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points

def lidar_to_gird_map( ang, dist ):
    xy_resolution = 0.02
    # ang[ np.argwhere(np.is_nan(ang)==True) ] = 0
    # dist[ np.argwhere(np.is_nan(dist)==True) ] = 4
    ox = np.sin(ang) * dist
    oy = np.cos(ang) * dist
    '''
    for i in range( 60 ):
        print( 75 + ox[i] / xy_resolution, 75 + oy[i] / xy_resolution, ang[i], dist[i] )
    '''
    occupancy_map = np.zeros( (150, 150), dtype = int )
    for (x, y) in zip(ox, oy):
        if( abs(x / xy_resolution) < 75 and abs(y / xy_resolution) < 75 ):
            
            points = bresenham( ( 75, 75 ), ( (int)(75 + x / xy_resolution), (int)(75 + y / xy_resolution) ) )
            for fa in points:
                occupancy_map[fa[0]][fa[1]] = -1
            occupancy_map[ (int)(75 + x / xy_resolution), (int)(75 + y / xy_resolution) ] = 8
            # print( 150 + x / xy_resolution, 150 - y / xy_resolution )

    '''
    for i in range( 0 , 299 ):
        for j in range( 0, 299 ):
            if( occupancy_map[i][j] == 1 ):
                print( i, j )
    
    for i in range( 149, -1, -1 ):
        for j in range( 0, 150 ):
            print( occupancy_map[j][i], end = '' )
        print()
    '''
    return occupancy_map

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
    obstacle_map = np.zeros( ( 705, 525 ) , dtype=int )
    obstacle_map[ :, 147:150 ] += 1
    obstacle_map[ :, 374:378 ] += 1
    # obstacle_map[ 145:150, : ] += 1
    # obstacle_map[ 550:555, : ] += 1
    for i in range( 150, 555 ):
        for j in range( 150, 375 ):
            obstacle_map[i][j] = -1
    
    for pos in obstacles:
        for x in np.arange(pos[0], pos[2]):
            for y in np.arange(pos[1], pos[3]):    
                obstacle_map[ (int)( x + 150 ), (int)( y + 150 ) ] = 1

    return obstacle_map

def cut_obstacle( vector_data, obstacle_map ):
    # print(obstacle_map)
    x = int(vector_data[0] / 0.02) + 150
    y = int(vector_data[1] / 0.02) + 150
    # print( "x,y ", x - 150 - 150, 224 + 150 - y - 150 )
    obstacle_map = obstacle_map[ x - 150: x + 150, y - 150: y + 150 ]
    # print( size_x_left - size_x_right, size_y_down - size_y_up )
    return obstacle_map, x - 150 - 150, 224 - y

def numpy_conv(inputs, filter, padding="VALID"):
    inputs = np.array(inputs)
    filter = np.array(filter)
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
    '''
    x_0, y_0 = [], []
    x_1, y_1 = [], []
    x_2, y_2 = [], []
    for i in range( 0, 300 ):
        for j in range( 0, 300 ):
            if( inputs[i][j] == 0 ):
                x_0.append(i)
                y_0.append(j)
            if( inputs[i][j] == 1 ):
                x_1.append(i)
                y_1.append(j)
            if( inputs[i][j] == -1 ):
                x_2.append(i)
                y_2.append(j)    

    plt.clf()
    # plt.plot( x_0, y_0, '.g' )
    plt.plot( x_1, y_1, '.' )
    # plt.plot( x_2, y_2, '.r' )

    x_0, y_0 = [], []
    x_1, y_1 = [], []
    x_2, y_2 = [], []
    for i in range( 0, 150 ):
        for j in range( 0, 150 ):
            if( filter[i][j] == 0 ):
                x_0.append(i + tx)
                y_0.append(j + ty)
            if( filter[i][j] == 2 ):
                x_1.append(i + tx)
                y_1.append(j + ty)
            if( filter[i][j] == -1 ):
                x_2.append(i + tx)
                y_2.append(j + ty) 

    # plt.plot( x_0, y_0, '.g' )
    plt.plot( x_1, y_1, '.b' )
    plt.plot( x_2, y_2, '.r' )
    plt.pause(0.001)
    plt.show( block = True )'''
    print( "axis", tx, ty )
    return tx + 75, ty + 75

ori_obstacle_map = get_obstacle()
g_obstacle_map = get_obstacle()

def update(obstacle):
    # print( obstacle )
    global g_obstacle_map
    g_obstacle_map = ori_obstacle_map
    for( xx, yy ) in obstacle:
        for x in np.arange( xx / 0.02 - 7.5, xx / 0.02 + 7.5 ):
            for y in np.arange( yy / 0.02 - 7.5, yy / 0.02 + 7.5 ):    
                g_obstacle_map[ (int)( x + 150 ), (int)( y + 150 ) ] = 1

def lidar_mapping( vector_data, laser_data ):
    ang, dist = [], []
    for i in range( 0, 61 ):
        #print( (float)( -vector_data[0][2] + ( 135 * ( i - 30 ) / 30 * pi / 180) )  )
        ang.append((float)( pi * 0.5 - vector_data[0][2] + ( 135 * ( i - 30 ) / 30 * pi / 180) ))
        dist.append( (float)(laser_data[i]) )
    
    occupancy_map = lidar_to_gird_map( ang, dist )
    cutted_obstacle_map, x, y = cut_obstacle( vector_data[0], g_obstacle_map )
    tx, ty = numpy_conv( cutted_obstacle_map, occupancy_map )
    x = ( x + tx ) * 0.02 
    y = ( 224 - ( y + ty ) ) * 0.02
    # print( "final", x, y )
    return x, y
