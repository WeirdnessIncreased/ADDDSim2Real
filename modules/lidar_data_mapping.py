import math
import matplotlib.pyplot as plt
import numpy as np
from cmath import pi
import cv2

status_x = 0
status_y = 0
ERROR_X = 0
ERROR_Y = 0
step = 0
y_step = 0
final_x_step = 0
final_y_step = 0

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
    res = cv2.filter2D( inputs, -1, filter )
    axis = np.where(res==np.max(res))
    
    tx = int(axis[0][0])
    ty = int(axis[1][0])
    # print( tx, ty )
    return tx, ty

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

def normal_method( vector_data, occupancy_map ):
    cutted_obstacle_map, x, y = cut_obstacle( vector_data[0], g_obstacle_map )
    tx, ty = numpy_conv( cutted_obstacle_map, occupancy_map )
    x = ( x + tx ) * 0.02 
    y = ( 224 - ( y + ty ) ) * 0.02
    # print( "final", x, y )
    return x, y

def check_x( system_axis, temp_x, temp_y, occupancy_map ):
    global status_x
    global ERROR_X
    # print("1")
    sys_x = system_axis
    obstacle_map = g_obstacle_map
    left_x, right_x = -1, -1
    # print("2")
    for i in range( 0, 75 ):
        if( occupancy_map[i][75] == 8 ):
            left_x = i
            break
    # print("3")
    left_dis = 75 - left_x
    dis_error = 999999
    real_x_pos = -1
    if( left_x != -1 ):
        # print("4")
        x_pos, y_pos = temp_x / 0.02, temp_y / 0.02
        x_pos, y_pos = (int)(x_pos), (int)(y_pos)
        # print("5")
        for y in range( y_pos - 16, y_pos + 16 ):
            if( y >= 0 and y <= 224 ):
                for i in range(  x_pos, x_pos - left_dis - 30, -1 ):
                    if( ( i >= 0 and i <= 404 ) and obstacle_map[ i + 150 ][ y + 150 ] == 1 ):
                        if( abs( x_pos - i - left_dis ) < dis_error ):
                            dis_error = abs( x_pos - i - left_dis )
                            real_x_pos = i + left_dis
                            # print("6")
        if( real_x_pos != -1 ):
            status_x = 1
        # print( "Find it on left!",( real_x_pos * 0.02 - sys_x ) )
            ERROR_X = ( real_x_pos * 0.02 - sys_x )
            return 1

    for i in range( 76, 150 ):
        if( occupancy_map[i][75] == 8 ):
            right_x = i
            break
    right_dis = right_x - 75
    real_x_pos = -1
    dis_error = 999999
    # print( "7", right_x )
    if( right_x != -1 ):
        '''
        x_0, y_0 = [], []
        for i in range( 0, 150 ):
            for j in range( 0, 150 ):
                if( occupancy_map[i][j] == 8 ):
                    x_0.append(i)
                    y_0.append(j)
        plt.plot( x_0, y_0, '.g' )
        plt.show( block = True )'''
        x_pos, y_pos = (int)( temp_x / 0.02 ), (int)(temp_y / 0.02)
        for y in range( y_pos - 16, y_pos + 16 ):
            if( y >= 0 and y <= 224 ):
                for i in range(  x_pos, x_pos + right_dis + 30 ):
                    if( ( i >= 0 and i <= 404 ) and obstacle_map[ i + 150 ][ y + 150 ] == 1 ):
                        if( abs( i - x_pos - right_dis ) < dis_error ):
                            dis_error = abs( i - x_pos - right_dis )
                            real_x_pos = i - right_dis
                            # p#rint("8")
        if( real_x_pos != -1 ):
            status_x = 1
            # print("9")
        # print( "Find it on right!",( real_x_pos * 0.02 - sys_x ) )
            ERROR_X = ( real_x_pos * 0.02 - sys_x )
            return 1
    return 0
    


def check_y( system_axis, temp_x, temp_y, occupancy_map ):
    global status_y
    global ERROR_Y
    sys_y = system_axis
    
    obstacle_map = get_obstacle()
    top_x, bottom_x = -1, -1
    
    for i in range( 0, 75 ):
        if( occupancy_map[75][i] == 8 ):
            top_x = i
            break
    # print( "top:", top_x )
    top_dis = 75 - top_x
    dis_error = 999999
    if( top_x != -1 ):
        x_pos, y_pos = temp_x / 0.02, temp_y / 0.02
        x_pos, y_pos = (int)(x_pos), (int)(y_pos)
        for x in range( x_pos - 16, x_pos + 16 ):
            if( x >= 0 and x <= 404 ):
                for i in range(  y_pos, y_pos - top_dis - 30, -1 ):
                    if( ( i >= 0 and i <= 224 ) and obstacle_map[ x + 150 ][ i + 150 ] == 1 ):
                        if( abs( y_pos - i - top_dis ) < dis_error ):
                            dis_error = abs( x_pos - i - top_dis )
                            real_y_pos = i + top_dis
        status_y = 1
        ERROR_Y = ( real_y_pos * 0.02 - sys_y )
        return 1 
    
    for i in range( 76, 150 ):
        if( occupancy_map[75][i] == 8 ):
            bottom_x = i
            break
    bottom_dis = bottom_x - 75
    dis_error = 999999
    if( bottom_x != -1 ):
        x_pos, y_pos = (int)( temp_x / 0.02 ), (int)( temp_y / 0.02 )
        for x in range( x_pos - 16, x_pos + 16 ):
            if( x >= 0 and x <= 404 ):
                for i in range(  y_pos, y_pos + bottom_dis + 30 ):
                    if( ( i >= 0 and i <= 224 ) and obstacle_map[ x + 150 ][ i + 150 ] == 1 ):
                        if( abs( i - y_pos - bottom_dis ) < dis_error ):
                            dis_error = abs( i - y_pos - bottom_dis )
                            real_y_pos = i - bottom_dis
        status_y = 1
        ERROR_Y = ( real_y_pos * 0.02 - sys_y )
        return 1
    return 0

def lidar_mapping( vector_data, laser_data ):
    global step
    step += 1
    bo = 1
    ang, dist = [], []
    for i in range( 0, 61 ):
        # print( (float)( -vector_data[0][2] + ( 135 * ( i - 30 ) / 30 * pi / 180) )  )
        ang.append((float)( pi * 0.5 - vector_data[0][2] + ( 135 * ( i - 30 ) / 30 * pi / 180) ))
        dist.append( (float)(laser_data[i]) )
    occupancy_map = lidar_to_gird_map( ang, dist )  
    
    temp_x, temp_y = normal_method( vector_data, occupancy_map )
    if( check_x( vector_data[0][0], temp_x, temp_y, occupancy_map ) ):
        x = vector_data[0][0] + ERROR_X
    else:
        x = vector_data[0][0] + ERROR_X
        bo = 0
    if( check_y( vector_data[0][1], temp_x, temp_y, occupancy_map ) ):
        y = vector_data[0][1] + ERROR_Y
    else:
        y = vector_data[0][1] + ERROR_Y
        bo = 0
    # print( "fixed info", ERROR_X, ERROR_Y )
    
    return x, y, bo
