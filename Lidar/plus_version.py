import math
import matplotlib.pyplot as plt
import numpy as np
from cmath import pi
import lidar_data_mapping as re

SETTED_ERROR_X = 0.3
SETTED_ERROR_Y = 0.1
status_x = 0
status_y = 0
ERROR_X = 0
ERROR_Y = 0

def check_x( system_axis, temp_x, temp_y, occupancy_map ):
    global status_x
    global ERROR_X
    sys_x = system_axis
    obstacle_map = re.get_obstacle()
    left_x, right_x = -1, -1
    
    for i in range( 0, 75 ):
        if( occupancy_map[i][75] == 8 ):
            left_x = i
            break
    left_dis = 75 - left_x
    dis_error = 999999
    if( left_x != -1 ):
        x_pos, y_pos = temp_x / 0.02, temp_y / 0.02
        for y in range( y_pos - 16, y_pos + 16 ):
            if( y >= 0 and y <= 224 ):
                for i in range(  x_pos, x_pos - left_dis - 30, -1 ):
                    if( ( i >= 0 and i <= 404 ) and obstacle_map[ i + 150 ][ y + 150 ] == 1 ):
                        if( abs( x_pos - i - left_dis ) < dis_error ):
                            dis_error = abs( x_pos - i - left_dis )
                            real_x_pos = i + left_dis
        status_x = 1
        ERROR_X = real_x_pos * 0.02 - sys_x

    for i in range( 76, 150 ):
        if( occupancy_map[i][75] == 8 ):
            right_x = i
            break
    right_dis = right_x - 75
    dis_error = 999999
    if( right_x != -1 ):
        x_pos, y_pos = (int)( temp_x / 0.02 ), (int)(temp_y / 0.02)
        for y in range( y_pos - 16, y_pos + 16 ):
            if( y >= 0 and y <= 224 ):
                for i in range(  x_pos, x_pos + right_dis + 30 ):
                    if( ( i >= 0 and i <= 404 ) and obstacle_map[ i + 150 ][ y + 150 ] == 1 ):
                        if( abs( i - x_pos - right_dis ) < dis_error ):
                            dis_error = abs( i - x_pos - right_dis )
                            real_x_pos = i - right_dis
        status_x = 1
        ERROR_X = real_x_pos * 0.02 - sys_x
    


def check_y( temp_x, temp_y, occupancy_map ):
    global status_y
    global ERROR_Y
    obstacle_map = re.get_obstacle()


def test_function( vector_data, laser_data ):

    vector_data[0][0] += SETTED_ERROR_X
    vector_data[0][1] += SETTED_ERROR_Y

    ang, dist = [], []
    for i in range( 0, 61 ):
        # print( (float)( -vector_data[0][2] + ( 135 * ( i - 30 ) / 30 * pi / 180) )  )
        ang.append((float)( pi * 0.5 - vector_data[0][2] + ( 135 * ( i - 30 ) / 30 * pi / 180) ))
        dist.append( (float)(laser_data[i]) )

    x = vector_data[0][0] + ERROR_X * status_x
    y = vector_data[0][1] + ERROR_Y * status_y

    if( ( status_x * status_y ) == 0 ):
        occupancy_map = re.lidar_to_gird_map( ang, dist )
        temp_x, temp_y = re.lidar_mapping( vector_data, laser_data )
        if( status_x == 0 ):
            if( check_x( vector_data[0][0], temp_x, temp_y, occupancy_map ) ):
                x = vector_data[0][0] + ERROR_X
            else:
                x = temp_x

        if( status_y == 0 ):
            if( check_y( temp_x, temp_y, occupancy_map ) ):
                y = vector_data[0][1] + ERROR_Y
            else:
                y = temp_y
    print( status_x, ERROR_X )
    return x, y 
        

    
