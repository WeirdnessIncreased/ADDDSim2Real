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

def check_x( temp_x, temp_y, occupancy_map ):
    global status_x
    global ERROR_X
    obstacle_map = re.get_obstacle()

def check_y( temp_x, temp_y, occupancy_map ):
    global status_y
    global ERROR_Y
    obstacle_map = re.get_obstacle()
    

def test_function( vector_data, laser_data ):

    vector_data[0][0] += SETTED_ERROR_X
    vector_data[0][1] += SETTED_ERROR_Y

    ang, dist = [], []
    for i in range( 0, 61 ):
        #print( (float)( -vector_data[0][2] + ( 135 * ( i - 30 ) / 30 * pi / 180) )  )
        ang.append((float)( pi * 0.5 - vector_data[0][2] + ( 135 * ( i - 30 ) / 30 * pi / 180) ))
        dist.append( (float)(laser_data[i]) )

    x = vector_data[0][0] + ERROR_X * status_x
    y = vector_data[0][1] + ERROR_Y * status_y

    if( ( status_x * status_y ) == 0 ):
        occupancy_map = re.lidar_to_gird_map( ang, dist )
        temp_x, temp_y = re.lidar_mapping( vector_data, laser_data )
        if( status_x == 0 ):
            if( check_x( temp_x, temp_y, occupancy_map ) ):
                x = vector_data[0][0] + ERROR_X
            else:
                x = temp_x

        if( status_y == 0 ):
            if( check_y( temp_x, temp_y, occupancy_map ) ):
                y = vector_data[0][1] + ERROR_Y
            else:
                y = temp_y

    return x, y 
        

    
