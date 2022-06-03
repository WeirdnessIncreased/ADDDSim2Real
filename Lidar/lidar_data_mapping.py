import math
from collections import deque
from platform import java_ver
import matplotlib.pyplot as plt
import numpy as np

def lidar_to_gird_map( ang, dist ):
    xy_resolution = 0.02
    ox = np.sin(ang) * dist
    oy = np.cos(ang) * dist
    
    for i in range( 60 ):
        print( 75 + ox[i] / xy_resolution, 75 + oy[i] / xy_resolution, ang[i], dist[i] )
    occupancy_map = np.zeros( (150, 150), dtype = int )
    for (x, y) in zip(ox, oy):
        if( abs(x / xy_resolution) < 75 and abs(y / xy_resolution) < 75 ):
            occupancy_map[ (int)(75 + x / xy_resolution), (int)(75 + y / xy_resolution) ] = 1
            # print( 150 + x / xy_resolution, 150 - y / xy_resolution )
    # occupancy_map = occupancy_map[::-1,:]
    # occupancy_map = occupancy_map.transpose( 1, 0 )
    '''
    for i in range( 0 , 299 ):
        for j in range( 0, 299 ):
            if( occupancy_map[i][j] == 1 ):
                print( i, j )
    '''
    for i in range( 149, -1, -1 ):
        for j in range( 0, 150 ):
            print( occupancy_map[j][i], end = '' )
        print()
    
    return occupancy_map

'''
a = [ 0, 0, 0, 0, 0, 0, 0, 0 ]
b = []
for i in range(7):
    b.append(a[:])
b[0][0] = 5
b[1][1] = 7
b[1][4] = 6
for i in range( 6, -1, -1 ):
    for j in range( 0, 7 ):
        print( b[j][i], end = '' )
    print()
'''