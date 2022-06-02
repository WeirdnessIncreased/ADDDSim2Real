import math
from collections import deque
from platform import java_ver
import matplotlib.pyplot as plt
import numpy as np

EXTEND_AREA = 1.0

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

def calc_grid_map_config( ox, oy, xy_resolution ):
    """
    Calculates the size, and the maximum distances according to the the
    measurement center
    """
    min_x = round(min(ox) - EXTEND_AREA / 2.0)
    min_y = round(min(oy) - EXTEND_AREA / 2.0)
    max_x = round(max(ox) + EXTEND_AREA / 2.0)
    max_y = round(max(oy) + EXTEND_AREA / 2.0)
    xw = int(round((max_x - min_x) / xy_resolution))
    yw = int(round((max_y - min_y) / xy_resolution))
    # print("Max_x, Min_x", max_x, min_x)
    return min_x, min_y, max_x, max_y, xw, yw

def atan_zero_to_twopi(y, x):
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0
    return angle

def init_flood_fill( center_point, obstacle_points, xy_points, min_coord, xy_resolution ):
    """
    center_point: center point
    obstacle_points: detected obstacles points (x,y)
    xy_points: (x,y) point pairs
    """
    center_x, center_y = center_point
    prev_ix, prev_iy = center_x - 1, center_y
    ox, oy = obstacle_points
    xw, yw = xy_points
    min_x, min_y = min_coord
    occupancy_map = (np.ones((xw, yw))) * 0.5
    for (x, y) in zip(ox, oy):
        # x coordinate of the the occupied area
        ix = int(round((x - min_x) / xy_resolution))
        # y coordinate of the the occupied area
        iy = int(round((y - min_y) / xy_resolution))
        free_area = bresenham((prev_ix, prev_iy), (ix, iy))
        for fa in free_area:
            occupancy_map[fa[0]][fa[1]] = 0  # free area 0.0
        prev_ix = ix
        prev_iy = iy
    return

def flood_fill(center_point, occupancy_map):
    """
    center_point: starting point (x,y) of fill
    occupancy_map: occupancy map generated from Bresenham ray-tracing
    """
    # Fill empty areas with queue method
    sx, sy = occupancy_map.shape
    fringe = deque()
    fringe.appendleft(center_point)
    while fringe:
        n = fringe.pop()
        nx, ny = n
        if nx > 0: # West 
            if occupancy_map[nx - 1, ny] == 0.5:
                occupancy_map[nx - 1, ny] = 0.0
                fringe.appendleft((nx - 1, ny))
       
        if nx < sx - 1: # East
            if occupancy_map[nx + 1, ny] == 0.5:
                occupancy_map[nx + 1, ny] = 0.0
                fringe.appendleft((nx + 1, ny))
        
        if ny > 0: # North
            if occupancy_map[nx, ny - 1] == 0.5:
                occupancy_map[nx, ny - 1] = 0.0
                fringe.appendleft((nx, ny - 1))
        
        if ny < sy - 1: # South
            if occupancy_map[nx, ny + 1] == 0.5:
                occupancy_map[nx, ny + 1] = 0.0
                fringe.appendleft((nx, ny + 1))

def generate_ray_casting_grid_map(ox, oy, xy_resolution, breshen=True):
    """
    The breshen boolean tells if it's computed with bresenham ray casting
    (True) or with flood fill (False)
    """
    min_x, min_y, max_x, max_y, x_w, y_w = calc_grid_map_config(
        ox, oy, xy_resolution)

    occupancy_map = np.ones((x_w, y_w)) / 2
    center_x = int(round(-min_x / xy_resolution))  # center x coordinate of the grid map
    center_y = int(round(-min_y / xy_resolution))  # center y coordinate of the grid map
    # occupancy grid computed with bresenham ray casting
    if breshen:
        for (x, y) in zip(ox, oy):
            # x coordinate of the the occupied area
            ix = int(round((x - min_x) / xy_resolution))
            # y coordinate of the the occupied area
            iy = int(round((y - min_y) / xy_resolution))
            laser_beams = bresenham((center_x, center_y), (
                ix, iy))  # line form the lidar to the occupied point
            for laser_beam in laser_beams:
                occupancy_map[laser_beam[0]][
                    laser_beam[1]] = 0.0  # free area 0.0
            occupancy_map[ix][iy] = 1.0  # occupied area 1.0
            occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
            occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
            occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
    # occupancy grid computed with with flood fill
    else:
        occupancy_map = init_flood_fill((center_x, center_y), (ox, oy), (x_w, y_w), (min_x, min_y), xy_resolution)
        flood_fill((center_x, center_y), occupancy_map)
        occupancy_map = np.array(occupancy_map, dtype=float)
        for (x, y) in zip(ox, oy):
            ix = int(round((x - min_x) / xy_resolution))
            iy = int(round((y - min_y) / xy_resolution))
            occupancy_map[ix][iy] = 1.0  # occupied area 1.0
            occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
            occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
            occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
    occupancy_map[center_x][center_y] = 2
    x_size = min( occupancy_map.shape[0] - center_x, center_x )
    y_size = min( occupancy_map.shape[1] - center_y, center_y )
    x_size = y_size = min( x_size, y_size )
    print( "10086", center_x, center_y )
    occupancy_map = occupancy_map[::-1,:]
    occupancy_map = occupancy_map.transpose( 1, 0 )
    # occupancy_map = occupancy_map[ center_x - x_size : center_x + x_size, center_y - y_size : center_y + y_size ]
    return occupancy_map, min_x, max_x, min_y, max_y, xy_resolution

def show( ox, oy, occupancy_map, xy_res ):
    plt.figure(1, figsize=(10, 4))
    plt.subplot(122)
    plt.imshow(occupancy_map, cmap="PiYG_r")
    # cmap = "binary" "PiYG_r" "PiYG_r" "bone" "bone_r" "RdYlGn_r"
    # plt.clim(-0.4, 1.4)
    # plt.gca().set_xticks(np.arange(-.5, xy_res[1], 1), minor=True)
    # plt.gca().set_yticks(np.arange(-.5, xy_res[0], 1), minor=True)
    # plt.grid(True, which="minor", color="w", linewidth=0.6, alpha=0.5)
    # plt.colorbar()
    # plt.subplot(121)
    # plt.plot([oy, np.zeros(np.size(oy))], [ox, np.zeros(np.size(oy))], "ro-")
    # plt.axis("equal")
    # plt.plot(0.0, 0.0, "ob")
    # plt.gca().set_aspect("equal", "box")
    # bottom, top = plt.ylim()  # return the current y-lim
    # plt.ylim((top, bottom))  # rescale y axis, to match the grid orientation
    # plt.grid(True)
    plt.show()


def lidar_to_gird_map( ang, dist ):

    xy_resolution = 0.02  # x-y grid resolution
    ox = np.sin(ang) * dist
    oy = np.cos(ang) * dist
    occupancy_map, min_x, max_x, min_y, max_y, xy_resolution = \
        generate_ray_casting_grid_map(ox, oy, xy_resolution, True)
    xy_res = np.array(occupancy_map).shape
    print( xy_res[0] )
    ''' 
    for i in range( 0, xy_res[0] ):
        for j in range( 0, xy_res[1] ):
            print( occupancy_map[i][j], end = '' )
        print()
    '''
    # occupancy_map = occupancy_map.transpose( 1, 0 )
    show( ox, oy, occupancy_map, xy_res )
    
    return occupancy_map

