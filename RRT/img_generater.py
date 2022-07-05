import cv2
import numpy as np

def main():
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
    obstacle_map = np.full( ( 405, 225 ), 255, dtype = np.uint8 )

    for pos in obstacles:
        for x in np.arange(pos[0], pos[2]):
            for y in np.arange(pos[1], pos[3]):    
                obstacle_map[ (int)( x ), (int)( y ) ] = 0

    save_path = "1.png"
    cv2.imwrite( save_path, obstacle_map )

main()