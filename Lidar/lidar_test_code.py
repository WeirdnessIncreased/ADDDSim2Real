from cmath import pi
from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import numpy as np
import cv2
import lidar_to_grid_map
import Convolution_matching
import lidar_data_mapping

def check_state_read_data( state, ang, dist, info=None ):
    image_data = state["color_image"]
    laser_data = np.array(state["laser"])
    vector_data = state["vector"]
    print("=======================state check====================")
    # laser scan distances from -135 deg to +135 deg, scan angle resolution is 270/(61-1) 
    print("laser shape: {}, max distance: {}, min distance: {}".format(laser_data.shape, np.max(laser_data), np.min(laser_data)))
    print("self pose: {}, self info: {}, enemy active: {}, enemy pose: {}, enemy_info: {}".format(vector_data[0], vector_data[1], vector_data[2], vector_data[3], vector_data[4]))

    # print( vector_data[0][2] )
    for i in range( 0, 61 ):
        #print( (float)( -vector_data[0][2] + ( 135 * ( i - 30 ) / 30 * pi / 180) )  )
        ang.append((float)( pi * 0.5 - vector_data[0][2] + ( 135 * ( i - 30 ) / 30 * pi / 180) ))
        dist.append( (float)(laser_data[i]) )
    #print(vector_data[0])
    print("-----------------------end check---------------------")
    return ang, dist, vector_data[0]

def check_state( state, info=None ):
    image_data = state["color_image"]
    laser_data = np.array(state["laser"])
    vector_data = state["vector"]
    print("=======================state check====================")
    # laser scan distances from -135 deg to +135 deg, scan angle resolution is 270/(61-1) 
    print("laser shape: {}, max distance: {}, min distance: {}".format(laser_data.shape, np.max(laser_data), np.min(laser_data)))
    print("self pose: {}, self info: {}, enemy active: {}, enemy pose: {}, enemy_info: {}".format(vector_data[0], vector_data[1], vector_data[2], vector_data[3], vector_data[4]))

    # print( vector_data[0][2] )
    for i in range( 0, 61 ):
        #print( (float)( -vector_data[0][2] + ( 135 * ( i - 30 ) / 30 * pi / 180) )  )
        ang.append((float)( pi * 0.5 - vector_data[0][2] + ( 135 * ( i - 30 ) / 30 * pi / 180) ))
        dist.append( (float)(laser_data[i]) )
    #print(vector_data[0])
    print("-----------------------end check---------------------")
    return ang, dist, vector_data[0]

env = CogEnvDecoder(env_name="../../mac_v2/cog_sim2real_env.app", no_graphics=False, time_scale=1, worker_id=1) # mac os
#env = CogEnvDecoder(env_name="../mac_v2/cog_sim2real_env.app", no_graphics=False, time_scale=1, worker_id=1)
num_episodes = 1
num_steps_per_episode = 1 # max: 1500
for i in range(num_episodes):
    #every time call the env.reset() will reset the envinronment
    observation = env.reset()
    
    for j in range(num_steps_per_episode):
        ang, dist = [], []
        # action = env.action_space.sample()
        action = [0, 0, 0, 0]  # [vx, vy, vw, fire]; vx: the velocity at which the vehicle moves forward, vy: the velocity at which the vehicle moves to the left, vw: Angular speed of the vehicle counterclockwise rotation, fire: Shoot or not
        obs, reward, done, info = env.step(action)
        #cv2.imshow("color_image", obs["color_image"])
        #cv2.waitKey(1)
        ang, dist, vector_data = check_state_read_data( obs, ang, dist, info )
        occupancy_map = lidar_data_mapping.lidar_to_gird_map( ang, dist )
        obstacle_map = Convolution_matching.get_obstacle(vector_data)
        # result, tx, ty = Convolution_matching.numpy_conv( obstacle_map, occupancy_map )
        # print( np.max(result), np.min(result))
        # print( tx, ty )
        # print(reward, done)
        # action = [0, 0, 0.9, 0]
        # obs, reward, done, info = env.step(action)
        # ang, dist, vector_data = check_state_read_data( obs, ang, dist, info )
        # occupancy_map = lidar_data_mapping.lidar_to_gird_map( ang, dist )
