from cmath import pi
from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import numpy as np
import cv2
import lidar_data_mapping

def check_state( state, info=None ):
    image_data = state["color_image"]
    laser_data = np.array(state["laser"])
    vector_data = state["vector"]
    print("=======================state check====================")
    # laser scan distances from -135 deg to +135 deg, scan angle resolution is 270/(61-1) 
    print("laser shape: {}, max distance: {}, min distance: {}".format(laser_data.shape, np.max(laser_data), np.min(laser_data)))
    print("self pose: {}, self info: {}, enemy active: {}, enemy pose: {}, enemy_info: {}".format(vector_data[0], vector_data[1], vector_data[2], vector_data[3], vector_data[4]))
    x, y = lidar_data_mapping.lidar_mapping( vector_data, laser_data )
    print("-----------------------end check---------------------")

env = CogEnvDecoder(env_name="../../mac_v2/cog_sim2real_env.app", no_graphics=False, time_scale=1, worker_id=1) # mac os
#env = CogEnvDecoder(env_name="../mac_v2/cog_sim2real_env.app", no_graphics=False, time_scale=1, worker_id=1)
num_episodes = 1
num_steps_per_episode = 1 # max: 1500
for i in range(num_episodes):
    #every time call the env.reset() will reset the envinronment
    observation = env.reset()
    
    for j in range(num_steps_per_episode):
        # action = env.action_space.sample()
        action = [0, 0, 0, 0]  # [vx, vy, vw, fire]; vx: the velocity at which the vehicle moves forward, vy: the velocity at which the vehicle moves to the left, vw: Angular speed of the vehicle counterclockwise rotation, fire: Shoot or not
        obs, reward, done, info = env.step(action)
        #cv2.imshow("color_image", obs["color_image"])
        #cv2.waitKey(1)
        check_state( obs, info )