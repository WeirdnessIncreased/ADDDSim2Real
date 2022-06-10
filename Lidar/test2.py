from cmath import pi
from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import numpy as np
import cv2
import lidar_data_mapping_2 as lidar_data_mapping_2

m_x = []
m_y = []

def check_state( state, info=None ):
    laser_data = np.array(state["laser"])
    vector_data = state["vector"]
    # print("=======================state check====================")
    print("self pose: {}, self info: {}, enemy active: {}, enemy pose: {}, enemy_info: {}".format(vector_data[0], vector_data[1], vector_data[2], vector_data[3], vector_data[4]))
    x, y = lidar_data_mapping_2.lidar_mapping( vector_data, laser_data )
    print( "heihei", vector_data[0][0], vector_data[0][1], x, y )
    if( abs(vector_data[0][0] - x) > 0.6 or abs(vector_data[0][1] - y) > 0.6 ):
        pass
    else:
        m_x.append( vector_data[0][0] - x )
        m_y.append( vector_data[0][1] - y )
        print( "mean", np.mean(m_x), np.mean(m_y) )
    # print("-----------------------end check---------------------")

env = CogEnvDecoder(env_name="../../mac_v2/cog_sim2real_env.app", no_graphics=False, time_scale=1, worker_id=1) # mac os
#env = CogEnvDecoder(env_name="../mac_v2/cog_sim2real_env.app", no_graphics=False, time_scale=1, worker_id=1)
num_episodes = 1
num_steps_per_episode = 1 # max: 1500
for i in range(num_episodes):
    #every time call the env.reset() will reset the envinronment
    obs = env.reset()
    vector_data = obs["vector"]
    dynamic_obstacles = [vector_data[5][:2], vector_data[6][:2], vector_data[7][:2], vector_data[8][:2], vector_data[9][:2]]
    lidar_data_mapping_2.update( dynamic_obstacles )
    for j in range(num_steps_per_episode):
        # action = env.action_space.sample()
        action = [ 0, 0, 0, 0 ]  # [vx, vy, vw, fire]; vx: the velocity at which the vehicle moves forward, vy: the velocity at which the vehicle moves to the left, vw: Angular speed of the vehicle counterclockwise rotation, fire: Shoot or not
        obs, reward, done, info = env.step(action)
        #cv2.imshow("color_image", obs["color_image"])
        #cv2.waitKey(1)
        check_state( obs, info )
