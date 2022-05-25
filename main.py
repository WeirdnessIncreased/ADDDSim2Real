from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import numpy as np
import PathPlanner as PathPlanner
import cv2

fixed_obstacle = {
'B1': [ 7.08, 1.00, 8.08, 1.2],
'B2': [ 5.78, 2.14 ,6.58 , 2.34], 
'B3': [ 6.58, 3.48, 6.78, 4.48],
'B4': [ 3.54, 0.935, 4.45 , 1.135], 
'B5': [ 3.864, 2.064, 4.216, 2.416], 
'B6': [ 3.54, 3.345, 4.45 , 3.545],
'B7': [ 1.5, 0, 1.7, 1],
'B8': [ 1.5, 2.14, 2.3, 2.34], 
'B9': [ 0, 3.28, 1, 3.48]
}

map_size = [8.08, 4.48]

def check_state(state, info=None):
    image_data = state["color_image"]
    laser_data = np.array(state["laser"])
    vector_data = state["vector"]
    # laser scan distances from -135 deg to +135 deg, scan angle resolution is 270/(61-1) 
    print("laser shape: {}, max distance: {}, min distance: {}".format(laser_data.shape, np.max(laser_data), np.min(laser_data)))
    # self_pose: [x, y, theta(rad)], self_info: [remaining HP, remaining bullet]
    # enemy_pose: [x, y, theta(rad)], enemy_info: [remaining HP, remaining bullet]
    print("self pose: {}, self info: {}, enemy active: {}, enemy pose: {}, enemy_info: {}".format(vector_data[0], vector_data[1], vector_data[2], vector_data[3], vector_data[4]))
    # goal_x: [x, y, is_activated?]
    print("goal 1: {}, goal 2: {}, goal 3: {}, goal 4: {}, goal 5:{}".format(vector_data[5], vector_data[6], vector_data[7], vector_data[8], vector_data[9]))
    # total counts of collisions, total collision time
    print("total collisions: {}, total collision time: {} ".format(vector_data[10][0], vector_data[10][1]))
    if info is not None:
        print("Number of goals have been activated: {}".format(info[1][3]))
        # attack damage is blue one caused damage to red one
        print("time taken: {}, attack damage: {}, score: {}".format(info[1][1], info[1][2], info[1][0]))
    print("-----------------------end check---------------------")


num_episodes = 10
num_steps_per_episode = 500
env = CogEnvDecoder(env_name="linux_v2/cog_sim2real_env.x86_64", no_graphics=False, time_scale=1, worker_id=1) # linux os

for i in range(num_episodes):
    observation = env.reset()
    for j in range(num_steps_per_episode):
        action = [0.5, 0.5, 0.1, 0]
        obs, reward, done, info = env.step(action)
        break;
        cv2.imshow("color_image", obs["color_image"])
        cv2.waitKey(1)
        check_state(obs, info)
        print(reward, done)
    break

ox1, oy1, ox2, oy2 = [], [], [], []

for name in fixed_obstacle:
    ox1.append(fixed_obstacle[name][0])
    oy1.append(fixed_obstacle[name][1])
    ox2.append(fixed_obstacle[name][2])
    oy2.append(fixed_obstacle[name][3])

vector_data = obs["vector"]
dynamic_obstacles = [vector_data[5][:2], vector_data[6][:2], vector_data[7][:2], vector_data[8][:2], vector_data[9][:2]]

for ob in dynamic_obstacles:
    x = ob[0]
    y = ob[1]
    ox1.append(x - 0.15)
    oy1.append(y - 0.15)
    ox2.append(x + 0.15)
    oy2.append(y + 0.15)

sx, sy = vector_data[0][0], vector_data[0][1]
gx, gy = vector_data[5][0], vector_data[5][1]
gy -= 0.5
mx, my = map_size[0], map_size[1]
obstacles = list(zip(ox1, oy1, ox2, oy2))
#print("          ",vector_data[5])
PathPlanner.get_path(sx, sy, gx, gy, mx, my, obstacles)
PathPlanner.get_path(gx, gy, vector_data[6][0], vector_data[6][1] - 0.5, mx, my, obstacles)
