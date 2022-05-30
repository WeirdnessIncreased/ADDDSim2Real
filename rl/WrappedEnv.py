import numpy as np
import math
import gym
from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import itertools

class RobotEnv(gym.Env):
    def __init__(self, env_name, time_scale=1, worker_id=1):
        self.time_scale = time_scale
        self.worker_id = worker_id
        self.cog_env = CogEnvDecoder(env_name=env_name, 
                                     no_graphics=False, 
                                     time_scale=self.time_scale, 
                                     worker_id=self.worker_id)

        state_high = np.ones(61 + 28)
        state_high[:61] *= 100
        state_high[61:] *= 10
        state_low = np.zeros(61 + 28)
        state_high[-1] = 181
        self.observation_space = gym.spaces.Box(state_high, state_low, dtype=np.float32)

        # Action may need remapping
        self.action_bound = np.array([2, 2, np.pi / 4, 1])
        self.action_space = gym.spaces.Box(np.ones(4), -np.ones(4), dtype=np.float32)
        
    def step(self, action):
        action *= self.action_bound # Rescale it
        if action[-1] > 0:
            action[-1] = 1
        else:
            action[-1] = 0
        
        _obs, reward, done, _info = self.cog_env.step(action)
        reward = self.calc_rewards_from_state(_obs['vector'], action)

        obs = np.zeros(61 + 28)
        vec_state = _obs['vector']
        vec_state[2] = [vec_state[2]]
        obs[:61] = _obs['laser']
        obs[61:] = np.array(list(itertools.chain.from_iterable(vec_state)))
        
        info = _info[0]
        info['judge_result'] = _info[1]
        return obs, reward, done, info

    def calc_rewards_from_state(self, obs, action):
        adv_r = (800 - obs[4][0]) + obs[1][0] # enemy HP, self HP
        adv_r *= 0.1
        distance_control = -1000000 * (np.sqrt(math.hypot(obs[0][0] - obs[3][0], obs[0][1] - obs[3][1])) - 2) ** 2
        # angle_control = -1000000 * abs(action[2])
        sx, sy, gx, gy = obs[0][0], obs[0][1], obs[3][0], obs[3][1]
        s_g_theta = np.arctan((gy - sy) / (gx - sx))
        if np.tan((gy - sy) / (gx - sx)) * (gy - sy) < 0:
            s_g_theta = s_g_theta + math.pi
        s_theta = obs[0][2]
        angle_control = -10000 * abs(s_theta - s_g_theta)
        # Time reward is tricker since if set to aggressive
        # It will encourage robot to suicide
        col_r = -100 * obs[-1][1] # collision time
        return adv_r + col_r + distance_control 

    def reset(self):
        _obs = self.cog_env.reset()
        obs = np.zeros(61 + 28)
        vec_state = _obs['vector']
        vec_state[2] = [vec_state[2]]
        obs[:61] = _obs['laser']
        obs[61:] = np.array(list(itertools.chain.from_iterable(vec_state)))
        return obs

    def render(self, mode="human"):
        self.cog_env.render(mode)

    def close(self):
        self.cog_env.close()

