from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import numpy as np
from agent import Agent

# env = CogEnvDecoder(env_name="mac_confrontation_v2/cog_confrontation_env.app", no_graphics=False, time_scale=1, worker_id=1) 
env = CogEnvDecoder(env_name="../mac_v2/cog_sim2real_env.app", no_graphics=False, time_scale=1, worker_id=2) 
# env = CogEnvDecoder(env_name="../linux_v3.0/cog_sim2real_env.x86_64", no_graphics=False, 
#                     time_scale=1, worker_id=2, seed=19260817, force_sync=True) # linux os

num_eval_episodes = 50

eval_agent = Agent(model_path="")

activated_goals_analy = []
time_token_analy = []
attack_damage_analy = []
score_analy = []

np.random.seed(19260817)

for i in range(num_eval_episodes):

    bias_x = np.random.uniform(-0.5, 0.5, 1)[0]
    bias_y = np.random.uniform(-0.5, 0.5, 1)[0]
    # bias_x = 0.1

    obs = env.reset()
    # if i <= 4: continue
    done = False
    info = None

    while not done:
        print('episode', i)
        print('info', info)
        print('vector', obs['vector'])
        obs["vector"][0][0] += bias_x
        obs["vector"][0][1] += bias_y
        print('noisy coordinate', obs['vector'][0])
        print("error:", bias_x, bias_y )
        # if info != None and info[1][3] == 5: break # attention
        action = eval_agent.agent_control(obs=obs, done=done, info=info)
        obs, reward, done, info = env.step(action)

    num_activted_goals = info[1][3]
    activated_goals_analy.append(num_activted_goals)
    time_token = info[1][1]
    time_token_analy.append(time_token)
    attack_damage = info[1][2]
    attack_damage_analy.append(attack_damage)
    score = info[1][0]
    if score <= -1000:
        print('found error at step', i)
        break
    score_analy.append(score)
    print('current mean score', np.mean(score))

mean_activated_goal = np.mean(activated_goals_analy)
mean_time_token = np.mean(time_token_analy)
mean_attack_damage = np.mean(attack_damage_analy)
mean_score = np.mean(score_analy)
print("mean activated goal: {}, mean time token: {}, mean attack damage: {}, mean score: {}".format(
    mean_activated_goal, mean_time_token, mean_attack_damage, mean_score))
