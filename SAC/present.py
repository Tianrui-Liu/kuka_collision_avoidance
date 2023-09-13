import time

from agent import SacAgent
import gym
import pybullet_envs
import argparse
import csv
import numpy as np
import torch
# from path_planning.envs.path_planning_kuka import Kukaenv
import path_planning
import GPUtil

act_low = np.array(
    [-0.96705972839, -2.09439510239, -2.96705972839, 0.19439510239, -2.96705972839, -2.09439510239, -3.05432619099])
act_high = np.array(
    [0.96705972839, 2.09439510239, 2.96705972839, 2.29439510239, 2.96705972839, 2.09439510239,
     3.05432619099])
offset = (act_high[3] + act_low[3]) / 2
scaling = (act_high[3] - act_low[3]) / 2
# rate = act_high[3] - offset
bound = act_high
bound[3] = scaling


def test(env, agent, mean=True):
    # env = gym.make(env_list[environment])
    # env = gym.make(environment)
    # print(env)
    while True:
        step = 0
        state = env.reset()
        # print(state)
        total_reward = 0
        while step < 300:
            step += 1
            action = agent.act(state, mean=mean)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            # time.sleep(0.2)
            if done:
                print("reward:", total_reward)
                time.sleep(3)
                break
    # return total_reward


if __name__ == '__main__':
    environment = 'Path_planning-v0'
    env = gym.make(environment)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # automatically choose the least utilized gpu, maybe the available one
    gpus = GPUtil.getGPUs()
    least_utilized_gpu = min(gpus, key=lambda gpu: gpu.load)
    gpu = least_utilized_gpu.id
    print("Using gpu :", gpu)
    print("——————————————————————————————————")

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=gpu)
    args = parser.parse_args()

    device = args.gpu

    # learn(device=args.gpu, environment=args.env, log=args.log)

    agent = SacAgent(state_dim, action_dim, act_bound=bound, offset=offset, device=device)
    agent.actor.load_state_dict(torch.load('612173.pth'))
    # agent.actor.load_state_dict(torch.load('1285521.pth'))

    # agent.actor.load_state_dict(torch.load('Path_planning-v01931635.pth'))
    test(env, agent, mean=False)
