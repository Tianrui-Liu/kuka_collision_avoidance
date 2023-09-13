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


def opt_cuda(t, device):
    if torch.cuda.is_available():
        cuda = "cuda:" + str(device)
        return t.cuda(cuda)
    else:
        return t


# def learn(device=0, environment=0, log=1):
def learn(device, environment='Path_planning-v0', log=1):
    env = gym.make(environment)
    log_dir = 'saves/' + str(environment) + '/log' + str(log) + '.csv'
    # log_dir = 'saves/kuka' + '/log' + str(log) + '.csv'
    with open(log_dir, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frames', 'return'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SacAgent(state_dim, action_dim, act_bound=bound, offset=offset, device=device)
    total_frames = 0
    # pretrained_paths = np.load('pretrained_path.npy')
    # pretrain_len=len(pretrained_paths[0])
    # while total_frames < pretrain_len:
    #     state = pretrained_paths[0][total_frames][0:24]
    #     next_state = pretrained_paths[0][total_frames][24:48]
    #     action = pretrained_paths[0][total_frames][48:55]
    #     reward = pretrained_paths[0][total_frames][55]
    #     done = pretrained_paths[0][total_frames][56]
    #     agent.remember(state, next_state, action, reward, done)
    #     total_frames += 1

    # prev_frame = pretrain_len
    # for warm up
    threshold = 500*10
    maxstep = 2e3
    while 1:
        state = env.reset()
        frame = 0
        datas=[]
        if not threshold:
            np.save("pretrain_path_with_collision",datas)
            return datas

        while threshold:
            action = env.action_space.sample()
            # else:
                # action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            frame += 1
            # total_frames += 1
            # agent.remember(state, next_state, action, reward, done and frame < maxstep)

            # if total_frames > threshold:
            #     agent.train()
            state = next_state
            if frame>maxstep:
                break

            if done:
                # print(state)
                # print(action)
                # print(reward)
                # print(done)
                data=np.concatenate((state,next_state,action,[reward],[int(done)]))
                print(threshold,"counting down: ",data)
                assert data.shape[0]==57
                datas.append(data)
                threshold-=1
                break


                # if total_frames >= threshold*10:
                #     with open(log_dir, "a+", newline='') as csvfile:
                #         writer = csv.writer(csvfile)
                #         testres=test(env, agent)
                #         writer.writerow([total_frames - threshold, testres])
                #     # reach or close to the target
                #     if testres>1000 or testres==-100:
                #         torch.save(agent.actor.cpu().state_dict(), "saves/" + str(total_frames - threshold) + ".pth")
                #         agent.actor = opt_cuda(agent.actor, device)
                # print('frames:', total_frames, 'done.   Interval:', total_frames - prev_frame)
                # print("————————————————————————————————————————————————————")
                # prev_frame = total_frames
                # break


def test(env, agent):
    # env = gym.make(env_list[environment])
    # env = gym.make(environment)
    # print(env)
    state = env.reset()
    # print(state)
    total_reward = 0
    step = 0
    while step < 3000:
        step += 1
        if (step == 2999):
            print("test failed")
        action = agent.act(state, mean=True)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        if done:
            break
    return total_reward


if __name__ == '__main__':
    # env_list = ['HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0',
    #             'AntBulletEnv-v0', 'HumanoidBulletEnv-v0']
    environment = 'Path_planning-v0'

    # automatically choose the least utilized gpu, maybe the available one  
    gpus = GPUtil.getGPUs()
    least_utilized_gpu = min(gpus, key=lambda gpu: gpu.load)
    gpu = least_utilized_gpu.id
    print("Using gpu :", gpu)
    print("——————————————————————————————————")

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=gpu)
    # parser.add_argument('-e', '--env', type=int, default=0)
    parser.add_argument('-l', '--log', type=int, default=1)
    args = parser.parse_args()

    # learn(device=args.gpu, environment=args.env, log=args.log)
    datas=learn(device=args.gpu, environment=environment, log=args.log)
    print(datas.shape)
