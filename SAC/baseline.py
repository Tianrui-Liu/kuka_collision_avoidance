import gym
import numpy as np

import pybullet_envs
import argparse
import csv
import numpy as np
import torch

import path_planning
import GPUtil

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

env = gym.make('Path_planning-v0')

model = SAC("MlpPolicy", env,tensorboard_log="3e5", learning_rate=3e-4, verbose=1,device=6)
# eval_callback = EvalCallback(env, best_model_save_path='./logs/',
#                              log_path='./logs/', eval_freq=1000,
#                              deterministic=True, render=False)
model.learn(total_timesteps=300000,  log_interval=4,progress_bar=True)
model.save("3e5")

# del model # remove to demonstrate saving and loading

# model = SAC.load("sac_pendulum")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()