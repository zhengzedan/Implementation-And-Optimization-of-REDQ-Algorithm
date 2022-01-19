import numpy as np
import random

import gym
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal

import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter
import argparse

from Networks.networks import Actor, Critic
from train_sac import train
from Regular_SAC.regular_SAC import Agent
from train_redq import REDQ_Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""训练SAC"""
import gym
env_name = "Pendulum-v1" #"HalfCheetahPyBulletEnv-v0"#"Pendulum-v0"
seed = 0
#Hyperparameter
lr = 3e-4
buffer_size = int(1e6)
batch_size = 1
tau = 0.005
gamma = 0.99

#writer = SummaryWriter("runs/"+args.info)
env = gym.make(env_name)
action_high = env.action_space.high[0]
action_low = env.action_space.low[0]
torch.manual_seed(seed)
env.seed(seed)
np.random.seed(seed)
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = Agent(state_size=state_size,
              action_size=action_size,
              random_seed=seed,
              action_prior="uniform")

t0 = time.time()


ep = 100
sac_scores = train(n_episodes=ep)
t1 = time.time()
env.close()
print("training took {} min!".format((t1-t0)/60))

"""训练结果绘图"""
import matplotlib.pyplot as plt
plt.plot(sac_scores)

plt.xlabel("Episodes")
plt.ylabel("Reward")

"""训练redq"""
# import pybulletgym
import gym
# import pybullet
# import pybullet_envs
env_name = "Pendulum-v1" #"HalfCheetahPyBulletEnv-v0"#"Pendulum-v0"
# env_name = "Ant-v2"
ep = 10000000
seed = 1
#Hyperparameter
lr = 3e-4
buffer_size = int(1e6)
batch_size = 1
tau = 0.005
gamma = 0.99

# RED-Q Parameter
N = 5
M = 2
G = 5

#writer = SummaryWriter("runs/"+args.info)
env = gym.make(env_name)
action_high = env.action_space.high[0]
action_low = env.action_space.low[0]
torch.manual_seed(seed)
env.seed(seed)
np.random.seed(seed)
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = REDQ_Agent(state_size=state_size,
              action_size=action_size,
              random_seed=seed,
              action_prior="uniform", N=N, M=M, G=G)

t0 = time.time()
scores = train(n_episodes=ep)
t1 = time.time()
env.close()
print("training took {} min!".format((t1-t0)/60))

"""结果对比绘图"""
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.title("Pendulum-v1")
plt.plot(scores, label="REDQ")
#plt.plot(sac_ensemble, color="r", label="SAC_Ensemble")
plt.plot(sac_scores, label="SAC")

plt.legend(loc=4)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.savefig("Pendulum_REDQ_5-2-5.jpg")