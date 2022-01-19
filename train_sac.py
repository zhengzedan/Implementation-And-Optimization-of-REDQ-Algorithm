
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

from Regular_SAC.regular_SAC import Agent

# import pybullet
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

def train(n_episodes=200, print_every=10):
    scores_deque = deque(maxlen=100)
    average_100_scores = []
    scores = []
    for i_episode in range(1, n_episodes+1):
        # print("i_episode:", i_episode)
        state = env.reset()
        state = state.reshape((1, state_size))
        score = 0
        while True:
            
            action = agent.act(state)
            action_v = action.numpy()
            action_v = np.clip(action_v, action_low, action_high)
            next_state, reward, done, info = env.step(action_v)
            next_state = next_state.reshape((1, state_size))
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break 

        scores_deque.append(score)
        scores.append(score)
        average_100_scores.append(np.mean(scores_deque))

        print("mean:", np.mean(scores_deque))
        
        # print('\rEpisode {:.2f} Reward: {:.2f}  Average100 Score:'.format(str(i_episode), str(score)), end="")
        print('\rEpisode:', i_episode)
        print('\rReward:', score)
        print("Average100 Score: ", np.mean(scores_deque))
        if i_episode % print_every == 0:
            # print('\rEpisode {:.2f}  Reward: {:.2f}  Average100 Score:'.format((i_episode), (score)))
            print('\rrEpisode:', i_episode)
            print('\rReward:', score)
            print("Average100 Score: ", np.mean(scores_deque))
            
    return scores

# sac_scores = train(n_episodes=ep)
# t1 = time.time()
# env.close()
# print("training took {} min!".format((t1-t0)/60))