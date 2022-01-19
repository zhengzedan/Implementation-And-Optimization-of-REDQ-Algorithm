
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
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        torch.manual_seed(seed)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)
        self.reset_parameters(init_w)

        self.in_channels = 1

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels= self.in_channels, out_channels= 25,kernel_size=1),
            nn.BatchNorm1d(25),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=1, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(25, 50, kernel_size=1),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=1, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(50, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_size)
        )
        
    def set_inChannel(self, in_channels):
        self.in_channels = in_channels

    def reset_parameters(self, init_w):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):
        
        state = state.tolist()
        # print("state:", state)
        x_ = []
        x_.append(state)
        # print("x_:", x_)
        state = torch.tensor(x_)
        # state = state.reshape(25, 1, 1, 3)
        # state.expand(1,1,3)
        # print("state size:", state.size())
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # mu = self.mu(x)

        # log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        x = self.layer1(state)
        # print("x:", x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        mu = x
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        log_std = x

        return mu, log_std
    
    def sample(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob, torch.tanh(mu)
        

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        # self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, 1)
        self.in_channels = 1
        self.fc1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=200, kernel_size=1),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=1, stride=2)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)

        )

        # self.reset_parameters()

    # def reset_parameters(self):
    #     self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    #     self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    #     self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def set_inChannel(self, in_channel):
        self.in_channels = in_channel

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # print("state in critic:", state)
        # print("action in critic:", action)
        x = torch.cat((state, action.t()), dim=1)
        
        # print("x size in Critic:", x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        state = state.tolist()
        # print("state:", state)
        x_ = []
        x_.append(state)
        # print("x_:", x_)
        state = torch.tensor(x_)
        x = self.fc1(state)
        x = self.fc2(x)
        return self.fc3(x)