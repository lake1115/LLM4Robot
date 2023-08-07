#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   comm_net.py
@Time    :   2023/08/03 16:06:50
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import numpy as np
class Comm_Net(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()

        self.embedding_size = 64

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space)
        )
        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, obs, mask, call, skill):
        if obs['obj_start_gps_compass'][:,0] < 1.5 and -0.5 < obs['obj_start_gps_compass'][:,1] < 0.5 and skill==3 and obs['is_holding'][:,0] == 0:
            return torch.Tensor([True])
        elif obs['obj_goal_gps_compass'][:,0] < 1.5 and -0.5 < obs['obj_goal_gps_compass'][:,1] < 0.5 and skill==3 and obs['is_holding'][:,0] == 1:
            return torch.Tensor([True])
        else:
            return call | (~mask).view(-1)
        
        
    def hard_code_forward(self, obs, mask, call, skill):
        if obs['obj_start_gps_compass'][:,0] < 1.5 and -0.5 < obs['obj_start_gps_compass'][:,1] < 0.5 and skill==3 and obs['is_holding'][:,0] == 0:
            return torch.Tensor([True])
        elif obs['obj_goal_gps_compass'][:,0] < 1.5 and -0.5 < obs['obj_goal_gps_compass'][:,1] < 0.5 and skill==3 and obs['is_holding'][:,0] == 1:
            return torch.Tensor([True])
        else:
            return call | (~mask).view(-1)
      

if __name__ == '__main__':
    pass

