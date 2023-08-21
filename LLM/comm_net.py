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


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__()
        self.Conv_BN = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.DownSample = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        out = self.Conv_BN(x)
        out = self.DownSample(out)
        return out
    
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class Comm_Net(nn.Module):
    def __init__(self, obs_space, action_space, delay):
        super().__init__()

        self.delay = delay
        img_dim = obs_space['head_depth'].shape[0] * obs_space['head_depth'].shape[1] 
        self.pick_skill_dim = obs_space['ee_pos'].shape[0] + obs_space['joint'].shape[0] + obs_space['relative_resting_position'].shape[0]  
        self.place_skill_dim = obs_space['ee_pos'].shape[0] + obs_space['joint'].shape[0] + obs_space['relative_resting_position'].shape[0] 
        self.nav_skill_dim = (obs_space['obj_start_gps_compass'].shape[0] + obs_space['obj_start_sensor'].shape[0]) * (self.delay-1)
        self.embedding_size = 64
        # Define image embedding  output [batch, 64, 16, 16]
        self.cnn_encode = nn.Sequential(
            nn.Conv2d(self.delay, 16, kernel_size=7, stride=2, padding=3,bias=False),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            DownSampleBlock(16, 32),
            DownSampleBlock(32, 64),   
            Reshape(64*img_dim//(4**4)),
            nn.Linear(64*img_dim//(4**4), self.embedding_size)
        )
        # Define sensor embedding for pick
        self.mlp_encode0 = nn.Linear(self.pick_skill_dim, self.embedding_size)
        # Define sensor embedding for place
        self.mlp_encode1 = nn.Linear(self.place_skill_dim, self.embedding_size)
        # Define sensor embedding for nav
        self.mlp_encode2 = nn.Linear(self.nav_skill_dim, self.embedding_size)
        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size*2, 32),
            nn.Tanh(),
            nn.Linear(32, action_space)
        )
        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size*2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def obs_to_embedding(self, batch_obs, skills):
        cnn_emb = []
        mlp_emb = []
        for k in range(len(batch_obs)):
            his_obs = batch_obs[k]
            skill = skills[k]
            imag = torch.cat([his_obs[i]['head_depth'] for i in range(len(his_obs))],dim=-1).transpose(1, 3)
            cnn_emb.append(self.cnn_encode(imag))
            if skill == 0:
                seq_ee_pos = his_obs[-1]['ee_pos'] 
                seq_joint = his_obs[-1]['joint']
                seq_relative_resting_pos = his_obs[-1]['relative_resting_position']
                data = torch.cat([seq_ee_pos,seq_joint, seq_relative_resting_pos], dim=1)
                mlp_emb.append(self.mlp_encode0(data))
            elif skill == 1:
                seq_ee_pos = his_obs[-1]['ee_pos'] 
                seq_joint = his_obs[-1]['joint']
                seq_relative_resting_pos = his_obs[-1]['relative_resting_position']
                data = torch.cat([seq_ee_pos,seq_joint, seq_relative_resting_pos], dim=1)
                mlp_emb.append(self.mlp_encode1(data))
            elif skill == 3:
                delta_start_gps = torch.cat([his_obs[i]['obj_start_gps_compass'] -his_obs[0]['obj_start_gps_compass'] for i in range(1,len(his_obs))], dim=-1)
                delta_start_sensor = torch.cat([his_obs[i]['obj_start_sensor'] -his_obs[0]['obj_start_sensor'] for i in range(1,len(his_obs))],dim=-1)
                data = torch.cat([delta_start_gps,delta_start_sensor], dim=1)
                mlp_emb.append(self.mlp_encode2(data))
            elif skill == 4:
                delta_goal_gps = torch.cat([his_obs[i]['obj_goal_gps_compass'] -his_obs[0]['obj_goal_gps_compass'] for i in range(1,len(his_obs))], dim=-1)
                delta_goal_sensor = torch.cat([his_obs[i]['obj_goal_sensor'] -his_obs[0]['obj_goal_sensor'] for i in range(1,len(his_obs))], dim=-1)
                data = torch.cat([delta_goal_gps,delta_goal_sensor], dim=1)
                mlp_emb.append(self.mlp_encode2(data))
            else:
                assert "illegal skill"
        cnn_emb = torch.cat(cnn_emb)
        mlp_emb = torch.cat(mlp_emb)
        
        return cnn_emb, mlp_emb


    def forward(self, batch_obs, skills):

        cnn_emb, mlp_emb = self.obs_to_embedding(batch_obs, skills)

        embedding = torch.cat([cnn_emb,mlp_emb],dim=1)
        act = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(act, dim=1))
        val = self.critic(embedding)
        value = val.squeeze(1)
        
        return dist, value
        
    # def forward(self, his_obs, skill):
   
    #     obs = his_obs[0]
    #     delta_depth_img = obs[0]['head_depth'] - obs[-1]['head_depth'] 
    #     delta_start_gps = obs[0]['obj_start_gps_compass'] - obs[-1]['obj_start_gps_compass']
    #     delta_goal_gps = obs[0]['obj_goal_gps_compass'] - obs[-1]['obj_goal_gps_compass']
    #     if  obs[-1]['is_holding'][:,0] == 0:
    #         if obs[-1]['obj_start_gps_compass'][:,0] < 1 and -0.5 < obs[-1]['obj_start_gps_compass'][:,1] < 0.5 and skill==3:
    #             return torch.tensor([True]), None,None
    #         elif torch.norm(delta_start_gps) < 0.1 and skill==3:
    #             return torch.tensor([True]), None,None
    #         else:
    #             return torch.tensor([False]), None,None
    #     elif  obs[-1]['is_holding'][:,0] == 1:
    #         if obs[-1]['obj_goal_gps_compass'][:,0] < 1 and -0.5 < obs[-1]['obj_goal_gps_compass'][:,1] < 0.5 and skill==4:
    #             return torch.tensor([True]), None,None
    #         elif torch.norm(delta_goal_gps) < 0.1 and skill==4:
    #             return torch.tensor([True]), None,None
    #         else:
    #             return torch.tensor([False]), None,None






        


if __name__ == '__main__':
    pass

