#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.utils.common import get_action_space_info
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from habitat_baselines.common.tensor_dict import DictTree, TensorDict
from einops import rearrange
@baseline_registry.register_storage
class KickstartingStorage(RolloutStorage):
    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        is_double_buffered: bool = False,
    ):

        super().__init__(
            numsteps,
            num_envs,
            observation_space,
            action_space,
            recurrent_hidden_state_size,
            num_recurrent_layers,
            is_double_buffered,
        )
        action_shape, discrete_actions = get_action_space_info(action_space)
        if action_shape is None:
            action_shape = action_space.shape
        self.buffers["teacher_actions_data"] = torch.zeros(numsteps + 1, num_envs, 22) ## 22 for mean and std
        self.buffers["teacher_skills"] = torch.zeros(numsteps + 1, num_envs, dtype=torch.int8)

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
        teacher_action_data=None,
        teacher_skills=None,
        **kwargs,
    ):
        if not self.is_double_buffered:
            assert buffer_index == 0

        next_step = dict(
            observations=next_observations,
            recurrent_hidden_states=next_recurrent_hidden_states,
            prev_actions=actions,
            masks=next_masks,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
            teacher_actions_data=teacher_action_data,
            teacher_skills=teacher_skills,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        if len(next_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index] + 1, env_slice),
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index], env_slice),
                current_step,
                strict=False,
            )

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Pretrain_RolloutBuffer
@Time    :   2023/10/27 15:01:39
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
class Pretrain_RolloutBuffer:

    def __init__(self, traj_size=10, step = 1024, n_envs = 1, ppo_epoch = 1):
        self.traj_size = traj_size
        self.obs  = [None for _ in range(self.traj_size)]
        self.action = [None for _ in range(self.traj_size)]
        self.action_dist = [None for _ in range(self.traj_size)]
        self.skill = [None for _ in range(self.traj_size)]
        self.rewards = [None for _ in range(self.traj_size)]
        self.returns = [None for _ in range(self.traj_size)]
        self.mask = [None for _ in range(self.traj_size)]
        self.step = step
        self.n_envs = n_envs
        self.ppo_epoch = ppo_epoch

        self.ptr = 0
        self.full = False
        self.traj_len = 0

        self.gamma = 0.99
        self.heads_num = 4
        self.span = 10
    def __len__(self):
        return self.ptr
    
    def reset(self):
        self.obs  = [None for _ in range(self.traj_size)]
        self.action = [None for _ in range(self.traj_size)]
        self.action_dist = [None for _ in range(self.traj_size)]
        self.skill = [None for _ in range(self.traj_size)]
        self.rewards = [None for _ in range(self.traj_size)]
        self.mask = [None for _ in range(self.traj_size)]
        self.returns = [None for _ in range(self.traj_size)]
        self.ptr = 0
        self.full = False
        self.traj_len = 0

    def compute_returns(self, rewards):
        traj_len = rewards.shape[0]
        returns = torch.zeros((traj_len+1,1),device=rewards.device)

        for step in reversed(range(traj_len)):
            returns[step] = (
                self.gamma
                * returns[step + 1]
                + rewards[step]
            )
        return returns[:traj_len]      

    def add(self, obs, action, action_dist, mask, rewards, cur_skill):
        
        traj, traj_env = torch.where(mask[:,:,0] == False)
        for i in range(self.n_envs):
            traj_idx = traj[torch.where(traj_env==i)[0]]
            for k in range(len(traj_idx)-1):
                self.obs[self.ptr]  = obs[traj_idx[k]:traj_idx[k+1],i]
                self.action[self.ptr] = action[traj_idx[k]:traj_idx[k+1],i]
                self.action_dist[self.ptr] = action_dist[traj_idx[k]:traj_idx[k+1],i]
                self.skill[self.ptr] = cur_skill[traj_idx[k]:traj_idx[k+1],i]
                self.rewards[self.ptr] = rewards[traj_idx[k]:traj_idx[k+1],i]
                self.mask[self.ptr] = mask[traj_idx[k]:traj_idx[k+1],i]
                self.returns[self.ptr] = self.compute_returns(rewards[traj_idx[k]:traj_idx[k+1],i])
                self.traj_len += traj_idx[k+1] - traj_idx[k]
                self.ptr += 1
                if self.ptr >= self.traj_size:
                    self.full = True
                    break
            if self.full:
                break



    def get_skill_weight(self, skill,mask, smooth=True):
        weight = torch.zeros(skill.shape[0],self.n_envs, self.heads_num)
        for t in range(self.n_envs):

                #idx_tensor = torch.cat(skill_list)
                idx_tensor = skill[:,t]
                for i in range(idx_tensor.shape[0]):
                    weight[i,t] = torch.eye(self.heads_num)[idx_tensor[i].item()]
                  
        return weight

    def get(self, smooth=True):
        obs = TensorDict()
        for k in self.obs[0].keys():
            key_data = []
            for i in range(self.traj_size):
                key_data.append(self.obs[i][k])
            obs[k] = torch.cat(key_data)
            
        
        action = torch.cat(self.action_dist) ## combine env  
        returns = torch.cat(self.returns)
        mask = torch.cat(self.mask) 
        skill = torch.cat(self.skill)
        #skill = torch.flatten(torch.transpose(torch.cat(self.skill),0,1),0,1).squeeze()
        skill_weight = torch.eye(self.heads_num)[skill.cpu().int()]
        # skill_weight = torch.flatten(skill_weight,0,1)
        # mask = torch.flatten(mask,0,1)
        return (obs, 
                action,
                skill_weight,
                returns,
                mask,
        )


    def sample(self, batch_size=256, smooth=True):

        random_indices = SubsetRandomSampler(range(self.traj_len))
        sampler = BatchSampler(random_indices, batch_size, drop_last=False)

        observations, actions, skills, returns, masks = self.get(smooth=smooth)

        for indices in sampler:
            obs_batch       = observations[indices]
            action_batch    = actions[indices]
            mask            = masks[indices]
            returns_batch   = returns[indices]
            skill_batch     = skills[indices]

            yield obs_batch, action_batch, skill_batch, returns_batch, mask

# '''
# @File    :   Pretrain_ReplayBuffer
# @Time    :   2023/10/27 15:01:39
# @Author  :   Hu Bin 
# @Version :   1.0
# @Desc    :   None
# '''
class Pretrain_ReplayBuffer:

    def __init__(self, buffer_size=10, step = 128, n_envs = 1, ppo_epoch = 1):
        self.buffer_size = buffer_size
        self.obs  = [None for _ in range(self.buffer_size)]
        self.action = [None for _ in range(self.buffer_size)]
        self.action_dist = [None for _ in range(self.buffer_size)]
        self.skill = [None for _ in range(self.buffer_size)]
        self.returns = [None for _ in range(self.buffer_size)]
        self.values = [None for _ in range(self.buffer_size)]
        self.mask = [None for _ in range(self.buffer_size)]
        self.step = step
        self.n_envs = n_envs
        self.ppo_epoch = ppo_epoch

        self.ptr = 0
        self.full = False
        self.traj_idx = [0]

        self.heads_num = 4
        self.span = 10
    def __len__(self):
        return self.ptr
    
    def reset(self):
        self.obs  = [None for _ in range(self.buffer_size)]
        self.action = [None for _ in range(self.buffer_size)]
        self.action_dist = [None for _ in range(self.buffer_size)]
        self.skill = [None for _ in range(self.buffer_size)]
        self.returns = [None for _ in range(self.buffer_size)]
        self.values = [None for _ in range(self.buffer_size)]
        self.mask = [None for _ in range(self.buffer_size)]
    
        self.ptr = 0
        self.full = False
        self.traj_idx = [0]

    def add(self, obs, action, action_dist, mask, returns,values, cur_skill):
        # add step lens data, instand trajectoy
        self.obs[self.ptr]  = obs[:self.step]
        self.action[self.ptr] = action[:self.step]
        self.action_dist[self.ptr] = action_dist[:self.step]
        self.skill[self.ptr] = cur_skill[:self.step]
        self.returns[self.ptr] = returns[:self.step]
        self.values[self.ptr] = values[:self.step]
        self.mask[self.ptr] = mask[:self.step]
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.ptr = 0
            self.full = True


    def get_skill_weight(self, skill,mask, smooth=True):
        weight = torch.zeros(skill.shape[0],self.n_envs, self.heads_num)
        for t in range(self.n_envs):
            if smooth:
                skill_list = []
                traj_idx = torch.where(mask[:,t,0]==False)[0].cpu().tolist()
                traj_idx.append(skill.shape[0])
                for i in range(len(traj_idx)-1):
                    traj_skill = skill[traj_idx[i]:traj_idx[i+1],t]
                    skill_list.append(traj_skill)
                
                    k = 0
                    for i in range(len(skill_list)):
                        idx_tensor = torch.tensor(skill_list[i])
                        for j in range(idx_tensor.shape[0]):
                            if j < self.span-2:
                                idx_weight = idx_tensor[:j+2] 
                            else:
                                idx_weight = idx_tensor[(j-self.span+2):(j+2)]
                            weight[k,t] = torch.eye(self.heads_num)[idx_weight].mean(0)
                            k += 1
            else:
                #idx_tensor = torch.cat(skill_list)
                idx_tensor = skill[:,t]
                for i in range(idx_tensor.shape[0]):
                    weight[i,t] = torch.eye(self.heads_num)[idx_tensor[i].item()]
                  
        return weight

    def get(self, smooth=True):
        obs = TensorDict()
        for k in self.obs[0].keys():
            key_data = []
            for i in range(self.buffer_size):
                key_data.append(self.obs[i][k])
            obs[k] = torch.flatten(torch.cat(key_data),0,1)
            
        
        action = torch.flatten(torch.cat(self.action_dist),0,1) ## combine env  
        returns = torch.flatten(torch.cat(self.returns),0,1)
        values = torch.flatten(torch.cat(self.values),0,1)
        mask = torch.cat(self.mask) 
        skill = torch.cat(self.skill)
        #skill = torch.flatten(torch.transpose(torch.cat(self.skill),0,1),0,1).squeeze()
        skill_weight = self.get_skill_weight(skill, mask, smooth=smooth)
        skill_weight = torch.flatten(skill_weight,0,1)
        mask = torch.flatten(mask,0,1)
        return (obs, 
                action,
                skill_weight,
                returns,
                values,
                mask,
        )


    def sample(self, batch_size=256, recurrent=False, smooth=True):
        if recurrent:
            random_indices = SubsetRandomSampler(range(len(self.traj_idx)-1))
            sampler = BatchSampler(random_indices, batch_size, drop_last=False)
        else:
            random_indices = SubsetRandomSampler(range(self.step*self.buffer_size*self.n_envs))
            sampler = BatchSampler(random_indices, batch_size, drop_last=True)

        observations, actions, skills, returns, values,masks = self.get(smooth=smooth)

        for indices in sampler:
            if recurrent:
                obs_batch       = [observations[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                action_batch    = [actions[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                mask            = [masks[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                skill_batch     = [skills[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]

                obs_batch       = pad_sequence(obs_batch, batch_first=False)
                action_batch    = pad_sequence(action_batch, batch_first=False)
                mask            = pad_sequence(mask, batch_first=False)
                skill_batch     = pad_sequence(skill_batch, batch_first=False)
            else:
                obs_batch       = observations[indices]
                action_batch    = actions[indices]
                mask            = masks[indices]
                skill_batch     = skills[indices]
                returns_batch   = returns[indices]
                values_batch   = values[indices]

            yield obs_batch, action_batch, skill_batch, returns_batch,values_batch, mask

