#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
import numpy as np
from gym import spaces
from torch import nn as nn
import torch.distributions as D
from habitat.tasks.nav.nav import (
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.utils.common import (
    CategoricalNet,
    GaussianNet,
    get_num_actions,
    kl_loss,
    mse_loss,
    was_loss,
    new_was_loss,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

from torch import Tensor

from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy, PointNavResNetNet

from habitat_baselines.common.tensor_dict import TensorDict
from habitat.core.spaces import ActionSpace,EmptySpace
from collections import OrderedDict
from gym.spaces import Box
from typing import Dict
def truncate_obs_space(space: spaces.Box, truncate_len: int) -> spaces.Box:
    """
    Returns an observation space with taking on the first `truncate_len` elements of the space.
    """
    return spaces.Box(
        low=space.low[..., :truncate_len],
        high=space.high[..., :truncate_len],
        dtype=np.float32,
    )

@baseline_registry.register_policy
class KSPolicy(PointNavResNetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        fuse_keys: Optional[List[str]] = None,
        heads_num: int = 4,
        **kwargs,
    ):
        super().__init__(
        observation_space = observation_space,
        action_space = action_space,
        hidden_size = hidden_size,
        num_recurrent_layers = num_recurrent_layers,
        rnn_type = rnn_type,
        resnet_baseplanes = resnet_baseplanes,
        backbone = backbone,
        normalize_visual_inputs = normalize_visual_inputs,
        force_blind_policy = force_blind_policy,
        policy_config = policy_config,
        aux_loss_config = aux_loss_config,
        fuse_keys = fuse_keys,
        **kwargs,
        )
        ####
        self.multihead = MultiHeadDist(
            observation_space = observation_space,
            action_space = kwargs['orig_action_space'],
            num_inputs = self.net.output_size,
            num_outputs = self.dim_actions,
            config = policy_config,
            heads_num = heads_num,
            )
        self.cur_skill = -1* np.ones(10, dtype=np.int8)  ## env_num?
        self.cur_skill_step = np.zeros(10, dtype=np.int32) 
        ###
        ### HKS
        # self.skill_use_num = 2
        # self.skill_weight_layer = torch.nn.Sequential(
        #                         nn.Linear(538,self.skill_use_num ),
        #                         nn.Softmax(dim=1),
        #                         )
        # self.skill_name = ['pick', 'place', 'nav','nav_to_goal']
        # self.skill_keys = []
        # self.skill_step = []
        # self.observation_space = observation_space
        # self.action_space = kwargs['orig_action_space']
        # self.policy_config = policy_config
        # self.teacher_action_distribution= torch.nn.ModuleList([self.load_skill(name) for name in self.skill_name]) 
        ###

    def load_skill(self, skill_name, pretrain=True):
        if skill_name == 'pick':
            config = self.policy_config.hierarchical_policy.defined_skills.pick
            #cls = eval(config.skill_name)
            ckpt_dict = torch.load('data/models/ks_pick.pth', map_location="cpu")
        elif skill_name == 'place':
            config = self.policy_config.hierarchical_policy.defined_skills.place
            ckpt_dict = torch.load('data/models/ks_place.pth', map_location="cpu")
            
        elif skill_name == 'nav':
            config = self.policy_config.hierarchical_policy.defined_skills.nav_to_obj
            ckpt_dict = torch.load('data/models/ks_nav.pth', map_location="cpu")
            skill_arg = [['goal0|0', 'robot_0']]
        elif skill_name == 'nav_to_goal':
            config = self.policy_config.hierarchical_policy.defined_skills.nav_to_obj
            ckpt_dict = torch.load('data/models/ks_nav.pth', map_location="cpu")         
            skill_arg = [['TARGET_goal0|0', 'robot_0']]
        policy_cfg = ckpt_dict["config"]
        self.skill_step.append(policy_cfg['habitat']['environment']['max_episode_steps'])
        policy = baseline_registry.get_policy('PointNavResNetPolicy')
        expected_obs_keys = policy_cfg.habitat.gym.obs_keys
        filtered_obs_space = spaces.Dict(
            {k: self.observation_space.spaces[k] for k in expected_obs_keys}
        )
        self.skill_keys.append(expected_obs_keys)
        for k in config.obs_skill_inputs:
            if k not in filtered_obs_space.spaces:
                raise ValueError(f"Could not find {k} for skill")
            space = filtered_obs_space.spaces[k]
            # There is always a 3D position
            filtered_obs_space.spaces[k] = truncate_obs_space(space, 3)


        filtered_action_space = ActionSpace(
            OrderedDict(
                (k, self.action_space[k])
                for k in policy_cfg.habitat.task.actions.keys()
            )
        )

        if "arm_action" in filtered_action_space.spaces and (
            policy_cfg.habitat.task.actions.arm_action.grip_controller is None
        ):
            filtered_action_space["arm_action"] = spaces.Dict(
                {
                    k: v
                    for k, v in filtered_action_space["arm_action"].items()
                    if k != "grip_action"
                }
            )
        actor_critic = policy.from_config(
            policy_cfg, filtered_obs_space, filtered_action_space
        )
        if pretrain:
            actor_critic.load_state_dict(ckpt_dict["state_dict"])
        return actor_critic
   
    def ideal_planner(self, batch_idx, obs, mask):
        if not mask:
            skill = 2
            self.cur_skill[batch_idx] = 2
            self.cur_skill_step[batch_idx] = 0
        elif self.cur_skill[batch_idx] == 2 and obs['obj_start_gps_compass'][0] <= 1.2:
            skill = 0
            self.cur_skill[batch_idx] = 0
            self.cur_skill_step[batch_idx] = 0
        elif self.cur_skill[batch_idx] == 2 and obs['obj_start_gps_compass'][0] > 1.2:
            skill = 2
            self.cur_skill[batch_idx] = 2
            self.cur_skill_step[batch_idx] += 1
        elif self.cur_skill[batch_idx] == 0 and obs['is_holding'][0] == 0:
            skill = 0
            self.cur_skill[batch_idx] = 0
            self.cur_skill_step[batch_idx] += 1
        elif self.cur_skill[batch_idx] == 0 and obs['is_holding'][0] == 1 and torch.norm(obs['relative_resting_position'])>0.15:
            skill = 0
            self.cur_skill[batch_idx] = 0
            self.cur_skill_step[batch_idx] += 1
        elif self.cur_skill[batch_idx] == 0 and obs['is_holding'][0] == 1 and torch.norm(obs['relative_resting_position'])<=0.15:
            skill = 3
            self.cur_skill[batch_idx] = 3
            self.cur_skill_step[batch_idx] = 0
        elif self.cur_skill[batch_idx] == 3 and obs['obj_goal_gps_compass'][0] <= 1.5:
            skill = 1
            self.cur_skill[batch_idx] = 1
            self.cur_skill_step[batch_idx] = 0
        elif self.cur_skill[batch_idx] == 3 and obs['obj_goal_gps_compass'][0] > 1.5:
            skill = 3
            self.cur_skill[batch_idx] = 3
            self.cur_skill_step[batch_idx] += 1
        elif self.cur_skill[batch_idx] == 1:
            skill = 1
            self.cur_skill[batch_idx] = 1
            self.cur_skill_step[batch_idx] += 1
        return skill    
    

    ### HKS
    # def act(
    #     self,
    #     observations,
    #     rnn_hidden_states,
    #     prev_actions,
    #     masks,
    #     deterministic=False,
    # ):
    #     features, rnn_hidden_states, _ = self.net(
    #         observations, rnn_hidden_states, prev_actions, masks
    #     )
    #     distribution = self.action_distribution(features)
    #     value = self.critic(features)

    #     if deterministic:
    #         if self.action_distribution_type == "categorical":
    #             action = distribution.mode()
    #         elif self.action_distribution_type == "gaussian":
    #             action = distribution.mean
    #     else:
    #         action = distribution.sample()

    #     action_log_probs = distribution.log_probs(action)

    #     ## teacher policy
    #     teacher_actions_mean = torch.zeros((masks.shape[0], self.skill_use_num, 11), device=masks.device)
    #     teacher_actions_std = torch.zeros((masks.shape[0], self.skill_use_num, 11), device=masks.device)

    #     for idx, skill in enumerate([0,2]):
    #         filtered_obs = TensorDict({k: observations[k] for k in self.skill_keys[skill]})
    #         if skill == 2:
    #             filtered_obs['goal_to_agent_gps_compass'] = observations['obj_start_gps_compass']
    #         elif skill == 3:
    #             filtered_obs['goal_to_agent_gps_compass'] = observations['obj_goal_gps_compass']


    #         net = self.teacher_action_distribution[skill].net
    #         action_distribution = self.teacher_action_distribution[skill].action_distribution
    #         features, rnn_s, _ = net(
    #             filtered_obs, rnn_hidden_states, prev_actions, masks
    #         )
    #         distribution = action_distribution(features)
   
    #         if skill == 2 or skill == 3:
    #             teacher_actions_mean[:, idx, -3:] = distribution.mean
    #             teacher_actions_std[:, idx, -3:] = distribution.stddev
    #             if skill == 2:
    #                 teacher_actions_mean[:, idx, -4] = -1.0
    #             elif skill == 3:
    #                 teacher_actions_mean[:, idx, -4] = 1.0
    #         if skill == 1 or skill == 0:
    #             teacher_actions_mean[:, idx, :10] = distribution.mean
    #             teacher_actions_std[:, idx, :10] = distribution.stddev
    #             if skill == 0:
    #                 index = torch.where(observations['is_holding']== 1)[0]
    #                 teacher_actions_mean[index, idx, -4] = 1.0
    #             elif skill == 1:
    #                 index = torch.where(observations['is_holding']== 0)[0]
    #                 teacher_actions_mean[index, idx, -4] = -1.0
    #     teacher_actions = torch.cat((teacher_actions_mean,teacher_actions_std),dim=-1)
    #     skill_weight = self.skill_weight_layer(self.net.feature_state)
    #     teacher_action_data = torch.sum(skill_weight.unsqueeze(-1) * teacher_actions, dim=1)

    #     return PolicyActionData(
    #         values=value,
    #         actions=action,
    #         action_log_probs=action_log_probs,
    #         rnn_hidden_states=rnn_hidden_states,
    #     ), teacher_action_data
    
    # def evaluate_actions(
    #     self,
    #     observations,
    #     rnn_hidden_states,
    #     prev_actions,
    #     masks,
    #     action,
    #     teacher_actions,
    #     teacher_skills,
    #     rnn_build_seq_info: Dict[str, torch.Tensor],
    # ):
    #     features, rnn_hidden_states, aux_loss_state = self.net(
    #         observations,
    #         rnn_hidden_states,
    #         prev_actions,
    #         masks,
    #         rnn_build_seq_info,
    #     )
    #     distribution = self.action_distribution(features)
    #     value = self.critic(features)

    #     action_log_probs = distribution.log_probs(action)
    #     distribution_entropy = distribution.entropy()

    #     batch = dict(
    #         observations=observations,
    #         rnn_hidden_states=rnn_hidden_states,
    #         prev_actions=prev_actions,
    #         masks=masks,
    #         action=action,
    #         rnn_build_seq_info=rnn_build_seq_info,
    #     )
    #     aux_loss_res = {
    #         k: v(aux_loss_state, batch)
    #         for k, v in self.aux_loss_modules.items()
    #     }
    #     ## ks loss 
    #     student_actions = torch.zeros((masks.shape[0], 22), device=masks.device)
    #     student_actions[:,:11] = distribution.mean  
    #     student_actions[:, 11:] = distribution.stddev 
    #     ks_loss = new_was_loss(student_actions, teacher_actions)


       
    #     return (
    #         value,
    #         ks_loss,
    #         #skill_classify_loss,
    #         action_log_probs,
    #         distribution_entropy,
    #         rnn_hidden_states,
    #         aux_loss_res,
    #     )
    ###

    ### our approach
    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        teacher_skill = None,
        deterministic=False,
    ):
        #ideal_skill = self.ideal_planner(observations,masks)

        actions = torch.zeros((masks.shape[0], 11), device=masks.device)
        action_log_probs = torch.zeros((masks.shape[0], 1), device=masks.device)
        values = torch.zeros((masks.shape[0], 1), device=masks.device)
        for batch_idx in range(masks.shape[0]):
            
            obs = observations[[batch_idx]]
            mask = masks[[batch_idx]]
            skill = self.ideal_planner(batch_idx, obs[0], mask[0])
            #skill = int(ideal_skill[batch_idx])
            filtered_obs = TensorDict({k: obs[k] for k in self.multihead.skill_keys[skill]})
            if skill == 2:
                filtered_obs['goal_to_agent_gps_compass'] = obs['obj_start_gps_compass']
            elif skill == 3:
                filtered_obs['goal_to_agent_gps_compass'] = obs['obj_goal_gps_compass']
            rnn_s = rnn_hidden_states[[batch_idx]]
            prev_a = prev_actions[[batch_idx]]
     
            
            net = self.multihead.multi_heads_action_distribution[skill].net
            action_distribution = self.multihead.multi_heads_action_distribution[skill].action_distribution
            critic = self.multihead.multi_heads_action_distribution[skill].critic
            features, rnn_s, _ = net(
                filtered_obs, rnn_s, prev_a, mask
            )
            distribution = action_distribution(features)
            values[batch_idx] = critic(features)

            if deterministic:
                if self.action_distribution_type == "categorical":
                    action = distribution.mode()
                elif self.action_distribution_type == "gaussian":
                    action = distribution.mean
            else:
                action = distribution.sample()

            action_log_probs[batch_idx] = distribution.log_probs(action)

            if skill == 2 or skill == 3:
                actions[batch_idx, -3:] = action
                if skill == 2:
                    actions[batch_idx, -4] = -1.0
                elif skill == 3:
                    actions[batch_idx, -4] = 1.0
            if skill == 1 or skill == 0:
                actions[batch_idx,:10] = action
                if skill == 0 and obs['is_holding']:
                    actions[batch_idx, -4] = 1.0
                elif skill == 1 and obs['is_holding'] == 0:
                    actions[batch_idx, -4] = -1.0

            if self.cur_skill_step[batch_idx] > self.multihead.skill_step[skill]:
                actions[batch_idx, -1] = 1.0
            else:
                actions[batch_idx, -1] = -1.0

        return PolicyActionData(
            values=values,
            actions=actions,
            action_log_probs=action_log_probs,
            rnn_hidden_states=rnn_hidden_states,
        )

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        teacher_actions,
        teacher_skills,
        rnn_build_seq_info: Dict[str, torch.Tensor],
    ):
        #ideal_skill = self.ideal_planner(TensorDict(observations),masks)
        teacher_weight = torch.eye(4)[teacher_skills.cpu().int()]
        print("training data components:", torch.mean(teacher_weight,dim=0))

        student_actions = torch.zeros((masks.shape[0], 22), device=masks.device)
        action_log_probs = torch.zeros((masks.shape[0], 1), device=masks.device)
        distribution_entropy = torch.zeros((masks.shape[0], 1), device=masks.device)
        values = torch.zeros((masks.shape[0], 1), device=masks.device)

        for skill in range(self.multihead.heads_num):
            batch_idx = torch.where(teacher_skills == skill)[0]
            if len(batch_idx) == 0:
                continue
            obs = TensorDict(observations)[batch_idx]
            
            filtered_obs = TensorDict({k: obs[k] for k in self.multihead.skill_keys[skill]})
            if skill == 2:
                filtered_obs['goal_to_agent_gps_compass'] = obs['obj_start_gps_compass']
            elif skill == 3:
                filtered_obs['goal_to_agent_gps_compass'] = obs['obj_goal_gps_compass']
            rnn_s = rnn_hidden_states
            prev_a = prev_actions[batch_idx]
            mask = masks[batch_idx]
            net = self.multihead.multi_heads_action_distribution[skill].net
            action_distribution = self.multihead.multi_heads_action_distribution[skill].action_distribution
            critic = self.multihead.multi_heads_action_distribution[skill].critic
            features, rnn_s, aux_loss_state = net(
                filtered_obs, rnn_s, prev_a, mask
            )
            distribution = action_distribution(features)            
            values[batch_idx] = critic(features)
            if skill == 0 or skill == 1:
                student_actions[batch_idx,:10] = distribution.mean  
                student_actions[batch_idx, 11:21] = distribution.stddev  
                action_log_probs[batch_idx] = distribution.log_probs(action[batch_idx, :10])
            elif skill == 2 or skill == 3:
                student_actions[batch_idx,8:11] = distribution.mean  
                student_actions[batch_idx, 19:22] = distribution.stddev 
                action_log_probs[batch_idx] = distribution.log_probs(action[batch_idx, -3:])
            distribution_entropy[batch_idx] = distribution.entropy()


        ks_loss = new_was_loss(teacher_actions, student_actions)

        batch = dict(
            observations=observations,
            rnn_hidden_states=rnn_hidden_states,
            prev_actions=prev_actions,
            masks=masks,
            action=action,
            rnn_build_seq_info=rnn_build_seq_info,
        )
        aux_loss_res = {
            k: v(aux_loss_state, batch)
            for k, v in self.aux_loss_modules.items()
        }

        return (
            values,
            ks_loss,
            #skill_classify_loss,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            aux_loss_res,
        )
   
    ###
'''
@File    :   MultiHeadEncoder
@Time    :   2023/10/24 17:20:00
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''  



class MultiHeadDist(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        num_inputs: int,
        num_outputs: int,
        config: "DictConfig",
        heads_num: int,
        span: int = 5,
    ):
        super().__init__()
        self.span = span

        self.observation_space = observation_space
        self.action_space = action_space

        self.skill_name = ['pick', 'place', 'nav','nav_to_goal']
        self.skill_keys = []
        self.skill_step = []
        self.heads_num = len(self.skill_name)
        self.config = config
        # self.multi_heads_action_distribution= torch.nn.ModuleList([self.load_skill(name) for name in self.skill_name])
        self.multi_heads_action_distribution = torch.nn.ModuleList([self.load_skill(name, pretrain=False) for name in self.skill_name])


        


    def load_skill(self, skill_name, pretrain=True):
        if skill_name == 'pick':
            config = self.config.hierarchical_policy.defined_skills.pick
            #cls = eval(config.skill_name)
            ckpt_dict = torch.load('data/models/ks_pick.pth', map_location="cpu")
        elif skill_name == 'place':
            config = self.config.hierarchical_policy.defined_skills.place
            ckpt_dict = torch.load('data/models/ks_place.pth', map_location="cpu")
            
        elif skill_name == 'nav':
            config = self.config.hierarchical_policy.defined_skills.nav_to_obj
            ckpt_dict = torch.load('data/models/ks_nav.pth', map_location="cpu")
            skill_arg = [['goal0|0', 'robot_0']]
        elif skill_name == 'nav_to_goal':
            config = self.config.hierarchical_policy.defined_skills.nav_to_obj
            ckpt_dict = torch.load('data/models/ks_nav.pth', map_location="cpu")         
            skill_arg = [['TARGET_goal0|0', 'robot_0']]
        policy_cfg = ckpt_dict["config"]
        self.skill_step.append(policy_cfg['habitat']['environment']['max_episode_steps'])
        policy = baseline_registry.get_policy('PointNavResNetPolicy')
        expected_obs_keys = policy_cfg.habitat.gym.obs_keys
        filtered_obs_space = spaces.Dict(
            {k: self.observation_space.spaces[k] for k in expected_obs_keys}
        )
        self.skill_keys.append(expected_obs_keys)
        for k in config.obs_skill_inputs:
            if k not in filtered_obs_space.spaces:
                raise ValueError(f"Could not find {k} for skill")
            space = filtered_obs_space.spaces[k]
            # There is always a 3D position
            filtered_obs_space.spaces[k] = truncate_obs_space(space, 3)


        filtered_action_space = ActionSpace(
            OrderedDict(
                (k, self.action_space[k])
                for k in policy_cfg.habitat.task.actions.keys()
            )
        )

        if "arm_action" in filtered_action_space.spaces and (
            policy_cfg.habitat.task.actions.arm_action.grip_controller is None
        ):
            filtered_action_space["arm_action"] = spaces.Dict(
                {
                    k: v
                    for k, v in filtered_action_space["arm_action"].items()
                    if k != "grip_action"
                }
            )
        actor_critic = policy.from_config(
            policy_cfg, filtered_obs_space, filtered_action_space
        )
        if pretrain:
            actor_critic.load_state_dict(ckpt_dict["state_dict"])
        return actor_critic
    def forward(self, features, teacher_skill):
        action = self.multi_heads_action_distribution[teacher_skill](features) 
        return action
    
