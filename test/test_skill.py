#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_skill.py
@Time    :   2023/09/06 10:26:44
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    MeasurementConfig,
    ThirdRGBSensorConfig,
)
from habitat_baselines.config.default import get_config
import os
import gym
import numpy as np

os.system('nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > tmp.txt')
memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
gpu_id = np.argmax(memory_gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) 
os.system('rm tmp.txt')

import habitat
import habitat.gym
from habitat.utils.visualizations.utils import observations_to_image, overlay_frame
from collections import defaultdict

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.config.default import patch_config
import hydra

from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.utils.info_dict import extract_scalars_from_info

import torch
from habitat_sim.utils import viz_utils as vut
import argparse
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_episode_env(env, episode):
    observations = env.reset()
    current_episodes_info = env.current_episode()
    while current_episodes_info.episode_id != episode:
        observations = env.reset() 
        current_episodes_info = env.current_episode()
    return env, observations
  
def insert_render_options(config):
    # Added settings to make rendering higher resolution for better visualization
    with habitat.config.read_write(config):
        config.habitat.simulator.concur_render = False
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        agent_config.sim_sensors.update(
            {"third_rgb_sensor": ThirdRGBSensorConfig(height=512, width=512)}
        )
    return config   

def main(args):
    setup_seed(0)
    config_path = "../habitat-baselines/habitat_baselines/config"
    config_name = "rearrange/" + args.task + ".yaml"
    hydra.initialize(config_path=config_path, job_name=args.task)
    config = hydra.compose(config_name=config_name, overrides=['habitat_baselines.evaluate=True',f'habitat_baselines.num_environments={args.num_environments}'])
    ## add third rgb sensor
    config = insert_render_options(patch_config(config))
    #config = patch_config(config)

    env = habitat.gym.make_gym_from_config(config)
    ## choose one episode 
    #env.env.env.habitat_env.episode_iterator=None
    obs_transforms = get_active_obs_transforms(config)
    action_space = env.action_space

    ## initial trainer
    trainer_init = baseline_registry.get_trainer(config.habitat_baselines.trainer_name)
    trainer = trainer_init(config)
    trainer.envs = env.env
    trainer.envs.num_envs = args.num_environments
    trainer.device = torch.device(args.device)
    trainer._env_spec = EnvironmentSpec(
        observation_space=env.env.observation_space,
        action_space=env.env.action_space,
        orig_action_space=env.env.original_action_space,
    )
    
    agent = trainer._create_agent(None)
    ## load model   
    ckpt_path = "data/models/" + args.model_dir + ".pth"
    #ckpt_path = "data/" + args.model_dir + "/checkpoints/latest.pth"
    ckpt_dict = trainer.load_checkpoint(ckpt_path, map_location="cpu")
    agent.load_state_dict(ckpt_dict)
    agent.eval()

    action_shape, discrete_actions = get_action_space_info(agent.policy_action_space)
 
    if args.show:
        video_file_path = os.path.join("video_dir",f"{args.task}")
        try:
            os.makedirs(video_file_path)
        except OSError:
            pass
    with torch.no_grad():
        success = []
        step = []
        for i in range(args.test_num):
            if args.seed is None:
                seed = i
                
            else:
                seed = args.seed
            env.seed(seed)

            # get_episode_env 
            if args.episode_idx is not None:
                observations = env.episode_reset(args.episode_idx)
            else:
                observations = env.reset() 
            print(f"current episode {env.current_episode().episode_id}, seed {seed}")
            current_episode_reward = torch.zeros(trainer.envs.num_envs, 1, device="cpu")
            recurrent_hidden_states = torch.zeros(
                (
                    trainer.config.habitat_baselines.num_environments,
                    *agent.hidden_state_shape,
                ),
                device=trainer.device,
            )
            should_update_recurrent_hidden_states = (
                np.prod(recurrent_hidden_states.shape) != 0
            )
            prev_actions = torch.zeros(
                trainer.config.habitat_baselines.num_environments,
                *action_shape,
                device=trainer.device,
                dtype=torch.long if discrete_actions else torch.float,
            )
            not_done_masks = torch.zeros(
                trainer.config.habitat_baselines.num_environments,
                1,
                device=trainer.device,
                dtype=torch.bool,
            )

            count_steps = 0
            done = False
            if args.show:
                video_writer = vut.get_fast_video_writer(video_file_path + "/seed" + str(seed) +".mp4", fps=60)
            while not done:
                ## get state
                batch = batch_obs([observations], device=trainer.device)
                batch = apply_obs_transforms_batch(batch, obs_transforms)

                ## get action
                action_data = agent.actor_critic.act(batch,recurrent_hidden_states,prev_actions,not_done_masks,deterministic=False,)

                # if action_data.should_inserts is None:  
                #     recurrent_hidden_states = action_data.rnn_hidden_states
                #     prev_actions.copy_(action_data.actions)  
                # else:
                #     for i, should_insert in enumerate(action_data.should_inserts):
                #         if should_insert.item():
                #             recurrent_hidden_states[i] = action_data.rnn_hidden_states[i]
                #             prev_actions[i].copy_(action_data.actions[i])
                prev_actions = torch.rand_like(action_data.actions)
                if is_continuous_action_space(action_space):
                    step_data = [np.clip(a.numpy(),action_space.low,action_space.high,) for a in action_data.env_actions.cpu()]
                else:
                    step_data = [a.item() for a in action_data.env_actions.cpu()] 
  
                ## step
                observations, reward, done, info = env.step(step_data[0])
                if args.show:
                    video_writer.append_data(env.render(mode="rgb_array"))
                not_done_masks = torch.tensor([not done], dtype=torch.bool, device=args.device)                
                current_episode_reward += reward
                count_steps += 1 
            print("Episode finished after {} steps.".format(info['num_steps']))
            print("Episode reward {}".format(current_episode_reward))


            #step.append(info['num_steps'])
            step.append(count_steps)
            if 'nav_to_obj_success' in info.keys():
                success.append(float(info['nav_to_obj_success'] | info['nav_to_pos_success']))
            elif 'pick_success' in info.keys():
                success.append(float(info['pick_success']))
            elif 'place_success' in info.keys():
                success.append(float(info['place_success']))
            elif 'success' in info.keys():
                success.append(float(info['success']))
        #torch.save(step, "result/" + args.model_dir+"_step.pt")
        #torch.save(success, "result/" + args.model_dir+"_success.pt")
        print('success rate', np.mean(success))
        print('step', np.mean(step))
    env.close()
    if args.show:
        video_writer.close()

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="depth_nav_skill", help="pick_skill, nav_skill, place_skill") 
    parser.add_argument("--model_dir", type=str, required=True, help="path to folder containing policy and run details")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_environments", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=10)
    parser.add_argument("--seed",type=int, default=None)
    parser.add_argument("--episode_idx",type=int, default=None)
    parser.add_argument("--show", default=False, action='store_true')

    args = parser.parse_args()
    main(args)

