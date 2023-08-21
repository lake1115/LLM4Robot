#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_env.py
@Time    :   2023/07/24 16:33:51
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''


# Play a teaser video
from dataclasses import dataclass

from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    MeasurementConfig,
    ThirdRGBSensorConfig,
)
from habitat_baselines.config.default import get_config
import os
import gym
import numpy as np
from hydra.core.config_store import ConfigStore

os.system('nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > tmp.txt')
memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu)) 
os.system('rm tmp.txt')

import habitat
import habitat.gym
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sensors import RearrangeReward
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
    append_text_underneath_image,
)


from LLM.actor import FixHierarchicalPolicy
# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

def insert_render_options(config):
    # Added settings to make rendering higher resolution for better visualization
    with habitat.config.read_write(config):
        config.habitat.simulator.concur_render = False
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
        #agent_config.sim_sensors.update({"third_rgb_sensor": ThirdRGBSensorConfig(height=256, width=256)} )
        agent_config.sim_sensors.update(extra_sensors)
        for render_view in extra_sensors.values():
            if render_view.uuid not in config.habitat.gym.obs_keys:
                config.habitat.gym.obs_keys.append(render_view.uuid)
    return config

from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.utils.info_dict import extract_scalars_from_info
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

import torch

import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(0)

def main():
    config_path = "habitat-baselines/habitat_baselines/config"
    config_name = "rearrange/rl_hierarchical_1.yaml"
    hydra.initialize(config_path=config_path, job_name="test_app")
    config = hydra.compose(config_name=config_name, overrides=['habitat_baselines.evaluate=True','habitat_baselines.num_environments=1','habitat_baselines.eval.evals_per_ep=1'])
    config = insert_render_options(patch_config(config))

    env = habitat.gym.make_gym_from_config(config)
    #env.seed()

    device = torch.device('cuda') 
    obs_transforms = get_active_obs_transforms(config)
    observation_space = apply_obs_transforms_obs_space(env.observation_space, obs_transforms)
    orig_action_space = env.env.original_action_space
    
    agent = FixHierarchicalPolicy(config.habitat_baselines.rl.policy,config, observation_space, env.action_space, orig_action_space,config.habitat_baselines.num_environments, device) 

    current_episode_reward = torch.zeros(1, device="cpu")
    not_done_masks = torch.zeros(1,device=device,dtype=torch.bool,)
    # To save the video
    video_file_path = "data/video_dir"
    # video_writer = vut.get_fast_video_writer(video_file_path, fps=30)
    rgb_frames = []
    with torch.no_grad():
        env.seed(0)
        obs = env.reset()
        print("current episode", env.current_episode().episode_id)
        batch = batch_obs([obs],device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)

        done = False

        while not done:
            step_data = agent.step_action(batch,not_done_masks,deterministic=False,)
            cur_skill = agent._cur_skills
            #step_data = env.action_space.sample()
            obs, reward, done, info = env.step(step_data[0])  
            batch = batch_obs([obs], device=device)
            batch = apply_obs_transforms_batch(batch, obs_transforms)  
            not_done_masks = torch.tensor([not done], dtype=torch.bool, device=device,)
            rewards = torch.tensor([reward], dtype=torch.float, device="cpu")
            current_episode_reward += rewards
                
            frame = observations_to_image({k: v[0] for k, v in batch.items()}, info)
            frame = overlay_frame(frame, info) ## with infos
            frame = append_text_underneath_image(frame,f"num_steps: {info['num_steps']} " + f"skill: {cur_skill}" +"\t")
            rgb_frames.append(frame)
            if done:
                frame = observations_to_image({k: v[0] * 0.0 for k, v in batch.items()}, info)
                metrics = extract_scalars_from_info(info)
                metric_strs = []
                for k in config.habitat_baselines.eval_keys_to_include_in_name:
                    metric_strs.append(f"{k}={metrics[k]:.0f}")
                video_name = f"episode={env.current_episode().episode_id}-" + "-".join(metric_strs)
                images_to_video(rgb_frames, video_file_path, video_name, fps=30, verbose=True)

           # video_writer.append_data(env.render(mode="rgb_array"))


        print("Episode finished after {} steps.".format(info['num_steps']))
        print("Episode reward {}".format(current_episode_reward))
        #video_writer.close()



if __name__ == '__main__':
    main()

