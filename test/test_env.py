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
from typing import TYPE_CHECKING, Union, cast
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
import habitat
import habitat.gym
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sensors import RearrangeReward
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
    maps,
)
import matplotlib.pyplot as plt
from habitat_baselines import PPOTrainer
from habitat_sim.utils import viz_utils as vut
# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def insert_render_options(config):
    # Added settings to make rendering higher resolution for better visualization
    with habitat.config.read_write(config):
        config.habitat.simulator.concur_render = False
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        agent_config.sim_sensors.update(
            {"third_rgb_sensor": ThirdRGBSensorConfig(height=256, width=256)}
        )
    return config


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
def main():
    #cfg_path = "habitat-lab/habitat/config/benchmark/rearrange/pick.yaml"
    cfg_path = "habitat-baselines/habitat_baselines/config/rearrange/depth_nav_pick.yaml"
    config = insert_render_options(habitat.get_config(cfg_path))
    env = habitat.gym.make_gym_from_config(config)
    done = False
    env.reset()

    ## topdown map
    # top_down_map = maps.get_topdown_map_from_sim(env.env.env._env.sim, map_resolution=1024)
    # recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
    # top_down_map = recolor_map[top_down_map]
    # plt.imshow(top_down_map)
    # plt.savefig('top_down_map.png')


    target_image = maps.pointnav_draw_target_birdseye_view(
        env.env.env.habitat_env.sim.agents[0].initial_state.position,
        -np.pi / 4,
        env.env.env.habitat_env.sim.target_start_pos[0],
        
        agent_radius_px=25,
    )
    count_steps = 0
    # To save the video
    video_file_path = "video_dir/example_env.mp4"
    video_writer = vut.get_fast_video_writer(video_file_path, fps=30)

    while not done:
        step_data = env.action_space.sample()
        obs, reward, done, info = env.step(step_data)  
        video_writer.append_data(env.render(mode="rgb_array"))

        count_steps += 1
    print("Episode finished after {} steps.".format(count_steps))
    video_writer.close()



if __name__ == '__main__':
    main()
