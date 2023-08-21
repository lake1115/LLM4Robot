#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   hierarchical_policy_test.py
@Time    :   2023/07/25 15:38:10
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


os.system('nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > tmp.txt')
memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu)) 
os.system('rm tmp.txt')

import habitat
import habitat.gym
from habitat.core.registry import registry
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
    append_text_underneath_image,
)

from habitat_sim.utils import viz_utils as vut
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


from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.config.default import patch_config
import hydra

from habitat_baselines.utils.common import batch_obs, get_action_space_info, inference_mode, is_continuous_action_space

from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch

from habitat_baselines.utils.info_dict import extract_scalars_from_info

from LLM.actor import FixHierarchicalPolicy, LLMHierarchicalPolicy

import torch

from typing import Any, Dict, List


from collections import defaultdict
from habitat.utils.visualizations.utils import images_to_video

import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(3)

#epsi = 26
def get_episode_env(env, episode):
    observations = env.reset()
    current_episodes_info = env.current_episodes()
    while current_episodes_info[0].episode_id != episode:
        observations = env.reset() 
        current_episodes_info = env.current_episodes()
    return env, observations
def main():

    config_path = "habitat-baselines/habitat_baselines/config"
    config_name = "rearrange/rl_hierarchical.yaml"

    hydra.initialize(config_path=config_path, job_name="test_app")
    config = hydra.compose(config_name=config_name, overrides=['habitat_baselines.evaluate=True','habitat_baselines.num_environments=1',f'habitat_baselines.eval.evals_per_ep=2'])
    config = insert_render_options(patch_config(config))
    
    trainer_init = baseline_registry.get_trainer(config.habitat_baselines.trainer_name)
    trainer = trainer_init(config)

    trainer.device = torch.device("cuda")
    trainer._init_envs(config, is_eval=True)
    trainer._create_obs_transforms()
    # agent = FixHierarchicalPolicy(config.habitat_baselines.rl.policy,config, trainer._env_spec.observation_space,trainer._env_spec.orig_action_space,config.habitat_baselines.num_environments)
    agent = LLMHierarchicalPolicy(config.habitat_baselines.rl.policy,config, trainer._env_spec.observation_space,trainer._env_spec.action_space, trainer._env_spec.orig_action_space,config.habitat_baselines.num_environments,trainer.device)
    agent.to(trainer.device)
    agent.eval()
    action_shape, discrete_actions = get_action_space_info(trainer._env_spec.action_space)
    current_episode_reward = torch.zeros(
            trainer.envs.num_envs, 1, device="cpu"
        )
    test_recurrent_hidden_states = torch.zeros(
        (
            trainer.config.habitat_baselines.num_environments,
            *agent.hidden_state_shape,
        ),
        device=trainer.device,
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
    stats_episodes: Dict[Any, Any] = {}  # dict of dicts that stores stats per episode
    ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)
    rgb_frames: List[List[np.ndarray]] = [[] for _ in range(trainer.config.habitat_baselines.num_environments)]

    with torch.no_grad():
        env = trainer.envs

        env, observations = get_episode_env(env, "26")
        #observations = env.reset()
        batch = batch_obs(observations, device=trainer.device)
        batch = apply_obs_transforms_batch(batch, trainer.obs_transforms)
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        num_steps = 0
        while (len(stats_episodes) < (sum(env.number_of_episodes) * evals_per_ep) and env.num_envs > 0):

            num_steps += 1
            current_episodes_info = env.current_episodes()
            step_data = agent.step_action(batch,not_done_masks,deterministic=False,)
        
            outputs = env.step(step_data)
            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]

            # if "move_obj_reward" in infos[0].keys():
            #     print(infos[0]['num_steps'], infos[0]["move_obj_reward"]+rewards_l[0])
            # elif "pick_reward" in infos[0].keys():
            #     print(infos[0]['num_steps'], infos[0]["pick_reward"])
            # ##
            # print(f"num steps:{num_steps}, to_object:{observations[0]['obj_start_gps_compass']}, to_goal:{observations[0]['obj_goal_gps_compass']}")
            # ##
            batch = batch_obs(observations, device=trainer.device)
            batch = apply_obs_transforms_batch(batch, trainer.obs_transforms)             
            not_done_masks = torch.tensor([[not done] for done in dones], dtype=torch.bool, device=trainer.device,)
            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = env.current_episodes()
            envs_to_pause = []
            for i in range(env.num_envs):
                k = next_episodes_info[i].episode_id
                if ep_eval_count[k]== evals_per_ep:
                    envs_to_pause.append(i)
   
                if len(trainer.config.habitat_baselines.eval.video_option) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image({k: v[i] for k, v in batch.items()}, infos[i])
                    if not not_done_masks[i].item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        frame = observations_to_image({k: v[i] * 0.0 for k, v in batch.items()}, infos[i])
                    frame = overlay_frame(frame, infos[i]) ## with infos
                    frame = append_text_underneath_image(frame,f"num_steps: {num_steps} " + agent.planner.dialogue_user + f"skill: {agent.rl_skill}" +"\t"+f"call_num: {agent._high_level_policy.call_num}",agent._high_level_policy.call)
                    rgb_frames[i].append(frame)
                # episode ended
                if not not_done_masks[i].item():
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = current_episodes_info[i].episode_id
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats
                    if (len(trainer.config.habitat_baselines.eval.video_option)> 0):
                        metrics = extract_scalars_from_info(infos[i])
                        metric_strs = []
                        for k in config.habitat_baselines.eval_keys_to_include_in_name:
                            metric_strs.append(f"{k}={metrics[k]:.2f}")
                        video_name = f"episode={current_episodes_info[i].episode_id}-" + "-".join(metric_strs)
                        images_to_video(rgb_frames[i], trainer.config.habitat_baselines.video_dir, video_name, fps=30, verbose=True)
                        rgb_frames[i] = []
                        num_steps = 0
                        agent.reset()

            not_done_masks = not_done_masks.to(device=trainer.device)
            env, test_recurrent_hidden_states, not_done_masks, current_episode_reward, prev_actions, batch, rgb_frames,= trainer._pause_envs(envs_to_pause,env,test_recurrent_hidden_states,not_done_masks,current_episode_reward,prev_actions,batch,rgb_frames,)
 

    aggregated_stats = {}
    fail_episodes = []
    for stat_key in next(iter(stats_episodes.values())).keys():
        aggregated_stats[stat_key] = np.mean([v[stat_key] for v in stats_episodes.values()])
        if stat_key == config.habitat_baselines.eval_keys_to_include_in_name[0]:
            for k, v in stats_episodes.items():
                if v[stat_key] == 0:
                    fail_episodes.append(k)
    print("eval_reward/average_reward", aggregated_stats)
    print("fail episodes", fail_episodes)
    env.close()


if __name__ == '__main__':
   main()

