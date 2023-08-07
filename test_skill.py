#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   my_test.py
@Time    :   2023/07/21 10:49:17
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
# Play a teaser video
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
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from collections import defaultdict

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
from habitat_baselines.utils.info_dict import extract_scalars_from_info
import torch

import argparse

def main(args):



    config_path = "habitat-baselines/habitat_baselines/config"
    config_name = "rearrange/" + args.task + ".yaml"
    hydra.initialize(config_path=config_path, job_name="test_app")
    config = hydra.compose(config_name=config_name, overrides=['habitat_baselines.evaluate=True',f'habitat_baselines.num_environments={args.num_environments}',f'habitat_baselines.eval.evals_per_ep={args.evals_per_ep}'])
    config = insert_render_options(patch_config(config))

    trainer_init = baseline_registry.get_trainer(config.habitat_baselines.trainer_name)
    trainer = trainer_init(config)
    ## load model
    ckpt_path = "data/models/" + args.model_dir + ".pth"
    ckpt_dict = trainer.load_checkpoint(ckpt_path, map_location="cpu")

    trainer.device = torch.device(args.device)
    trainer._init_envs(config, is_eval=True)
    agent = trainer._create_agent(None)
    action_shape, discrete_actions = get_action_space_info(agent.policy_action_space)
    agent.load_state_dict(ckpt_dict)
    agent.eval()
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


    stats_episodes: Dict[Any, Any] = {} 
    ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)
    rgb_frames: List[List[np.ndarray]] = [[] for _ in range(trainer.config.habitat_baselines.num_environments)]

    with torch.no_grad():
        env = trainer.envs
        observations = env.reset()  # noqa: F841
        batch = batch_obs(observations, device=trainer.device)
        batch = apply_obs_transforms_batch(batch, trainer.obs_transforms)
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        while (len(stats_episodes) < (sum(env.number_of_episodes) * evals_per_ep) and env.num_envs > 0):
            current_episodes_info = env.current_episodes()

            action_data = agent.actor_critic.act(batch,test_recurrent_hidden_states,prev_actions,not_done_masks,deterministic=False,)
            if action_data.should_inserts is None:  
                test_recurrent_hidden_states = action_data.rnn_hidden_states
                prev_actions.copy_(action_data.actions)  
            else:
                for i, should_insert in enumerate(action_data.should_inserts):
                    if should_insert.item():
                        test_recurrent_hidden_states[i] = action_data.rnn_hidden_states[i]
                        prev_actions[i].copy_(action_data.actions[i])

            if is_continuous_action_space(trainer._env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        trainer._env_spec.action_space.low,
                        trainer._env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]  
            outputs = env.step(step_data)
            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
           # obs, reward, done, info = env.step(step_data)  # noqa: F841
            batch = batch_obs(observations, device=trainer.device)
            batch = apply_obs_transforms_batch(batch, trainer.obs_transforms)  # type: ignore
            not_done_masks = torch.tensor([[not done] for done in dones], dtype=torch.bool, device=trainer.device,)
            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = env.current_episodes()
            envs_to_pause = []
            for i in range(env.num_envs):
                k = next_episodes_info[i].episode_id
                if ep_eval_count[k]== evals_per_ep:
                    envs_to_pause.append(i)
                # episode ended
                if not not_done_masks[i].item():
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k =current_episodes_info[i].episode_id
                    ep_eval_count[k] += 1
                    # use episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats
   
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
    print("fail episodes", fail_episodes)
    metric_strs = []
    print("eval_reward/average_reward", aggregated_stats["reward"])
    for k in config.habitat_baselines.eval_keys_to_include_in_name:
        metric_strs.append(f"{k}={aggregated_stats[k]:.2f}")
    print(metric_strs)
    trainer.envs.close()

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="depth_pick_skill", help="pick_skill, nav_skill, place_skill") 
    parser.add_argument("--model_dir", type=str, required=True, help="path to folder containing policy and run details")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evals_per_ep", type=int, default=1)
    parser.add_argument("--num_environments", type=int, default=1)
    parser.add_argument("--show", default=False, action='store_true')

    args = parser.parse_args()
    main(args)

