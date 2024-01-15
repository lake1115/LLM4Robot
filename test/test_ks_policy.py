#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_ks_policy.py
@Time    :   2023/11/23 11:07:19
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
from habitat.utils.visualizations.utils import images_to_video, append_text_underneath_image
from habitat_baselines.utils.info_dict import extract_scalars_from_info

import torch
from habitat_sim.utils import viz_utils as vut
import argparse
import random
from LLM import LLM_TeacherPolicy, Fix_TeacherPolicy
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
        extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
        agent_config.sim_sensors.update(extra_sensors)
        for render_view in extra_sensors.values():
            if render_view.uuid not in config.habitat.gym.obs_keys:
                config.habitat.gym.obs_keys.append(render_view.uuid)
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
    # agent = trainer._create_agent(None)

    trainer._agent = trainer._create_agent(None)

    teacher = LLM_TeacherPolicy(config.habitat_baselines.rl.policy,config, trainer._env_spec.observation_space, trainer._env_spec.action_space, trainer._env_spec.orig_action_space, trainer.envs.num_envs, trainer.device)
    # teacher = Fix_TeacherPolicy(config.habitat_baselines.rl.policy,config, trainer._env_spec.observation_space, trainer._env_spec.action_space, trainer._env_spec.orig_action_space, trainer.envs.num_envs, trainer.device)
    ## load model   
    # ckpt_path = args.model_dir + ".pth"
    ckpt_path = "task/" + args.model_dir + "/checkpoints/latest.pth"
    ckpt_dict = trainer.load_checkpoint(ckpt_path, map_location="cpu")
    # agent.load_state_dict(ckpt_dict)
    # trainer.load_state_dict(ckpt_dict)
    agent = trainer._agent
    agent.eval()

    action_shape, discrete_actions = get_action_space_info(agent.policy_action_space)
 
    if args.show:
        rgb_frames = []
        video_file_path = os.path.join("video_dir",f"{args.task}")
        try:
            os.makedirs(video_file_path)
        except OSError:
            pass
    with torch.no_grad():
        success = []
        reward_list = []
        step = []
        for i in range(args.test_num):
            rgb_frames = []
            if args.seed is None:
                seed = i 
            else:
                seed = args.seed
            setup_seed(seed)
            env.seed(seed)

            # get_episode_env 
            if args.episode_idx is not None:
                observations = env.episode_reset(args.episode_idx)
               # env.env.env._env.sim.reset()
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
            teacher.reset()
            
            count_steps = 0
            done = False
            wait_num = 0
            while not done:
                ## get state
                batch = batch_obs([observations], device=trainer.device)
                batch = apply_obs_transforms_batch(batch, obs_transforms)

                ## get action
                action_data = agent.actor_critic.act(batch,recurrent_hidden_states,prev_actions,not_done_masks,deterministic=False,)
            
                # action_data = teacher.act(batch, recurrent_hidden_states, prev_actions, not_done_masks,deterministic=False,)
                  
                if is_continuous_action_space(action_space):
                    step_data = [np.clip(a.numpy(),action_space.low,action_space.high,) for a in action_data.env_actions.cpu()]
                else:
                    step_data = [a.item() for a in action_data.env_actions.cpu()] 
                # always want pick    
                # if teacher._cur_skills == 0:
                #     step_data[0][-4] = 1
                # print(env.env.env._env.sim.get_physics_contact_points())
                # if env.env.env._env._task.measurements.measures['move_obj_reward'].get_metric() == 0:
                #     wait_num += 1 
                #     if wait_num > 10:
                #         step_data[0][-1] = 1            
                ## step
                observations, reward, done, info = env.step(step_data[0])
                # print(f"{info['num_steps']}, {observations['obj_start_gps_compass'][0]},{observations['obj_goal_gps_compass'][0]}")
                # print(f"step: {info['num_steps']}, skill: {teacher._cur_skills} action: {step_data[0][-4]}")

                # print(f"skill: {teacher._cur_skills} reward: {reward}")
                if args.show:
                    frame = observations_to_image({k: v[0] for k, v in batch.items()}, info)
                    frame = overlay_frame(frame, info) ## with infos
                    frame = append_text_underneath_image(frame,f"num_steps: {info['num_steps']} " + f" Option: {agent.actor_critic.cur_skill[0]}")
                    rgb_frames.append(frame)
                if done:
                    if args.show:
                        frame = observations_to_image({k: v[0] * 0.0 for k, v in batch.items()}, info)
                        # metrics = extract_scalars_from_info(info)
                        # metric_strs = []
                        # for k in config.habitat_baselines.eval_keys_to_include_in_name:
                        #     metric_strs.append(f"{k}={metrics[k]:.0f}")
                        video_name = f"episode={env.current_episode().episode_id}-" + f"seed={seed}_{i}"
                        images_to_video(rgb_frames, video_file_path, video_name, fps=30, verbose=True)
                not_done_masks = torch.tensor([not done], dtype=torch.bool, device=args.device)                
                current_episode_reward += reward
                count_steps += 1 
            print("Episode finished after {} steps.".format(info['num_steps']))
            print("Episode reward {}".format(current_episode_reward))
            print(info['composite_stage_goals'], info['composite_success'])

            #step.append(info['num_steps'])
            step.append(count_steps)
            reward_list.append(current_episode_reward.item())
            success.append((info['composite_stage_goals']['stage_0_5_success'],info['composite_stage_goals']['stage_1_success'],float(info['composite_success'])))

        print('success rate', np.mean(success,0))
        print('reward', np.mean(reward_list,0))
        print('step', np.mean(step))
    env.close()


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="ks_depth_student_rearrange") 
    parser.add_argument("--model_dir", type=str, required=True, help="path to folder containing policy and run details")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_environments", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=10)
    parser.add_argument("--seed",type=int, default=None)
    parser.add_argument("--episode_idx",type=int, default=None)
    parser.add_argument("--show", default=False, action='store_true')

    args = parser.parse_args()
    main(args)

