#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_call_LLM.py
@Time    :   2023/08/09 11:24:00
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
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)

import sys
sys.path.append(os.getcwd())

from LLM import FixHierarchicalPolicy, LLMHierarchicalPolicy

import torch

from typing import Any, Dict, List

from LLM import Comm_Net
from collections import defaultdict
from habitat.utils.visualizations.utils import images_to_video
import argparse
import random
import time
import sys
import algos
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_episode_env(env, episode):
    observations = env.reset()
    current_episodes_info = env.current_episodes()
    while current_episodes_info[0].episode_id != episode:
        observations = env.reset() 
        current_episodes_info = env.current_episodes()
    return env, observations


class Env:
    def __init__(self, args, policy=None):
        config_path = "../habitat-baselines/habitat_baselines/config"
        config_name = "rearrange/" + args.task + ".yaml"
        self.gamma = args.gamma
        self.lam = args.lam
        self.ask_lambda = args.ask_lambda
        self.batch_size = args.batch_size
        self.episode = args.episode
        self.device = torch.device(args.device)
        self.seed = args.seed
        hydra.initialize(config_path=config_path, job_name="test_app")
        config = hydra.compose(config_name=config_name, overrides=['habitat_baselines.evaluate=True',f'habitat_baselines.num_environments=1',f'habitat_baselines.eval.evals_per_ep={args.evals_per_ep}'])
        self.config = insert_render_options(patch_config(config))
        self.env = habitat.gym.make_gym_from_config(self.config)
        self.env.env.env.habitat_env.episode_iterator=None  ## set no episode iterator
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(self.env.observation_space, self.obs_transforms)
        action_space = self.env.action_space
        orig_action_space = self.env.env.original_action_space         
        #self.agent = LLMHierarchicalPolicy(self.config.habitat_baselines.rl.policy,self.config, observation_space, action_space, orig_action_space, 1, self.device) 
        self.agent = FixHierarchicalPolicy(self.config.habitat_baselines.rl.policy,self.config, observation_space, action_space, orig_action_space, 1, self.device) 

        if policy is not None:
            ## communication network for ask
            print("load policy!")
            self.agent._high_level_policy.comm_net = policy.to(self.device)        
        
        self.buffer = algos.Buffer(self.gamma, self.lam, self.device)    
        self.logger = algos.create_logger(args)
        self.ppo_algo = algos.PPO(self.agent._high_level_policy.comm_net, device=self.device, save_path=self.logger.dir, batch_size=self.batch_size)

        self.n_itr = args.n_itr
        self.traj_per_itr = args.traj_per_itr
        self.total_steps = 0

    def train(self):
        start_time = time.time()
        for itr in range(self.n_itr):
            print("********** Iteration {} ************".format(itr))
            print("time elapsed: {:.2f} s".format(time.time() - start_time))        
            ## collecting ##
            sample_start = time.time()
            buffer = []
            i = 0
            while i < self.traj_per_itr:
                traj_buffer = self.collect()
                if len(traj_buffer):
                    buffer.append(traj_buffer) 
                    i += 1
                   
            self.buffer = algos.Merge_Buffers(buffer,device=self.device)
            total_steps = len(self.buffer)
            samp_time = time.time() - sample_start
            print("{:.2f} s to collect {:6n} timesteps | {:3.2f}sample/s.".format(samp_time, total_steps, (total_steps)/samp_time))
            self.total_steps += total_steps

            ## training ##
            optimizer_start = time.time()
            mean_losses = self.ppo_algo.update_policy(self.buffer)
            opt_time = time.time() - optimizer_start
            print("{:.2f} s to optimizer| loss {:6.3f}, entropy {:6.3f}.".format(opt_time, mean_losses[0], mean_losses[1]))

            ## eval_policy ##
            evaluate_start = time.time()
            eval_reward, eval_len, eval_success, eval_interactions = self.eval(trajs=1, seed=self.seed)
            eval_time = time.time() - evaluate_start
            print("{:.2f} s to evaluate.".format(eval_time))
            if self.logger is not None:
                avg_eval_reward = eval_reward
                avg_batch_reward = np.mean(self.buffer.ep_returns)
                std_batch_reward = np.std(self.buffer.ep_returns)
                avg_ep_len = np.mean(self.buffer.ep_lens)
                avg_ep_comm = np.mean(self.buffer.ep_interactions)
                success_rate = np.mean(self.buffer.ep_success,0)
                avg_eval_len = eval_len
                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Timesteps', self.total_steps) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (test)', round(avg_eval_reward,2)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Ep Lens (test) ', round(avg_eval_len,2)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Ep Comm (test) ', round(eval_interactions,2)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (train)', round(avg_batch_reward,2)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Success', np.round(success_rate,2)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean len', round(avg_ep_len,2)) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean Comm', round(avg_ep_comm,2)) + "\n")
                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.flush()

                self.logger.add_scalar("Test/Return", avg_eval_reward, itr)
                self.logger.add_scalar("Test/Mean Eplen", avg_eval_len, itr)
                self.logger.add_scalar("Test/Comm", eval_interactions, itr)
                self.logger.add_scalar("Train/Return Mean", avg_batch_reward, itr)
                self.logger.add_scalar("Train/Return Std", std_batch_reward, itr)
                self.logger.add_scalar("Train/Eplen", avg_ep_len, itr)
                self.logger.add_scalar("Train/Comm", avg_ep_comm, itr)
                self.logger.add_scalar("Train/Success Rate1", success_rate[0], itr)
                self.logger.add_scalar("Train/Success Rate2", success_rate[1], itr)
                self.logger.add_scalar("Train/Success Rate3", success_rate[2], itr)
                self.logger.add_scalar("Train/Loss", mean_losses[0], itr)
                self.logger.add_scalar("Train/Mean Entropy", mean_losses[1], itr)

                self.ppo_algo.save()
    
    def collect(self, seed=None):
        if seed is not None:
            self.env.seed(seed)

        with torch.no_grad():
            buffer = algos.Buffer(self.gamma, self.lam)
            obs = self.env.reset()
            self.agent.reset()
            done = False
            not_done_masks = torch.zeros(1,device=self.device,dtype=torch.bool,)
            pre_skill = None
            while not done:
                batch = batch_obs([obs],device=self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)
                step_data = self.agent.step_action(batch,not_done_masks,deterministic=False,)

                cur_skill = self.agent._cur_skills
                if pre_skill == cur_skill: # additional penalty term for repeat same skill
                    repeat_feedback = torch.tensor([1.])
                elif pre_skill is None:
                    repeat_feedback = torch.tensor([0.])
                else:
                    repeat_feedback = torch.tensor([-10.])
                pre_skill = cur_skill
                rl_act, rl_log_prob, rl_value, rl_skill = self.agent.rl_data()

                obs, reward, done, info = self.env.step(step_data[0])  

                if rl_log_prob is not None and rl_value is not None:
                    comm_penalty = (self.ask_lambda * repeat_feedback) * (rl_act) ## communication penalty
                    comm_reward = reward - comm_penalty 
                    buffer.store(self.agent.his_obs, rl_act.cpu(), comm_reward, rl_value.cpu(), rl_log_prob.cpu(), rl_skill) 

                not_done_masks = torch.tensor([not done], dtype=torch.bool, device=self.device)

            buffer.finish_path(last_val=0, interactions=self.agent._high_level_policy.call_num, success=(info['composite_stage_goals'],info['composite_success']))
            
        return buffer

          
    def eval(self, trajs=1, seed=None):
        if seed is not None:
            self.env.seed(seed)
       
        if args.record_video:
            rgb_frames: List = []
            dir_path = os.path.join(self.logger.dir, 'video_dir')
            try:
                os.makedirs(dir_path)
            except OSError:
                pass
        with torch.no_grad():
            obs = self.env.reset()
            print(f"current episode {self.env.current_episode().episode_id}, seed {seed}")
            self.agent.reset()
            done = False
            not_done_masks = torch.zeros(1,device=self.device,dtype=torch.bool,)
            pre_skill = None
            ep_reward = 0
            while not done:
                batch = batch_obs([obs],device=self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)
                step_data = self.agent.step_action(batch,not_done_masks,deterministic=False,)

                cur_skill = self.agent._cur_skills
                if pre_skill == cur_skill: # additional penalty term for repeat same skill
                    repeat_feedback = 1.
                elif pre_skill is None:
                    repeat_feedback = 0.
                else:
                    repeat_feedback = -1.
                pre_skill = cur_skill
                rl_act, _, _, rl_skill = self.agent.rl_data()

                obs, reward, done, info = self.env.step(step_data[0])  

                comm_penalty = (self.ask_lambda * repeat_feedback) * (rl_act.item()) ## communication penalty
                comm_reward = reward - comm_penalty 
                not_done_masks = torch.tensor([not done], dtype=torch.bool, device=self.device)
                ep_reward += comm_reward


                if args.record_video:
                    frame = observations_to_image({k: v[0] for k, v in batch.items()}, info)
                    if not not_done_masks.item():
                        frame = observations_to_image({k: v[0] * 0.0 for k, v in batch.items()}, info)
                    frame = overlay_frame(frame, info) ## with infos
                    frame = append_text_underneath_image(frame,f"num_steps: {info['num_steps']} " + self.agent.planner.dialogue_user + f"skill: {rl_skill}" +"\t"+f"call_num: {self.agent._high_level_policy.call_num}",self.agent._high_level_policy.call)
                    rgb_frames.append(frame)
            # episode ended
            if args.record_video:
                metrics = extract_scalars_from_info(info)
                metric_strs = []
                for k in self.config.habitat_baselines.eval_keys_to_include_in_name:
                    metric_strs.append(f"{k}={metrics[k]:.0f}")
                video_name = f"episode={self.env.current_episode().episode_id}-"+ f"seed={seed}" #+ "-".join(metric_strs)
                images_to_video(rgb_frames, dir_path, video_name, fps=30, verbose=True) 
            print("mission completed:", info['composite_success'])
            return ep_reward, info['num_steps'],  (info['composite_stage_goals'],info['composite_success']), self.agent._high_level_policy.call_num
        
    def eval_fix(self, trajs=1, seed=None):
        if seed is not None:
            self.env.seed(seed)
       
        if args.record_video:
            rgb_frames: List = []
            dir_path = os.path.join(self.logger.dir, 'video_dir')
            try:
                os.makedirs(dir_path)
            except OSError:
                pass
        with torch.no_grad():
            obs = self.env.reset()
            print(f"current episode {self.env.current_episode().episode_id}, seed {seed}")
            self.agent.reset()
            done = False
            not_done_masks = torch.zeros(1,device=self.device,dtype=torch.bool,)

            ep_reward = 0
            while not done:
                batch = batch_obs([obs],device=self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)
                step_data = self.agent.step_action(batch,not_done_masks,deterministic=True,)

                cur_skill = self.agent._cur_skills

                obs, reward, done, info = self.env.step(step_data[0])  

            
                not_done_masks = torch.tensor([not done], dtype=torch.bool, device=self.device)
                ep_reward += reward


                if args.record_video:
                    frame = observations_to_image({k: v[0] for k, v in batch.items()}, info)
                    if not not_done_masks.item():
                        frame = observations_to_image({k: v[0] * 0.0 for k, v in batch.items()}, info)
                    frame = overlay_frame(frame, info) ## with infos
                    frame = append_text_underneath_image(frame,f"num_steps: {info['num_steps']} " + f"skill: {cur_skill}" +"\t")
                    rgb_frames.append(frame)
            # episode ended
            if args.record_video:
                metrics = extract_scalars_from_info(info)
                metric_strs = []
                for k in self.config.habitat_baselines.eval_keys_to_include_in_name:
                    metric_strs.append(f"{k}={metrics[k]:.0f}")
                video_name = f"episode={self.env.current_episode().episode_id, seed}-" + "-".join(metric_strs)
                images_to_video(rgb_frames, dir_path, video_name, fps=30, verbose=True) 
            print("mission :", info['composite_success'])
            return ep_reward, info['num_steps'], (info['composite_stage_goals'],info['composite_success'])
if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evals_per_ep", type=int, default=1)
    parser.add_argument("--num_environments", type=int, default=1)
    parser.add_argument("--episode", type=str, default='21')
    parser.add_argument("--task", type=str, default='rl_hierarchical')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ask_lambda", type=float, default=0.01, help="weight on communication penalty term")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--policy",   type=str, default='ppo')
    parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
    parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
    parser.add_argument("--n_itr", type=int, default=1000, help="Number of iterations of the learning algorithm")
    parser.add_argument("--traj_per_itr", type=int, default=10)
    parser.add_argument("--save_name", type=str, required=True, help="path to folder containing policy and run details")
    parser.add_argument("--logdir", type=str, default="./log/")          # Where to log diagnostics to
    parser.add_argument("--record_video", default=False, action='store_true')
    #args = parser.parse_args()

    if sys.argv[1] == 'llm':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        output_dir = os.path.join(args.logdir, args.policy, args.task, "llm")
        policy = torch.load(output_dir + "/acmodel.pt")
        policy.eval()
        habitat_env = Env(args,policy)
        eval_rewards = []
        eval_lens = []
        eval_success = []
        eval_call_num = []
        for i in range(10):
            seed = i
            setup_seed(seed)
            reward, len, success, call_num = habitat_env.eval(trajs=1, seed=seed)
            success_rate = (success[0]['stage_0_5_success'], success[0]['stage_1_success'], success[1])
            eval_rewards.append(reward)
            eval_lens.append(len)
            eval_success.append(success_rate)
            eval_call_num.append(call_num)
        print(np.mean(eval_rewards))
        print(np.mean(eval_lens))
        print(np.mean(eval_success,0))
        print(np.mean(eval_call_num))
    elif sys.argv[1] == 'fix':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        habitat_env = Env(args)
        eval_rewards = []
        eval_lens = []
        eval_success = []

        for i in range(10):
            setup_seed(i)
            reward, len, success = habitat_env.eval_fix(trajs=1, seed=i)
            success_rate = (success[0]['stage_0_5_success'], success[0]['stage_1_success'], success[1])
            eval_rewards.append(reward)
            eval_lens.append(len)
            eval_success.append(success_rate)
        print(np.mean(eval_rewards))
        print(np.mean(eval_lens))
        print(np.mean(eval_success,0))        
    # output_dir = os.path.join(args.logdir, args.policy, args.task, "llm2")
    # policy = torch.load(output_dir + "/acmodel.pt")
    # policy.eval()
    # habitat_env = Env(args,policy)

    # habitat_env = Env(args)
    # habitat_env.train()


    # ################# baseline eval ###############
    # eval_rewards = []
    # eval_lens = []
    # eval_success = []

    # for i in range(10):
    #     setup_seed(i)
    #     reward, len, success = habitat_env.eval_fix(trajs=1, seed=i)
    #     success_rate = (success[0]['stage_0_5_success'], success[0]['stage_1_success'], success[1])
    #     eval_rewards.append(reward)
    #     eval_lens.append(len)
    #     eval_success.append(success_rate)
    # print(np.mean(eval_rewards))
    # print(np.mean(eval_lens))
    # print(np.mean(eval_success,0))


    # ################# llm eval ###############


