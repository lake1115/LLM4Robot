#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   habitat_env.py
@Time    :   2023/08/21 09:55:49
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import os
import gym
import numpy as np

import habitat
import habitat.gym
from habitat.config.default import get_agent_config
from habitat.utils.visualizations.utils import observations_to_image, overlay_frame, append_text_underneath_image
from habitat.config.default import patch_config
import hydra
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.utils.info_dict import extract_scalars_from_info
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch, apply_obs_transforms_obs_space, get_active_obs_transforms
from habitat.utils.visualizations.utils import images_to_video


# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

from LLM.actor import FixHierarchicalPolicy, LLMHierarchicalPolicy
import torch
import time
import sys
import algos
ROBOT_SKILL_TO_IDX = {
   "pick": 0,
   "place": 1,
   "nav": 3,
   "nav2": 4,  ## same as nav, maybe useless
   "reset": 5,
   "init": -1,
}
IDX_TO_ROBOT_SKILL = dict(zip(ROBOT_SKILL_TO_IDX.values(), ROBOT_SKILL_TO_IDX.keys()))
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


class Env:
    def __init__(self, args, policy=None):
        config_path = "habitat-baselines/habitat_baselines/config"
        config_name = "rearrange/" + args.task + ".yaml"
        self.gamma = args.gamma
        self.lam = args.lam
        self.ask_lambda = args.ask_lambda
        self.batch_size = args.batch_size
        self.episode_idx = args.episode_idx
        self.device = torch.device(args.device)
        self.seed = args.seed
        self.ckpt_num = args.ckpt_num
        self.eval_type = args.eval_type
        self.record_video = args.record_video

        hydra.initialize(config_path=config_path, job_name=f"{args.eval_type}")
        config = hydra.compose(config_name=config_name, overrides=['habitat_baselines.evaluate=True',f'habitat_baselines.num_environments=1',f'habitat_baselines.eval.evals_per_ep={args.evals_per_ep}'])
        self.config = insert_render_options(patch_config(config))
        self.env = habitat.gym.make_gym_from_config(self.config)
        #self.env.env.env.habitat_env.episode_iterator=None  ## set no episode iterator
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(self.env.observation_space, self.obs_transforms)
        action_space = self.env.action_space
        orig_action_space = self.env.env.original_action_space 
        self.logger = algos.create_logger(args)   
        if args.eval_type.lower() == "fix":     
            self.agent = FixHierarchicalPolicy(self.config.habitat_baselines.rl.policy,self.config, observation_space, action_space, orig_action_space, 1, self.device) 
        elif args.eval_type.lower() == "always":     
            self.agent = LLMHierarchicalPolicy(self.config.habitat_baselines.rl.policy,self.config, observation_space, action_space, orig_action_space, 1, self.device, always_call=True)
        elif self.eval_type.lower() == "llm" or self.eval_type.lower() == "random":
            self.agent = LLMHierarchicalPolicy(self.config.habitat_baselines.rl.policy,self.config, observation_space, action_space, orig_action_space, 1, self.device) 

            if policy is not None:
                ## communication network for ask
                print("load policy!")
                self.agent._high_level_policy.comm_net = policy.to(self.device)        
            
            self.buffer = algos.Buffer(self.gamma, self.lam, self.device)    
           
            self.ppo_algo = algos.PPO(self.agent._high_level_policy.comm_net, device=self.device, save_path=self.logger.dir, batch_size=self.batch_size)

            self.n_itr = args.n_itr
            self.traj_per_itr = args.traj_per_itr
            self.total_steps = 0
         

    def train(self):
        start_time = time.time()
        ckpt_itr = 0
        for itr in range(1, self.n_itr+1):
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
            eval_reward,eval_comm_reward, eval_len, eval_success, eval_interactions = self.eval(trajs=1, seed=self.seed)
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
            if itr % (self.n_itr//self.ckpt_num) == 0:
                self.ppo_algo.save(f"ckpt.{ckpt_itr}")
                ckpt_itr += 1
    
    def collect(self, seed=None):
        if seed is not None:
            self.env.seed(seed)

        with torch.no_grad():
            buffer = algos.Buffer(self.gamma, self.lam)
            if self.episode_idx is None:
                obs = self.env.reset()
            else:
                obs = self.env.episode_reset(self.episode_idx)
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
                    repeat_feedback = torch.tensor([-1.])
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
       
        if self.record_video:
            rgb_frames = []
            dir_path = os.path.join(self.logger.dir, 'video_dir')
            try:
                os.makedirs(dir_path)
            except OSError:
                pass
        with torch.no_grad():
            if self.episode_idx is None:
                obs = self.env.reset()
            else:
                obs = self.env.episode_reset(self.episode_idx)
            print(f"current episode {self.env.current_episode().episode_id}, seed {seed}")
            self.agent.reset()
            done = False
            not_done_masks = torch.zeros(1,device=self.device,dtype=torch.bool,)
            pre_skill = None
            ep_reward = 0
            ep_comm_reward = 0
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
                ep_comm_reward += comm_reward
                ep_reward += reward


                if self.record_video:
                    frame = observations_to_image({k: v[0] for k, v in batch.items()}, info)
                    third_rgb = {'third_rgb': batch['third_rgb'][0]}
                    if not not_done_masks.item():
                        frame = observations_to_image({k: v[0] * 0.0 for k, v in batch.items()}, info)
                        third_rgb = {'third_rgb': batch['third_rgb'][0]*0.0}
                    third_rgb_frame = observations_to_image(third_rgb, info)
                    frame = overlay_frame(frame, info) ## with infos
                    #frame = append_text_underneath_image(frame,f"num_steps: {info['num_steps']} " + self.agent.planner.dialogue_user + f"skill: {rl_skill}" +"\t"+f"call_num: {self.agent._high_level_policy.call_num}",self.agent._high_level_policy.call)
                    #third_rgb_frame = append_text_underneath_image(third_rgb_frame, self.agent._high_level_policy.planner.dialogue_user + f" Option: {IDX_TO_ROBOT_SKILL[self.agent.rl_skill.item()]}"+ f"  Ask: {self.agent._high_level_policy.call}",self.agent._high_level_policy.call)
                    rgb_frames.append(frame)
                    #rgb_frames.append(third_rgb_frame)
            # episode ended
            if self.record_video:
                metrics = extract_scalars_from_info(info)
                metric_strs = []
                for k in self.config.habitat_baselines.eval_keys_to_include_in_name:
                    metric_strs.append(f"{k}={metrics[k]:.0f}")
                video_name = f"episode={self.env.current_episode().episode_id}-"+ f"seed={seed}" #+ "-".join(metric_strs)
                images_to_video(rgb_frames, dir_path, video_name, fps=30, verbose=True) 
            print("mission completed:", info['composite_success'])
            return ep_reward,ep_comm_reward, info['num_steps'],  (info['composite_stage_goals'],info['composite_success']), self.agent._high_level_policy.call_num

    def eval_fix(self, trajs=1, seed=None):
        if seed is not None:
            self.env.seed(seed)
       
        if self.record_video:
            rgb_frames = []
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

                if self.record_video:
                    frame = observations_to_image({k: v[0] for k, v in batch.items()}, info)
                    if not not_done_masks.item():
                        frame = observations_to_image({k: v[0] * 0.0 for k, v in batch.items()}, info)
                    #frame = overlay_frame(frame, info) ## with infos
                    #frame = append_text_underneath_image(frame,f"num_steps: {info['num_steps']} " + f"skill: {cur_skill}" +"\t")
                    rgb_frames.append(frame)
            # episode ended
            if self.record_video:
                metrics = extract_scalars_from_info(info)
                metric_strs = []
                for k in self.config.habitat_baselines.eval_keys_to_include_in_name:
                    metric_strs.append(f"{k}={metrics[k]:.0f}")
                video_name = f"episode={self.env.current_episode().episode_id, seed}-" + "-".join(metric_strs)
                images_to_video(rgb_frames, dir_path, video_name, fps=30, verbose=True) 
            print("mission completed:", info['composite_success'])
            return ep_reward, ep_reward, info['num_steps'], (info['composite_stage_goals'],info['composite_success']), 0



if __name__ == '__main__':
   pass

