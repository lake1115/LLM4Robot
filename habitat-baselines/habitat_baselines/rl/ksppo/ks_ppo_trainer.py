#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.construct_single_env import construct_envs
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401.
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401.
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    NON_SCALAR_METRICS,
    extract_scalars_from_info,
    extract_scalars_from_infos,
)

from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
import sys
sys.path.append(os.getcwd())
from LLM import LLM_TeacherPolicy, Fix_TeacherPolicy
from copy import deepcopy
from habitat_baselines.rl.ksppo.ks_rollout_storage import Pretrain_RolloutBuffer, Pretrain_ReplayBuffer
@baseline_registry.register_trainer(name="ksppo")
class KSPPOTrainer(PPOTrainer):

    def __init__(self, config=None, hrl_config=None):
        super().__init__(config)
        self._teacher = None
        self.scene_idx = 3
        self.seed = 1
        self.pretrain_flag = config.habitat_baselines.rl.ksppo.pretrain_flag
        self.ideal = True

        if self.pretrain_flag:
            self.pretrain_replaybuffer = Pretrain_ReplayBuffer(step=config.habitat_baselines.rl.ppo.num_steps,n_envs=config.habitat_baselines.num_environments,ppo_epoch=config.habitat_baselines.rl.ppo.ppo_epoch)
    def _create_teacher(self):
        # TO DO
        # use self.config to define policy over teachers = ideal, LLM or learned?
        if self.ideal:
            Teacher_policy = LLM_TeacherPolicy(self.config.habitat_baselines.rl.policy,self.config, self.envs.observation_spaces[0], self.envs.action_spaces[0], self.envs.orig_action_spaces[0], self.envs.num_envs, self.device)
        else:
            Teacher_policy = Fix_TeacherPolicy(self.config.habitat_baselines.rl.policy,self.config, self.envs.observation_spaces[0], self.envs.action_spaces[0], self.envs.orig_action_spaces[0], self.envs.num_envs, self.device)
        
        return Teacher_policy

    def _init_train(self, resume_state=None):
        super()._init_train(resume_state)
        self._teacher = self._create_teacher()

    def _init_envs(self, config=None, is_eval: bool = False):
        if config is None:
            config = self.config

        self.envs = construct_envs( 
            self.scene_idx,
            config,
            workers_ignore_signals=is_slurm_batch_job(),
            enforce_scenes_greater_eq_environments=is_eval,
        )
        self._env_spec = EnvironmentSpec(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            orig_action_space=self.envs.orig_action_spaces[0],
        )     
        print(self.envs.current_episodes()[0])
        # random.seed(self.seed)
        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)
    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        t_sample_action = time.time()

        # Sample actions
        with inference_mode():
            step_batch = self._agent.rollouts.get_current_step(
                env_slice, buffer_index
            )

            profiling_wrapper.range_push("compute actions")

            if self.pretrain_flag:
                action_data = self._teacher.act(
                    step_batch["observations"],
                    step_batch["recurrent_hidden_states"],
                    step_batch["prev_actions"],
                    step_batch["masks"],
                    deterministic=True, ##
                )
                teacher_action_data = action_data.policy_info
                student_data = self._agent.actor_critic.act(
                    step_batch["observations"],
                    step_batch["recurrent_hidden_states"],
                    step_batch["prev_actions"],
                    step_batch["masks"],
                )
                action_data.values = student_data.values
                teacher_skills = self._teacher._cur_skills.cpu().numpy()
            else:
                teacher_data = self._teacher.act(
                    step_batch["observations"],
                    step_batch["recurrent_hidden_states"],
                    step_batch["prev_actions"],
                    step_batch["masks"],
                    deterministic=False, ##
                )
        
                action_data = self._agent.actor_critic.act(
                    step_batch["observations"],
                    step_batch["recurrent_hidden_states"],
                    step_batch["prev_actions"],
                    step_batch["masks"],
                    #teacher_skills, # get from teacher now
                )
                teacher_skills = self._agent.actor_critic.cur_skill
                #print(teacher_skills, step_batch["observations"]['obj_start_gps_compass'][:,0],step_batch["observations"]['obj_goal_gps_compass'][:,0])
                # teacher_skills = self._teacher._cur_skills.cpu().numpy()
                # action_data = teacher_data
                teacher_action_data = teacher_data.policy_info
            
                ### for HKS don't use LLM teacher
                # action_data, teacher_action_data = self._agent.actor_critic.act(
                #     step_batch["observations"],
                #     step_batch["recurrent_hidden_states"],
                #     step_batch["prev_actions"],
                #     step_batch["masks"],
                #     #teacher_skills, # get from teacher now
                # )
                # teacher_skills = -1* np.ones(10, dtype=np.int8) # useless
                ###

        self.pth_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        t_step_env = time.time()

        for index_env, act in zip(
            range(env_slice.start, env_slice.stop),
            action_data.env_actions.cpu().unbind(0),
        ):
            if is_continuous_action_space(self._env_spec.action_space):
                # Clipping actions to the specified limits
                act = np.clip(
                    act.numpy(),
                    self._env_spec.action_space.low,
                    self._env_spec.action_space.high,
                )
            else:
                act = act.item()
            self.envs.async_step_at(index_env, act)

        self.env_time += time.time() - t_step_env

        self._agent.rollouts.insert(
            next_recurrent_hidden_states=action_data.rnn_hidden_states,
            actions=action_data.actions,
            action_log_probs=action_data.action_log_probs,
            value_preds=action_data.values,
            teacher_action_data = teacher_action_data,
            buffer_index=buffer_index,
            should_inserts=action_data.should_inserts,
            teacher_skills=teacher_skills
        )


    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self):
        t_update_model = time.time()

        if self.pretrain_flag:
          
            losses: Dict = {}
            with inference_mode():
                step_batch = self._agent.rollouts.get_last_step()

                next_value = self._agent.actor_critic.get_value(
                    step_batch["observations"],
                    step_batch["recurrent_hidden_states"],
                    step_batch["prev_actions"],
                    step_batch["masks"],
                )

            self._agent.rollouts.compute_returns(
                next_value,
                self._ppo_cfg.use_gae,
                self._ppo_cfg.gamma,
                self._ppo_cfg.tau,
            )

            obs = deepcopy(self._agent.rollouts.buffers["observations"])
            mask = deepcopy(self._agent.rollouts.buffers["masks"])
            action = deepcopy(self._agent.rollouts.buffers["actions"])
            action_dist = deepcopy(self._agent.rollouts.buffers["teacher_actions_data"])
            skill = deepcopy(self._agent.rollouts.buffers["teacher_skills"])
            returns = deepcopy(self._agent.rollouts.buffers['returns'])
            values = deepcopy(self._agent.rollouts.buffers['value_preds'])
            self.pretrain_replaybuffer.add(obs=obs, action=action,action_dist=action_dist, mask=mask, returns = returns, values=values, cur_skill=skill)
            if self.pretrain_replaybuffer.full:
                self._agent.train()
                self._teacher.eval()
                losses = self._agent.updater.pretrain_update(self.pretrain_replaybuffer)
                #self.pretrain_replaybuffer.reset()
                
            self._agent.rollouts.after_update()
            self._agent.after_update()
        else:
            with inference_mode():
                step_batch = self._agent.rollouts.get_last_step()

                next_value = self._agent.actor_critic.get_value(
                    step_batch["observations"],
                    step_batch["recurrent_hidden_states"],
                    step_batch["prev_actions"],
                    step_batch["masks"],
                )

            self._agent.rollouts.compute_returns(
                next_value,
                self._ppo_cfg.use_gae,
                self._ppo_cfg.gamma,
                self._ppo_cfg.tau,
            )

            self._agent.train()

            losses = self._agent.updater.update(self._agent.rollouts)
            self._agent.rollouts.after_update()

            self._agent.after_update()

        self.pth_time += time.time() - t_update_model
        return losses
        
    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        resume_state = load_resume_state(self.config)
        self._init_train(resume_state)

        count_checkpoints = 0
        prev_time = 0

        if self._is_distributed:
            torch.distributed.barrier()

        resume_run_id = None
        # if resume_state is not None:
        #     ## load last ckpt by Hu Bin
        #     print("load pretrain model")
        #     checkpoint_path = self.config.habitat_baselines.checkpoint_folder + '/pretrain.pth'
        #     ckpt_dict = self.load_checkpoint(
        #         checkpoint_path, map_location="cpu"
        #     )
        #     self._agent.load_ckpt_state_dict(ckpt_dict)
            
        with (
            get_writer(
                self.config,
                resume_run_id=resume_run_id,
                flush_secs=self.flush_secs,
                purge_step=int(self.num_steps_done),
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                self._agent.pre_rollout()

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                        run_id=writer.get_run_id(),
                    )

                    save_resume_state(
                        dict(
                            **self._agent.get_resume_state(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                self._agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                for buffer_index in range(self._agent.nbuffers):
                    self._compute_actions_and_step_envs(buffer_index)

                for step in range(self._ppo_cfg.num_steps):
                    is_last_step = (
                        self.should_end_early(step + 1)
                        or (step + 1) == self._ppo_cfg.num_steps
                    )

                    for buffer_index in range(self._agent.nbuffers):
                        count_steps_delta += self._collect_environment_result(
                            buffer_index
                        )

                        if (buffer_index + 1) == self._agent.nbuffers:
                            profiling_wrapper.range_pop()  # _collect_rollout_step

                        if not is_last_step:
                            if (buffer_index + 1) == self._agent.nbuffers:
                                profiling_wrapper.range_push(
                                    "_collect_rollout_step"
                                )

                            self._compute_actions_and_step_envs(buffer_index)

                    if is_last_step:
                        break

                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                losses = self._update_agent()

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    losses,
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()


    @rank0_only
    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        ## for our approach ##
        checkpoint = {"state_dict" : self._agent._actor_critic.multihead.state_dict(),
                       "config": self.config,
        }
        ##
        # checkpoint = { **self._agent.get_save_state(),
        #                "config": self.config,
        # }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state  # type: ignore

        torch.save(
            checkpoint,
            os.path.join(
                self.config.habitat_baselines.checkpoint_folder, file_name
            ),
        )
        torch.save(
            checkpoint,
            os.path.join(
                self.config.habitat_baselines.checkpoint_folder, "latest.pth"
            ),
        )
    def load_state_dict(self, state: Dict) -> None:
        self._agent._actor_critic.multihead.load_state_dict(state["state_dict"])
        # if self._updater is not None:
        #     self._updater.load_state_dict(
        #         {
        #             "actor_critic." + k: v
        #             for k, v, in state["state_dict"].items()
        #         }
        #     )
        #     if "optim_state" in state:
        #         self._updater.optimizer.load_state_dict(state["optim_state"])
        #     if "lr_sched_state" in state:
        #         self._lr_scheduler.load_state_dict(state["lr_sched_state"][0])

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Some configurations require not to load the checkpoint, like when using
        # a hierarchial policy
        if self.config.habitat_baselines.eval.should_load_ckpt:
            #checkpoint_path = 'task/ks_depth_student_rearrange_finetune/checkpoints/latest.pth'
            # map_location="cpu" is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
            step_id = ckpt_dict["extra_state"]["step"]
            print(step_id)
        else:
            ckpt_dict = {"config": None}

        config = self._get_resume_state_config_or_new_config(
            ckpt_dict["config"]
        )

        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        if len(self.config.habitat_baselines.eval.video_option) > 0:
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors
            extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
            with read_write(agent_sensors):
                agent_sensors.update(extra_sensors)
            with read_write(config):
                if config.habitat.gym.obs_keys is not None:
                    for render_view in extra_sensors.values():
                        if render_view.uuid not in config.habitat.gym.obs_keys:
                            config.habitat.gym.obs_keys.append(
                                render_view.uuid
                            )
                config.habitat.simulator.debug_render = True

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        self._init_envs(config, is_eval=True)

        self._agent = self._create_agent(None)
        # self._teacher = self._create_teacher()
        action_shape, discrete_actions = get_action_space_info(
            self._agent.policy_action_space
        )

        if self._agent.actor_critic.should_load_agent_state:
            self.load_state_dict(ckpt_dict)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )

        test_recurrent_hidden_states = torch.zeros(
            (
                self.config.habitat_baselines.num_environments,
                *self._agent.hidden_state_shape,
            ),
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        rgb_frames: List[List[np.ndarray]] = [
            [] for _ in range(self.config.habitat_baselines.num_environments)
        ]
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = (
            self.config.habitat_baselines.test_episode_count
        )
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        self._agent.eval()
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and self.envs.num_envs > 0
        ):
            current_episodes_info = self.envs.current_episodes()

            with inference_mode():

                action_data = self._agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    for i, should_insert in enumerate(
                        action_data.should_inserts
                    ):
                        if should_insert.item():
                            test_recurrent_hidden_states[
                                i
                            ] = action_data.rnn_hidden_states[i]
                            prev_actions[i].copy_(action_data.actions[i])  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(self._env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            policy_infos = self._agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    if not not_done_masks[i].item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()}, infos[i]
                        )
                    frame = overlay_frame(frame, infos[i])
                    rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if (
                        len(self.config.habitat_baselines.eval.video_option)
                        > 0
                    ):
                        generate_video(
                            video_option=self.config.habitat_baselines.eval.video_option,
                            video_dir=self.config.habitat_baselines.video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes_info[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(infos[i]),
                            fps=self.config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        rgb_frames[i] = []

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values()]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()
