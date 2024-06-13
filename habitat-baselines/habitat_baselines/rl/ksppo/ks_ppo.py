#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import inspect
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import numpy as np

from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ksppo.ks_rollout_storage import KickstartingStorage,Pretrain_RolloutBuffer
from habitat_baselines.rl.ksppo.ks_policy import KSPolicy
from habitat_baselines.utils.common import (
    LagrangeInequalityCoefficient,
    inference_mode,
    kl_loss,
    mse_loss,
    was_loss,
)

from habitat_baselines.rl.ppo.ppo import PPO
EPS_PPO = 1e-5


@baseline_registry.register_updater
class KSPPO(PPO):
    @classmethod
    def from_config(cls, actor_critic: KSPolicy, ppo_config, ksppo_config):
        ppo_config = {k.lower(): v for k, v in ppo_config.items()}
        ksppo_config = {k.lower(): v for k, v in ksppo_config.items()}
        config = {**ppo_config, **ksppo_config}
        param_dict = dict(actor_critic=actor_critic)
        sig = inspect.signature(cls.__init__)
        for p in sig.parameters.values():
            if p.name == "self" or p.name in param_dict:
                continue

            assert p.name in config, "{} parameter '{}' not in config".format(
                cls.__name__, p.name
            )

            param_dict[p.name] = config[p.name]

        return cls(**param_dict)
    def __init__(
        self,
        actor_critic: KSPolicy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        init_kickstarting_coef: float, ##
        min_kickstarting_coef: float, ##
        kickstarting_coef_descent: float, ##
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = False,
        use_normalized_advantage: bool = True,
        entropy_target_factor: float = 0.0,
        use_adaptive_entropy_pen: bool = False,
    ) -> None:
        super().__init__(
            actor_critic,
            clip_param,
            ppo_epoch,
            num_mini_batch,
            value_loss_coef,
            entropy_coef,
            lr,
            eps,
            max_grad_norm,
            use_clipped_value_loss,
            use_normalized_advantage,
            entropy_target_factor,
            use_adaptive_entropy_pen,
        )
        self.kickstarting_coef = init_kickstarting_coef
        self.min_kickstarting_coef = min_kickstarting_coef
        self.kickstarting_coef_descent = kickstarting_coef_descent
        self.kickstarting_update_num = 0
        #self.critic_optimizer = torch.optim.Adam(self.actor_critic.critic.parameters(),lr=lr,eps=eps)
        self.optimizer = torch.optim.Adam([
                                        {'params': self.actor_critic.multihead.multi_heads_action_distribution[0].parameters(),'lr' :lr,'eps':eps},
                                        {'params': self.actor_critic.multihead.multi_heads_action_distribution[2].parameters(),'lr' :lr,'eps':eps}])
        
        # self.optimizer = torch.optim.Adam([
        #                                 {'params': self.actor_critic.multihead.multi_heads_action_distribution[0].parameters(),'lr' :lr,'eps':eps},
        #                                 {'params': self.actor_critic.multihead.multi_heads_action_distribution[1].parameters(),'lr' :lr,'eps':eps},
        #                                 {'params': self.actor_critic.multihead.multi_heads_action_distribution[2].parameters(),'lr' :lr,'eps':eps},
        #                                 {'params': self.actor_critic.multihead.multi_heads_action_distribution[3].parameters(),'lr' :lr,'eps':eps},
        #                                 ])
        #self.optimizer = torch.optim.Adam(self.actor_critic.multihead.multi_heads_action_distribution[1].parameters(),lr=lr,eps=eps)

    def _update_from_batch(self, batch, epoch, rollouts, learner_metrics):
        """
        Performs a gradient update from the minibatch.
        """

        def record_min_mean_max(t: torch.Tensor, prefix: str):
            for name, op in (
                ("min", torch.min),
                ("mean", torch.mean),
                ("max", torch.max),
            ):
                learner_metrics[f"{prefix}_{name}"].append(op(t))

        self._set_grads_to_none()

        (
            values,
            ks_loss,
            #skill_classify_loss,
            action_log_probs,
            dist_entropy,
            _,
            aux_loss_res,
        ) = self._evaluate_actions(
            batch["observations"],
            batch["recurrent_hidden_states"],
            batch["prev_actions"],
            batch["masks"],
            batch["actions"],
            batch["teacher_actions_data"],
            batch["teacher_skills"],
            batch["rnn_build_seq_info"],
        )

        ratio = torch.exp(action_log_probs - batch["action_log_probs"])

        surr1 = batch["advantages"] * ratio
        surr2 = batch["advantages"] * (
            torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
        )
        action_loss = -torch.min(surr1, surr2)

        values = values.float()
        orig_values = values

        if self.use_clipped_value_loss:
            delta = values.detach() - batch["value_preds"]
            value_pred_clipped = batch["value_preds"] + delta.clamp(
                -self.clip_param, self.clip_param
            )

            values = torch.where(
                delta.abs() < self.clip_param,
                values,
                value_pred_clipped,
            )

        value_loss = 0.5 * F.mse_loss(
            values, batch["returns"], reduction="none"
        )

        if "is_coeffs" in batch:
            assert isinstance(batch["is_coeffs"], torch.Tensor)
            ver_is_coeffs = batch["is_coeffs"].clamp(max=1.0)
            mean_fn = lambda t: torch.mean(ver_is_coeffs * t)
        else:
            mean_fn = torch.mean

        action_loss, value_loss, dist_entropy = map(
            mean_fn,
            (action_loss, value_loss, dist_entropy),
        )
        
      
        all_losses = [
            self.kickstarting_coef * ks_loss ,
            self.value_loss_coef * value_loss,
            action_loss,
        ]

        if isinstance(self.entropy_coef, float):
            all_losses.append(-self.entropy_coef * dist_entropy)
        else:
            all_losses.append(self.entropy_coef.lagrangian_loss(dist_entropy))

        all_losses.extend(v["loss"] for v in aux_loss_res.values())

        total_loss = torch.stack(all_losses).sum()

        total_loss = self.before_backward(total_loss)
        total_loss.backward()
        self.after_backward(total_loss)

        grad_norm = self.before_step()
        self.optimizer.step()
        #self.critic_optimizer.step()
        self.after_step()

        with inference_mode():
            if "is_coeffs" in batch:
                record_min_mean_max(batch["is_coeffs"], "ver_is_coeffs")
            record_min_mean_max(orig_values, "value_pred")
            record_min_mean_max(ratio, "prob_ratio")
            
            learner_metrics["ks_loss"].append(ks_loss)
            #learner_metrics["skill_classify_loss"].append(skill_classify_loss)
            learner_metrics["kickstarting_coef"].append(self.kickstarting_coef)
            learner_metrics["value_loss"].append(value_loss)
            learner_metrics["action_loss"].append(action_loss)
            learner_metrics["dist_entropy"].append(dist_entropy)
            
            if epoch == (self.ppo_epoch - 1):
                learner_metrics["ppo_fraction_clipped"].append(
                    (ratio > (1.0 + self.clip_param)).float().mean()
                    + (ratio < (1.0 - self.clip_param)).float().mean()
                )

            learner_metrics["grad_norm"].append(grad_norm)
            if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
                learner_metrics["entropy_coef"].append(
                    self.entropy_coef().detach()
                )

            for name, res in aux_loss_res.items():
                for k, v in res.items():
                    learner_metrics[f"aux_{name}_{k}"].append(v.detach())

            if "is_stale" in batch:
                assert isinstance(batch["is_stale"], torch.Tensor)
                learner_metrics["fraction_stale"].append(
                    batch["is_stale"].float().mean()
                )

    def pretrain_update(
        self,
        replaybuffer,
    ) -> Dict[str, float]:
        
        learner_metrics: Dict = {}

        losses = []
        ks_losses = []
        skill_classify_losses = []
        for i in range(replaybuffer.ppo_epoch): ## epochs
            for batch in replaybuffer.sample(batch_size=256,smooth=False):
                obs_batch, action_batch, skill_batch, returns,values_preds, masks,= batch
                prev_actions = None
                rnn_hidden_states = None
                rnn_build_seq_info = None
                features, rnn_hidden_states, aux_loss_state = self.actor_critic.net(
                    obs_batch,
                    rnn_hidden_states,
                    prev_actions,
                    masks,
                    rnn_build_seq_info,
                )
                #distribution = self.actor_critic.action_distribution(features, self.actor_critic.net.feature_state,teacher_weight=skill_batch)
                distribution = self.actor_critic.action_distribution(features)
                #action = distribution.mean
                values = self.actor_critic.critic(features)
                #skill_classify_loss = self.actor_critic.action_distribution.skill_weight_loss(teacher_weight=skill_batch)

                #action_loss = kl_loss(action_batch, distribution)
                #action_loss = mse_loss(action_batch, distribution)
                ks_loss = was_loss(action_batch, distribution)
                if self.use_clipped_value_loss:
                    delta = values.detach() - values_preds
                    value_pred_clipped = values_preds + delta.clamp(
                        -self.clip_param, self.clip_param
                    )

                    values = torch.where(
                        delta.abs() < self.clip_param,
                        values,
                        value_pred_clipped,
                    )
                value_loss = 0.5 * F.mse_loss(values, returns)
                #print(action_loss.item())
                # len(torch.where(skill_batch==3)[0])
                print("training data components:", torch.mean(skill_batch,dim=0))

                total_loss = ks_loss + self.value_loss_coef * value_loss#+ skill_classify_loss
                #loss = action_loss 
                self.optimizer.zero_grad()
                total_loss = self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)
                grad_norm = self.before_step()
                self.optimizer.step()
                self.after_step()
                losses.append(total_loss.item())
                ks_losses.append(ks_loss.item())
                #skill_classify_losses.append(skill_classify_loss.item())
            print(f"pretrain_loss {i}:  {np.mean(losses)}")
        learner_metrics['pretrain_loss'] = np.mean(losses)
        learner_metrics['ks_loss'] = np.mean(ks_losses)
        #learner_metrics['skill_classify_loss'] = np.mean(skill_classify_losses)
            
        return learner_metrics

    def update(
        self,
        rollouts: KickstartingStorage,
    ) -> Dict[str, float]:
        advantages = self.get_advantages(rollouts)

        learner_metrics: Dict[str, List[Any]] = collections.defaultdict(list)

        for epoch in range(self.ppo_epoch):
            profiling_wrapper.range_push("KSPPO.update epoch")
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for _bid, batch in enumerate(data_generator):
                self._update_from_batch(
                    batch, epoch, rollouts, learner_metrics
                )
            profiling_wrapper.range_pop()  # PPO.update epoch
            
        self._update_kickstarting_coef()

        self._set_grads_to_none()

        with inference_mode():
            return {
                k: float(
                    torch.stack(
                        [torch.as_tensor(v, dtype=torch.float32) for v in vs]
                    ).mean()
                )
                for k, vs in learner_metrics.items()
            }    

        
    def _update_kickstarting_coef(self):
        self.kickstarting_update_num += 1
        if self.kickstarting_coef <= self.min_kickstarting_coef:
            self.kickstarting_coef = self.min_kickstarting_coef
        else:
            self.kickstarting_coef -= self.kickstarting_coef_descent

        ## step num = env_num * step  -> ks_num = stop_num / step / env_num
        if self.kickstarting_update_num >= 1e3:
            self.kickstarting_coef = 0

