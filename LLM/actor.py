#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   actor.py
@Time    :   2023/08/01 17:07:08
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
from typing import Any, Dict, List, Optional, Tuple
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.hrl.hierarchical_policy import HierarchicalPolicy
import torch
from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.utils.common import get_num_actions
from habitat_baselines.rl.ppo.policy import Policy, PolicyActionData
from .planner import Pick_Place_Planner
from .comm_net import Comm_Net
class FixedHighLevelPolicy(HighLevelPolicy):
    _solution_actions: List[Tuple[str, List[str]]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._solution_actions = self._parse_solution_actions(
            self._pddl_prob.solution
        )

        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)
    def _parse_solution_actions(self, solution):
        solution_actions = []
        for i, hl_action in enumerate(solution):
            sol_action = (
                hl_action.name,
                [x.name for x in hl_action.param_values],
            )
            solution_actions.append(sol_action)

            if self._config.add_arm_rest and i < (len(solution) - 1):
                solution_actions.append(parse_func("reset_arm(0)"))

        # Add a wait action at the end.
        solution_actions.append(parse_func("wait(30)"))

        return solution_actions
    def apply_mask(self, mask):
        """
        Apply the given mask to the next skill index.

        Args:
            mask: Binary mask of shape (num_envs, ) to be applied to the next
                skill index.
        """
        self._next_sol_idxs *= mask.cpu().view(-1)
    def _get_next_sol_idx(self, batch_idx, immediate_end):
        """
        Get the next index to be used from the list of solution actions.

        Args:
            batch_idx: The index of the current environment.

        Returns:
            The next index to be used from the list of solution actions.
        """
        if self._next_sol_idxs[batch_idx] >= len(self._solution_actions):
            baselines_logger.info(
                f"Calling for immediate end with {self._next_sol_idxs[batch_idx]}"
            )
            immediate_end[batch_idx] = True
            return len(self._solution_actions) - 1
        else:
            return self._next_sol_idxs[batch_idx].item()

    def get_next_skill(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        plan_masks,
        deterministic,
        log_info,
    ):
        next_skill = torch.zeros(self._num_envs)
        skill_args_data = [None for _ in range(self._num_envs)]
        immediate_end = torch.zeros(self._num_envs, dtype=torch.bool)
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan == 1.0:
                use_idx = self._get_next_sol_idx(batch_idx, immediate_end)

                skill_name, skill_args = self._solution_actions[use_idx]
                baselines_logger.info(
                    f"Got next element of the plan with {skill_name}, {skill_args}"
                )
                if skill_name not in self._skill_name_to_idx:
                    raise ValueError(
                        f"Could not find skill named {skill_name} in {self._skill_name_to_idx}"
                    )
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]

                skill_args_data[batch_idx] = skill_args  # type: ignore[call-overload]

                self._next_sol_idxs[batch_idx] += 1

        return next_skill, skill_args_data, immediate_end, {}

class FixHierarchicalPolicy(HierarchicalPolicy):
    def __init__(self,config, full_config, observation_space, action_space, num_envs):
        super().__init__(config, full_config, observation_space, action_space, num_envs)
        self.hidden_state_shape = (self.num_recurrent_layers,full_config.habitat_baselines.rl.ppo.hidden_size)
        self._high_level_policy = FixedHighLevelPolicy(config.hierarchical_policy.high_level_policy, self._pddl, num_envs, self._name_to_idx, observation_space, action_space,)

    @property
    def num_recurrent_layers(self):
        if self._high_level_policy.num_recurrent_layers != 0:
            return self._high_level_policy.num_recurrent_layers
        else:
            return self._skills[0].num_recurrent_layers


class LLMHighLevelPolicy(HighLevelPolicy):

    def __init__(self, config, pddl_problem, num_envs,skill_name_to_idx, observation_space,action_space, planner):
        super().__init__(config, pddl_problem, num_envs,skill_name_to_idx, observation_space,action_space)
        
        self.planner = planner
        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)
        self.current_skill = None

    def apply_mask(self, mask):
        self._next_sol_idxs *= mask.cpu().view(-1)

    def _get_next_sol_idx(self, batch_idx, immediate_end):
        if self._next_sol_idxs[batch_idx] >= len(self._solution_actions):
            #print(f"Calling for immediate end with {self._next_sol_idxs[batch_idx]}")

            immediate_end[batch_idx] = True
            return len(self._solution_actions) - 1
        else:
            return self._next_sol_idxs[batch_idx].item()

    def updata_solution_actions(self, observations):
        
        self._solution_actions = self.planner(observations)
        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)

    def get_next_skill(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        plan_masks,
        deterministic,
        log_info,
    ):
        next_skill = torch.zeros(self._num_envs)
        skill_args_data = [None for _ in range(self._num_envs)]
        immediate_end = torch.zeros(self._num_envs, dtype=torch.bool)
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan == 1.0:
                self.updata_solution_actions(observations)
                # self._solution_actions = self.planner(observations)
                use_idx = self._get_next_sol_idx(batch_idx, immediate_end)
                
                skill_name, skill_args = self._solution_actions[use_idx]
                self.current_skill = skill_name
                print(f"Got next element of the plan with {skill_name}, {skill_args}")
                if skill_name not in self._skill_name_to_idx:
                    raise ValueError(
                        f"Could not find skill named {skill_name} in {self._skill_name_to_idx}"
                    )
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]

                skill_args_data[batch_idx] = skill_args  # type: ignore[call-overload]

                self._next_sol_idxs[batch_idx] += 1

        return next_skill, skill_args_data, immediate_end, {}

class LLMHierarchicalPolicy(HierarchicalPolicy):
    def __init__(self,config, full_config, observation_space, action_space, num_envs):
        super().__init__(config, full_config, observation_space, action_space, num_envs)
        self.hidden_state_shape = (self.num_recurrent_layers,full_config.habitat_baselines.rl.ppo.hidden_size)
        ## define by LLM
        self.planner = Pick_Place_Planner() ## create_plan
        self._high_level_policy = LLMHighLevelPolicy(config.hierarchical_policy.high_level_policy, self._pddl, num_envs, self._name_to_idx, observation_space, action_space,self.planner)
        self.comm_net = Comm_Net(None, 2)
    @property
    def num_recurrent_layers(self):
        if self._high_level_policy.num_recurrent_layers != 0:
            return self._high_level_policy.num_recurrent_layers
        else:
            return self._skills[0].num_recurrent_layers
        
    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        masks_cpu = masks.cpu()
        log_info: List[Dict[str, Any]] = [{} for _ in range(self._num_envs)]
        self._high_level_policy.apply_mask(masks_cpu)  # type: ignore[attr-defined]

        call_high_level: torch.BoolTensor = torch.zeros(
            (self._num_envs,), dtype=torch.bool
        )
        bad_should_terminate: torch.BoolTensor = torch.zeros(
            (self._num_envs,), dtype=torch.bool
        )

        hl_wants_skill_term = self._high_level_policy.get_termination(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            self._cur_skills,
            log_info,
        )
        # Initialize empty action set based on the overall action space.
        actions = torch.zeros(
            (self._num_envs, get_num_actions(self._action_space)),
            device=masks.device,
        )

        grouped_skills = self._broadcast_skill_ids(
            self._cur_skills,
            sel_dat={
                "observations": observations,
                "rnn_hidden_states": rnn_hidden_states,
                "prev_actions": prev_actions,
                "masks": masks,
                "actions": actions,
                "hl_wants_skill_term": hl_wants_skill_term,
            },
            # Only decide on skill termination if the episode is active.
            should_adds=masks,
        )

        # Check if skills should terminate.
        for skill_id, (batch_ids, dat) in grouped_skills.items():
            if skill_id == -1:
                # Policy has not prediced a skill yet.
                call_high_level[batch_ids] = 1.0
                continue
            # TODO: either change name of the function or assign actions somewhere
            # else. Updating actions in should_terminate is counterintuitive

            (
                call_high_level[batch_ids],
                bad_should_terminate[batch_ids],
                actions[batch_ids],
            ) = self._skills[skill_id].should_terminate(
                **dat,
                batch_idx=batch_ids,
                log_info=log_info,
                skill_name=[
                    self._idx_to_name[self._cur_skills[i].item()]
                    for i in batch_ids
                ],
            )

        # Always call high-level if the episode is over.
        #call_high_level = call_high_level | (~masks_cpu).view(-1)
        
        #call_high_level = torch.tensor([True])
        call_high_level = self.comm_net(observations, masks_cpu, call_high_level, self._cur_skills)
        # If any skills want to terminate invoke the high-level policy to get
        # the next skill.
        hl_terminate = torch.zeros(self._num_envs, dtype=torch.bool)
        hl_info: Dict[str, Any] = self._high_level_policy.create_hl_info()
        if call_high_level.sum() > 0:
            (
                new_skills,
                new_skill_args,
                hl_terminate,
                hl_info,
            ) = self._high_level_policy.get_next_skill(
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                call_high_level,
                deterministic,
                log_info,
            )

            sel_grouped_skills = self._broadcast_skill_ids(
                new_skills,
                sel_dat={},
                should_adds=call_high_level,
            )

            for skill_id, (batch_ids, _) in sel_grouped_skills.items():
                self._skills[skill_id].on_enter(
                    [new_skill_args[i] for i in batch_ids],
                    batch_ids,
                    observations,
                    rnn_hidden_states,
                    prev_actions,
                )
                if "rnn_hidden_states" not in hl_info:
                    rnn_hidden_states[batch_ids] *= 0.0
                    prev_actions[batch_ids] *= 0
                elif self._skills[skill_id].has_hidden_state:
                    raise ValueError(
                        f"The code does not currently support neural LL and neural HL skills. Skill={self._skills[skill_id]}, HL={self._high_level_policy}"
                    )
            #self._cur_skills = ((~call_high_level) * self._cur_skills) + (call_high_level * new_skills)
            self._cur_skills = new_skills

        grouped_skills = self._broadcast_skill_ids(
            self._cur_skills,
            sel_dat={
                "observations": observations,
                "rnn_hidden_states": rnn_hidden_states,
                "prev_actions": prev_actions,
                "masks": masks,
            },
        )
        for skill_id, (batch_ids, batch_dat) in grouped_skills.items():
            action_data = self._skills[skill_id].act(
                observations=batch_dat["observations"],
                rnn_hidden_states=batch_dat["rnn_hidden_states"],
                prev_actions=batch_dat["prev_actions"],
                masks=batch_dat["masks"],
                cur_batch_idx=batch_ids,
            )

            # LL skills are not allowed to terminate the overall episode.
            actions[batch_ids] += action_data.actions
            # Add actions from apply_postcond
            rnn_hidden_states[batch_ids] = action_data.rnn_hidden_states
        actions[:, self._stop_action_idx] = 0.0

        should_terminate = bad_should_terminate | hl_terminate
        if should_terminate.sum() > 0:
            # End the episode where requested.
            for batch_idx in torch.nonzero(should_terminate):
                print(f"Calling stop action for batch {batch_idx}, bad_should_terminate{bad_should_terminate}, hl_terminate{hl_terminate}")
                actions[batch_idx, self._stop_action_idx] = 1.0

        action_kwargs = {
            "rnn_hidden_states": rnn_hidden_states,
            "actions": actions,
        }
        action_kwargs.update(hl_info)

        return PolicyActionData(
            take_actions=actions,
            policy_info=log_info,
            should_inserts=call_high_level,
            **action_kwargs,
        )
if __name__ == '__main__':
   pass

