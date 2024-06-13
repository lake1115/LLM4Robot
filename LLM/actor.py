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
from copy import deepcopy
from collections import deque
import numpy as np
from habitat_baselines.utils.common import get_action_space_info, inference_mode, is_continuous_action_space

class FixedHighLevelPolicy(HighLevelPolicy):
    _solution_actions: List[Tuple[str, List[str]]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._solution_actions = self._parse_solution_actions(
            self._pddl_prob.solution
        )
        self._solution_actions = self._solution_actions[:-1]
        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)

    def reset(self):
        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)
        self._solution_actions = self._parse_solution_actions(
            self._pddl_prob.solution
        )
        self._solution_actions = self._solution_actions[:-1]

        
    def _parse_solution_actions(self, solution):
        solution_actions = []
        for i, hl_action in enumerate(solution):
            sol_action = (
                hl_action.name,
                [x.name for x in hl_action.param_values],
            )
            ## use nav_to_receptacle by Hu Bin
            if 'TARGET_goal0|0' in sol_action[1] and 'nav' in sol_action[0]:
                sol_action =  (
                'nav_to_receptacle',
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
    def __init__(self,config, full_config, observation_space, action_space, orig_action_space, num_envs, device='cpu'):
        super().__init__(config, full_config, observation_space, orig_action_space, num_envs)
        self.device = device
        self.hidden_state_shape = (self.num_recurrent_layers,full_config.habitat_baselines.rl.ppo.hidden_size)
        self._high_level_policy = FixedHighLevelPolicy(config.hierarchical_policy.high_level_policy, self._pddl, num_envs, self._name_to_idx, observation_space, action_space,)
####################
        #self.planner = Pick_Place_Planner() ## create_plan
        ##########
        self.num_envs = num_envs
        self.action_space = action_space
        self.action_shape, self.discrete_actions = get_action_space_info(self.action_space)
        self.recurrent_hidden_states = torch.zeros((self.num_envs,*self.hidden_state_shape,),device=self.device)
        self.prev_actions = torch.zeros(self.num_envs,*self.action_shape,device=self.device,dtype=torch.long if self.discrete_actions else torch.float,)
        self.to(self.device)
    @property
    def num_recurrent_layers(self):
        if self._high_level_policy.num_recurrent_layers != 0:
            return self._high_level_policy.num_recurrent_layers
        else:
            return self._skills[0].num_recurrent_layers
    def step_action(self, observations, not_done_masks, deterministic=False):
        action_data = self.act(observations, self.recurrent_hidden_states, self.prev_actions,not_done_masks,deterministic=deterministic,)
        ##############
        #self.planner(observations, self._high_level_policy._skill_name_to_idx[self._high_level_policy._solution_actions[self._high_level_policy._next_sol_idxs-1][0]])
        ###############
        ## always insert perv_action
       
        self.recurrent_hidden_states = action_data.rnn_hidden_states
        self.prev_actions.copy_(action_data.actions)  

        if is_continuous_action_space(self.action_space):
            # Clipping actions to the specified limits
            step_data = [np.clip(a.numpy(),self.action_space.low,self.action_space.high,) for a in action_data.env_actions.cpu()]
        else:
            step_data = [a.item() for a in action_data.env_actions.cpu()]      
        return step_data
    
    def reset(self):
        self._high_level_policy.reset()
        self._cur_skills = torch.tensor([-1], dtype=torch.long)
        self.recurrent_hidden_states = torch.zeros((self.num_envs,*self.hidden_state_shape,),device=self.device)
        self.prev_actions = torch.zeros(self.num_envs,*self.action_shape,device=self.device,dtype=torch.long if self.discrete_actions else torch.float,)
 
class LLMHighLevelPolicy(HighLevelPolicy):

    def __init__(self, config, pddl_problem, num_envs,skill_name_to_idx, observation_space,action_space, device, always_call=False):
        super().__init__(config, pddl_problem, num_envs,skill_name_to_idx, observation_space,action_space)
        
        self.planner = Pick_Place_Planner() ## create_plan
        self.device = device
        self.delay = 5
        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)

        self.comm_net = Comm_Net(observation_space, 2, self.delay).to(self.device)
        self.call = False
        self.call_num = -1 ## always call at first

        self.always_call = always_call

    def reset(self):
        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)
        self.call = False
        self.call_num = -1
        # print(f"[INFO]: resetting the task")
        self.planner.reset()

    def apply_mask(self, mask):
        self._next_sol_idxs *= mask.cpu().view(-1)

    def _get_next_sol_idx(self, batch_idx):
        if self._next_sol_idxs[batch_idx] >= len(self._solution_actions[batch_idx]):
            #print(f"Calling for immediate end with {self._next_sol_idxs[batch_idx]}")
            if self._solution_actions[batch_idx] == 'wait':
                self.immediate_end[batch_idx] = True
            return len(self._solution_actions[batch_idx]) - 1
        else:
            return self._next_sol_idxs[batch_idx].item()

    def updata_solution_actions(self, observations,cur_skill, hl_wants_skill_term, masks):
        
        self._solution_actions = self.planner(observations, cur_skill, hl_wants_skill_term, masks)
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
        self.immediate_end = torch.zeros(self._num_envs, dtype=torch.bool)
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan == 1.0:

                #use_idx = self._get_next_sol_idx(batch_idx)
                skill_name, skill_args = self._solution_actions[batch_idx][0]
                if skill_name == 'wait':
                    self.immediate_end[batch_idx] = True
                # if self.immediate_end:
                #     print(f"Mission Completed!!")
                # else:
                #     print(f"Got next element of the plan with {skill_name}, {skill_args}")
                if skill_name not in self._skill_name_to_idx:
                    raise ValueError(
                        f"Could not find skill named {skill_name} in {self._skill_name_to_idx}"
                    )
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]

                skill_args_data[batch_idx] = skill_args  # type: ignore[call-overload]

                self._next_sol_idxs[batch_idx] += 1

        return next_skill, skill_args_data, self.immediate_end, {}

    # def get_termination(self, his_obs, rnn_hidden_states, prev_actions, masks, cur_skills, log_info, deterministic):
    #     if self.always_call:
    #         ask_flag = torch.tensor([1],dtype=torch.bool)
    #         log_probs = None
    #         value = None
    #     else:
    #         if cur_skills == -1: ## initial
    #             ask_flag = torch.tensor([1],dtype=torch.bool)
    #             log_probs = None
    #             value = None
    #         elif len(his_obs[0]) < self.delay or cur_skills == 5: ## his_obs number need equal to delay  or reset arm
    #             ask_flag = torch.tensor([0],dtype=torch.bool)
    #             log_probs = None
    #             value = None
    #         else:
    #             dist, value = self.comm_net(his_obs, cur_skills)
    #             if deterministic:
    #                 ask_flag = torch.argmax(dist.probs,dim=1)
    #                 #print(dist.probs)
    #             else:
    #                 ask_flag = dist.sample()
    #             log_probs = dist.log_prob(ask_flag)

    #     if ask_flag:
    #         # print(f"Call LLM for help! {self.call_num}. {cur_skills}")
    #         self.call = True
    #         self.call_num += 1 
    #         self.updata_solution_actions(his_obs, cur_skills)
    #     else:
    #         self.call = False
    #     return ask_flag.cpu(), log_probs, value

    # def get_termination(self, his_obs, rnn_hidden_states, prev_actions, masks, cur_skills, log_info):
    #     if self.always_call:
    #         ask_flag = torch.ones(self._num_envs,dtype=torch.bool)
 
    #         # print(f"Call LLM for help! {self.call_num}. {cur_skills}")
    #         self.call = True
    #         self.call_num += 1 
    #         self.updata_solution_actions(his_obs, cur_skills)
    #     else:
    #         self.call = False
    #     return ask_flag.cpu()
class LLMHierarchicalPolicy(HierarchicalPolicy):
    def __init__(self,config, full_config, observation_space, action_space, orig_action_space, num_envs, device='cpu', always_call=True):
        super().__init__(config, full_config, observation_space, orig_action_space, num_envs)

        self.device = device
        self.hidden_state_shape = (self.num_recurrent_layers,full_config.habitat_baselines.rl.ppo.hidden_size)
        ## define by LLM

        self._high_level_policy = LLMHighLevelPolicy(config.hierarchical_policy.high_level_policy, self._pddl, num_envs, self._name_to_idx, observation_space, action_space,self.device, always_call)
        self.delay = 5
        self.his_obs = deque(maxlen=self.delay)
        self.cur_skills_step :int = 0
        
        self.num_envs = num_envs
        self.action_space = action_space
        self.action_shape, self.discrete_actions = get_action_space_info(self.action_space)
        self.recurrent_hidden_states = torch.zeros((self.num_envs,*self.hidden_state_shape,),device=self.device)
        self.prev_actions = torch.zeros(self.num_envs,*self.action_shape,device=self.device,dtype=torch.long if self.discrete_actions else torch.float,)
        # RL value for training
        self.rl_action = None
        self.rl_log_prob = None
        self.rl_value = None
        self.rl_skill = None

        self.to(self.device)

    def reset(self):
        self._high_level_policy.reset()
        self.his_obs.clear()
        self.cur_skills_step :int = 0
        self._cur_skills = torch.tensor([-1], dtype=torch.long)
        self.recurrent_hidden_states = torch.zeros((self.num_envs,*self.hidden_state_shape,),device=self.device)
        self.prev_actions = torch.zeros(self.num_envs,*self.action_shape,device=self.device,dtype=torch.long if self.discrete_actions else torch.float,)
        # RL value for training
        self.rl_action = None
        self.rl_log_prob = None
        self.rl_value = None
        self.rl_skill = None

    @property
    def num_recurrent_layers(self):
        if self._high_level_policy.num_recurrent_layers != 0:
            return self._high_level_policy.num_recurrent_layers
        else:
            return self._skills[0].num_recurrent_layers
        
    def rl_data(self):
        return self.rl_action, self.rl_log_prob, self.rl_value, self.rl_skill

    def step_action(self, observations, not_done_masks, deterministic=False):
        action_data = self.act(observations, self.recurrent_hidden_states, self.prev_actions,not_done_masks,deterministic=deterministic,)
 
        self.recurrent_hidden_states = action_data.rnn_hidden_states
        self.prev_actions.copy_(action_data.actions)  
    
        if is_continuous_action_space(self.action_space):
            # Clipping actions to the specified limits
            step_data = [np.clip(a.numpy(),self.action_space.low,self.action_space.high,) for a in action_data.env_actions.cpu()]
        else:
            step_data = [a.item() for a in action_data.env_actions.cpu()]      
        return step_data


    def act(self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False, ):
        masks_cpu = masks.cpu()
        log_info: List[Dict[str, Any]] = [{} for _ in range(self._num_envs)]
        self._high_level_policy.apply_mask(masks_cpu)  # type: ignore[attr-defined]

        call_high_level: torch.BoolTensor = torch.zeros((self._num_envs,), dtype=torch.bool)
        bad_should_terminate: torch.BoolTensor = torch.zeros((self._num_envs,), dtype=torch.bool)

        self.his_obs.append(deepcopy(observations))
        
        self.rl_action, self.rl_log_prob, self.rl_value = self._high_level_policy.get_termination(
            [self.his_obs],
            rnn_hidden_states,
            prev_actions,
            masks,
            self._cur_skills,
            log_info,
            deterministic,
        )
        self.rl_skill = self._cur_skills
        hl_wants_skill_term = self.rl_action.to(torch.bool)
        # Initialize empty action set based on the overall action space.
        actions = torch.zeros( (self._num_envs, get_num_actions(self._action_space)),device=masks.device,)

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

            call_high_level[batch_ids], bad_should_terminate[batch_ids], actions[batch_ids] = self._skills[skill_id].should_terminate(**dat, batch_idx=batch_ids, log_info=log_info, skill_name=[self._idx_to_name[self._cur_skills[i].item()] for i in batch_ids] )
      
        # Always call high-level if the episode is over.
        call_high_level = call_high_level | (~masks_cpu).view(-1)
        # If any skills want to terminate invoke the high-level policy to get
        # the next skill.
        hl_terminate = torch.zeros(self._num_envs, dtype=torch.bool)
        hl_info: Dict[str, Any] = self._high_level_policy.create_hl_info()
        if call_high_level.sum() > 0:
            new_skills, new_skill_args, hl_terminate, hl_info = self._high_level_policy.get_next_skill(observations, rnn_hidden_states, prev_actions, masks, call_high_level, deterministic, log_info,)

            sel_grouped_skills = self._broadcast_skill_ids(new_skills, sel_dat={}, should_adds=call_high_level,)

            for skill_id, (batch_ids, _) in sel_grouped_skills.items():
                self._skills[skill_id].on_enter(
                    [new_skill_args[i] for i in batch_ids],
                    batch_ids,
                    observations,
                    rnn_hidden_states,
                    prev_actions,
                )
                ## when call high level, initial rnn_hidden_states and prev_actions
                rnn_hidden_states[batch_ids] *= 0.0
                prev_actions[batch_ids] *= 0

                
            if new_skills == self._cur_skills:
                self._skills[skill_id]._cur_skill_step = torch.tensor([self.cur_skills_step])
            else:
                self.cur_skills_step = 0
            self.cur_skills_step += 1
            self._cur_skills = ((~call_high_level) * self._cur_skills) + (call_high_level * new_skills)
            #self._cur_skills = new_skills

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
                # print(f"Calling stop action for batch {batch_idx}, bad_should_terminate{bad_should_terminate}, hl_terminate{hl_terminate}")
                actions[batch_idx, self._stop_action_idx] = 1.0

        action_kwargs = {"rnn_hidden_states": rnn_hidden_states, "actions": actions,}
        action_kwargs.update(hl_info)

        return PolicyActionData(take_actions=actions,policy_info=log_info,should_inserts=None,**action_kwargs,)
    

class Fix_TeacherPolicy(FixHierarchicalPolicy):
   
    def step_action(self, observations, recurrent_hidden_states, prev_actions, not_done_masks, deterministic=False):
        with torch.no_grad():
            action_data = self.act(observations, recurrent_hidden_states, prev_actions,not_done_masks,deterministic=deterministic,)

            if is_continuous_action_space(self.action_space):
                # Clipping actions to the specified limits
                step_data = [np.clip(a.numpy(),self.action_space.low,self.action_space.high,) for a in action_data.env_actions.cpu()]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]    
              
        return step_data

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
        policy_info = torch.zeros(
            (self._num_envs, get_num_actions(self._action_space)*2),
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
        call_high_level = call_high_level | (~masks_cpu).view(-1)
        # End the episode where requested.
        for batch_idx, not_done in enumerate(masks_cpu):
            if not_done == False:
                #print(f"teacher find env {batch_idx} reset")
                self._cur_skills[batch_idx] = -1

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
            self._cur_skills = ((~call_high_level) * self._cur_skills) + (
                call_high_level * new_skills
            )

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
                deterministic=deterministic,
            )

            # LL skills are not allowed to terminate the overall episode.
            actions[batch_ids] = action_data.actions
            policy_info[batch_ids] = torch.cat(action_data.policy_info,dim=1)

            # Add actions from apply_postcond
            rnn_hidden_states[batch_ids] = action_data.rnn_hidden_states

        ## the stop actions dim is useless for MLP nav policy
        actions[:, self._stop_action_idx] = 0.
        policy_info[:,self._stop_action_idx] = 0.
        policy_info[:,11+self._stop_action_idx] = 0.

        should_terminate = bad_should_terminate | hl_terminate
        if should_terminate.sum() > 0:
            # End the episode where requested.
            for batch_idx in torch.nonzero(should_terminate):
                baselines_logger.info(
                    f"Calling stop action for batch {batch_idx}, {bad_should_terminate}, {hl_terminate}"
                )
                actions[batch_idx, self._stop_action_idx] = 1.0
                policy_info[batch_idx, self._stop_action_idx] = 1.0
                
        action_kwargs = {
            "rnn_hidden_states": rnn_hidden_states,
            "actions": actions,
        }
        action_kwargs.update(hl_info)

        return PolicyActionData(
            take_actions=actions,
            policy_info=policy_info,
            should_inserts=call_high_level,
            **action_kwargs,
        )


class LLM_TeacherPolicy(LLMHierarchicalPolicy):
   
    def step_action(self, observations, recurrent_hidden_states, prev_actions, not_done_masks, deterministic=False):
        with torch.no_grad():
            action_data = self.act(observations, recurrent_hidden_states, prev_actions,not_done_masks,deterministic=deterministic,)

            if is_continuous_action_space(self.action_space):
                # Clipping actions to the specified limits
                step_data = [np.clip(a.numpy(),self.action_space.low,self.action_space.high,) for a in action_data.env_actions.cpu()]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]    
              
        return step_data

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
        policy_info = torch.zeros(
            (self._num_envs, get_num_actions(self._action_space)*2),
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
        self._high_level_policy.updata_solution_actions(observations, self._cur_skills, call_high_level, masks)
        # Always call high-level if the episode is over.
        #call_high_level = call_high_level | (~masks_cpu).view(-1)
        call_high_level: torch.BoolTensor = torch.ones(
            (self._num_envs,), dtype=torch.bool
        )  
        # End the episode where requested.
        for batch_idx, not_done in enumerate(masks_cpu):
            if not_done == False:
                #print(f"teacher find env {batch_idx} reset")
                self._cur_skills[batch_idx] = -1

        # If any skills want to terminate invoke the high-level policy to get
        # the next skill.
        hl_terminate = torch.zeros(self._num_envs, dtype=torch.bool)
        hl_info: Dict[str, Any] = self._high_level_policy.create_hl_info()
        if 1:#call_high_level.sum() > 0:
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
                ## add to count current skill step by Hu Bin
                # if new_skills[batch_ids] == self._cur_skills[batch_ids]:
                #     self._skills[skill_id]._cur_skill_step = torch.tensor([self.cur_skills_step])
                # else:
                #     self.cur_skills_step = 0
                # self.cur_skills_step += 1
                ##
            self._cur_skills = call_high_level * new_skills

        

        # print(self._cur_skills,self.cur_skills_step )
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
                deterministic = deterministic,
            )


            # LL skills are not allowed to terminate the overall episode.
            if skill_id != 4.0:
                actions[batch_ids] = action_data.actions
                policy_info[batch_ids] = torch.cat(action_data.policy_info,dim=1)

            # Add actions from apply_postcond
            rnn_hidden_states[batch_ids] = action_data.rnn_hidden_states
        ## the stop actions dim is useless for MLP nav policy
        actions[batch_ids, self._stop_action_idx] = -1.
        policy_info[batch_ids,self._stop_action_idx] = -1.
        policy_info[batch_ids,11+self._stop_action_idx] = 0.


        should_terminate = bad_should_terminate | hl_terminate
        if should_terminate.sum() > 0:
            # End the episode where requested.
            for batch_idx in torch.nonzero(should_terminate):
                baselines_logger.info(
                    f"Calling stop action for batch {batch_idx}, {bad_should_terminate}, {hl_terminate}"
                )
                actions[batch_idx, self._stop_action_idx] = 1.0
                policy_info[batch_idx, self._stop_action_idx] = 1.0
                
        action_kwargs = {
            "rnn_hidden_states": rnn_hidden_states,
            "actions": actions,
        }
        action_kwargs.update(hl_info)

        return PolicyActionData(
            take_actions=actions,
            policy_info=policy_info,
            should_inserts=call_high_level,
            **action_kwargs,
        )

if __name__ == '__main__':
   pass

