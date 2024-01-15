#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mediator.py
@Time    :   2023/08/01 16:30:32
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''


ROBOT_SKILL_TO_IDX = {
   "pick": 0,
   "place": 1,
   "wait": 2,               ## for end robot
   "nav": 3,
   "nav_to_receptacle": 4,  ## same as nav, maybe useless
   "open_fridge": 5,
   "close_fridge": 6,
   "open_cab": 7,
   "close_cab": 8,
   "initialization": -1,
}
IDX_TO_ROBOT_SKILL = dict(zip(ROBOT_SKILL_TO_IDX.values(), ROBOT_SKILL_TO_IDX.keys()))

from abc import ABC, abstractmethod
import re
import torch
class Base_Mediator(ABC):
   """The base class for Base_Mediator."""
   def __init__(self):
      super().__init__()
      self.obj_coordinate = {}

   @abstractmethod
   def RL2LLM(self):
      pass
   @abstractmethod
   def LLM2RL(self):
      pass
    
   def reset(self):
      self.obj_coordinate = {}


class Habitat_Mediator(Base_Mediator):
   def __init__(self):
      super().__init__()
      self.add_arm_rest = False
      self.context = ''
   def append_context(self, str):
      if self.context != "":
         self.context += ", "
      self.context += f"{str}" 
    # ## obs2text
   def RL2LLM(self, obs, cur_skill, skill_done):
         self.context = ''
         # if obs['obj_goal_gps_compass'][0] < 0.5 and obs['is_holding'][0] == 0:
         #    self.append_context('stop')
         # else:
            # if cur_skill == 2 and obs['is_holding'][0] == 1:
            #    self.append_context('targeted object')
         if cur_skill == -1:
            self.append_context('untargeted object')
         elif cur_skill == 2 and obs['obj_start_gps_compass'][0] <= 1.2:
            self.append_context('targeted object')
         elif cur_skill == 2 and obs['obj_start_gps_compass'][0] > 1.2:
            self.append_context('untargeted object')
         elif cur_skill == 0 and obs['is_holding'][0] == 0:
            self.append_context('targeted object')
         elif cur_skill == 0 and obs['is_holding'][0] == 1 and torch.norm(obs['relative_resting_position'])>0.15:
            self.append_context('targeted object')
         elif cur_skill == 0 and obs['is_holding'][0] == 1 and torch.norm(obs['relative_resting_position'])<=0.15:
            self.append_context('untargeted goal')
         elif cur_skill == 3 and obs['obj_goal_gps_compass'][0] > 1.5:
            self.append_context('untargeted goal')
         elif cur_skill == 3 and obs['obj_goal_gps_compass'][0] <= 1.5:
            self.append_context('targeted goal') 
         # elif cur_skill == 1 and obs['is_holding'][0] == 0 and torch.norm(obs['relative_resting_position']) < 0.2:
         #    self.append_context('stop') 
         elif cur_skill == 1:
            self.append_context('targeted goal')
            # elif cur_skill == 1 and skill_done == True:
            #    self.append_context('targeted goal')

               
               #print("new env")
            # if obs['is_holding'][0] == 0:
            #    if cur_skill == 0:  ## pick now, don't back to nav
            #       self.append_context('targeted object')
            #    elif cur_skill == 1 and hl_wants_skill_term == True : ## place now, stop
            #       self.append_context('stop') 
            #    # elif obs['obj_start_gps_compass'][0] < 1.5 and -0.5 < obs['obj_start_gps_compass'][1] < 0.5:
            #    #    self.append_context('targeted object')
            #    # elif cur_skill == 2 and obs['obj_start_gps_compass'][0] < 1.5:
            #    #    self.append_context('targeted object')
            #    elif cur_skill == 2 and hl_wants_skill_term == True:
            #       self.append_context('targeted object')
            #    else:
            #       self.append_context('untargeted object')
            # elif obs['is_holding'][0] == 1:
            #    if cur_skill == 1:  ## place now, don't back to nav
            #       self.append_context('targeted goal')
            #    # elif obs['obj_goal_gps_compass'][0] < 1.5 and -0.5 < obs['obj_goal_gps_compass'][1] < 0.5:
            #    #    self.append_context('targeted goal') 
            #    # elif cur_skill == 3 and obs['obj_goal_gps_compass'][0] < 1.5:
            #    #    self.append_context('targeted goal') 
            #    elif cur_skill == 3 and hl_wants_skill_term == True:
            #       self.append_context('targeted goal') 
            #    elif cur_skill == 0 and hl_wants_skill_term == True:
            #       self.append_context('untargeted goal')
            #    # else:
            #    #    self.append_context('untargeted goal')
         context = self.context
         return context
    
   def parse_func(self, x: str):
      """
      Parses out the components of a function string.
      :returns: First element is the name of the function, second argument are the function arguments.
      """
      try:
         name = x.split(" ")[0]
         args_list = x.split(" ")[1:]
         if name.lower() in ROBOT_SKILL_TO_IDX.keys():
            args_list.append('robot_0')
      except IndexError as e:
         raise ValueError(f"Cannot parse '{x}'") from e

      if len(args_list) == 1 and args_list[0] == "":
         args_list = []

      return name, args_list    
      
   def LLM2RL(self, plan):
         if plan == "{wait}":
            skill_list = self.parse_func("wait 5")
            return [skill_list]      
         plan = re.findall(r'{(.*?)}', plan)
         lines = plan[0].split(',')
         skill_list = []
         #for line in lines:

         for i, line in enumerate(lines):
            action, object = self.parse_func(line.strip())
            solution_actions = (action, object)
            skill_list.append(solution_actions)
            if self.add_arm_rest and i < len(lines)-1:
               skill_list.append(self.parse_func("reset_arm 0"))

         return skill_list
if __name__ == '__main__':
   pass

   # plan = "{nav goal, pick goal, nav target_goal, place goal target_goal}"
   # test = Habitat_Mediator()
   # skill = test.LLM2RL(plan)
   # print(skill)
