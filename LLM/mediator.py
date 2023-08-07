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
   "nav": 0,
   "pick": 1,
   "place": 2,
   "open_fridge": 3,
   "close_fridge": 4,
   "open_cab": 5,
   "close_cab": 6,
}
IDX_TO_ROBOT_SKILL = dict(zip(ROBOT_SKILL_TO_IDX.values(), ROBOT_SKILL_TO_IDX.keys()))

from abc import ABC, abstractmethod
import re
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
      self.add_arm_rest = True
      self.context = ''
   def append_context(self, str):
      if self.context != "":
         self.context += ", "
      self.context += f"{str}" 
    # ## obs2text
   def RL2LLM(self, obs):
         self.context = ''
         if obs['is_holding'][:,0] == 0:
            if obs['obj_start_gps_compass'][:,0] < 1.5 and -0.5 < obs['obj_start_gps_compass'][:,1] < 0.5:
               self.append_context('targeted object')
            else:
               self.append_context('untargeted object')
         elif obs['is_holding'][:,0] == 1:
            if obs['obj_goal_gps_compass'][:,0] < 1.5 and -0.5 < obs['obj_goal_gps_compass'][:,1] < 0.5:
               self.append_context('targeted goal')
            else:
               self.append_context('untargeted goal')
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
      
         plan = re.findall(r'{(.*?)}', plan)
         lines = plan[0].split(',')
         skill_list = []
         #for line in lines:
         for i, line in enumerate(lines):
            action, object = self.parse_func(line.strip())
            solution_actions = (action, object)
            skill_list.append(solution_actions)
            if self.add_arm_rest:
               skill_list.append(self.parse_func("reset_arm 0"))
         return skill_list
if __name__ == '__main__':
   pass

   # plan = "{nav goal, pick goal, nav target_goal, place goal target_goal}"
   # test = Habitat_Mediator()
   # skill = test.LLM2RL(plan)
   # print(skill)
