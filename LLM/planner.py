#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   planner.py
@Time    :   2023/08/01 16:30:11
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import os, requests
from typing import Any
from .mediator import *


from abc import ABC, abstractmethod

class Base_Planner(ABC):
    """The base class for Planner."""

    def __init__(self):
        super().__init__()
        self.dialogue_system = ''                  
        self.dialogue_user = ''
        self.dialogue_logger = ''         
        self.show_dialogue = False
        self.llm_model = None
        self.llm_url = None
    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show

    ## initial prompt, write in 'task_info.json
    def initial_planning(self, decription, example):
        if self.llm_model is None:
            assert "no select Large Language Model"
        prompts = decription + example
        self.dialogue_system += decription + "\n"
        self.dialogue_system += example + "\n"

        ## set system part
        server_error_cnt = 0
        while server_error_cnt<10:
            try:
                url = self.llm_url
                headers = {'Content-Type': 'application/json'}
                
                data = {'model': self.llm_model, "messages":[{"role": "system", "content": prompts}]}
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    result = response.json()                    
                    server_flag = 1
                                
                   
                if server_flag:
                    break
                    
            except Exception as e:
                server_error_cnt += 1
                print(e)    

    def query_codex(self, prompt_text):
        server_flag = 0
        server_error_cnt = 0
        response = ''
        while server_error_cnt<10:
            try:
                url = self.llm_url
                headers = {'Content-Type': 'application/json'}
                # prompt_text
                
                data = {'model': self.llm_model, "messages":[{"role": "user", "content": prompt_text }]}
                response = requests.post(url, headers=headers, json=data)

                if response.status_code == 200:
                    result = response.json()                    
                    server_flag = 1
                                
                   
                if server_flag:
                    break
                    
            except Exception as e:
                server_error_cnt += 1
                print(e)
        if result is None:
            return
        else:
            return result['messages'][-1][-1] 

    def check_plan_isValid(self, plan):
        if "{" in plan and "}" in plan:
            return True
        else:
            return False
        
    def step_planning(self, text):
        ## seed for LLM and get feedback
        plan = self.query_codex(text)
        if plan is not None:
            ## check Valid, llm may give wrong answer
            while not self.check_plan_isValid(plan):
                print("%s is illegal Plan! Replan ...\n" %plan)
                plan = self.query_codex(text)
        return plan

    @abstractmethod
    def forward(self):
        pass


class Pick_Place_Planner(Base_Planner):
    def __init__(self, seed=0):
        super().__init__()
        
        self.mediator = Habitat_Mediator()

        self.llm_model = "vicuna-7b-0"
        self.llm_url = 'http://10.109.116.3:8000/v1/chat/completions'

    def __call__(self, *args):
        return self.forward(*args)
    
    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show
        ## reset dialogue
        if self.show_dialogue:
            print(self.dialogue_system)

       
    def forward(self, obs, cur_skill, hl_wants_skill_term, masks):
        env_skill = []
        for i in range(cur_skill.shape[0]):
            if not masks[i]:
                cur_skill[i] = -1 ## reset
            text = self.mediator.RL2LLM(obs[i], cur_skill[i], hl_wants_skill_term[i])
            # print(text)
            plan = self.ideal_planning(text)
            #plan = "{nav goal0|0, pick goal0|0, nav_to_receptacle TARGET_goal0|0, place goal0|0 TARGET_goal0|0}"
        
            # self.dialogue_logger += text
            # self.dialogue_logger += plan
            # self.dialogue_user = "Robot: " + text
            # #self.dialogue_user += "LLM: " + plan
            # if self.show_dialogue:
            #     print(self.dialogue_user)
            skill = self.mediator.LLM2RL(plan)
            env_skill.append(skill)
        return env_skill
    
    def ideal_planning(self, text):
        if text == "untargeted object":
            # plan = "{nav goal0|0}"
            plan = "{nav goal0|0, pick goal0|0, nav_to_receptacle TARGET_goal0|0, place goal0|0 TARGET_goal0|0}"
        elif text == "targeted object":
            # plan = "{pick goal0|0}"
            plan = "{pick goal0|0, nav_to_receptacle TARGET_goal0|0, place goal0|0 TARGET_goal0|0}"
        elif text == "untargeted goal": 
            # plan = "{nav_to_receptacle TARGET_goal0|0}"
            plan = "{nav_to_receptacle TARGET_goal0|0, place goal0|0 TARGET_goal0|0}"
        elif text == "targeted goal":
            plan = "{place goal0|0 TARGET_goal0|0}"
        elif text == "mission completed":
            plan = "{wait}"
        elif text == "stop":
            plan = "{wait}"
        return plan
def Planner(task,seed=0):
    if task.lower() == "pick_place":
        planner = Pick_Place_Planner(seed)
    else:
        assert "unknown task"
    return planner

if __name__ == '__main__':
   pass

