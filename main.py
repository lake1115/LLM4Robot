#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/08/21 10:11:17
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
import argparse
import os,json, sys
import numpy as np
import torch
import random
from habitat_env import Env
# single gpu    
os.system('nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > tmp.txt')
memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu)) 
os.system('rm tmp.txt')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

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
    parser.add_argument("--n_itr", type=int, default=100, help="Number of iterations of the learning algorithm")
    parser.add_argument("--traj_per_itr", type=int, default=10)
    parser.add_argument("--save_name", type=str, required=True, help="path to folder containing policy and run details")
    parser.add_argument("--ckpt_num", type=int, default=10, help="Number of save model")
    parser.add_argument("--logdir", type=str, default="./log/")          # Where to log diagnostics to
    parser.add_argument("--record_video", default=False, action='store_true')
    parser.add_argument("--test_num", type=int, default=10)
    parser.add_argument("--eval_type", type=str, default="LLM", help="LLM, fix, always, random")


    if sys.argv[1] == 'train':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()    
        env = Env(args)
        env.train()
    elif sys.argv[1] == 'eval':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()

        print("env name: %s for %s" %(args.task, args.save_name))
        if args.eval_type.lower() == "llm":
            output_dir = os.path.join(args.logdir, args.policy, args.task, args.save_name)
            policy = torch.load(output_dir + "/latest.pt")
            policy.eval()
            env = Env(args,policy)
        else:
            env = Env(args)

        eval_rewards = []
        eval_comm_rewards = []
        eval_lens = []
        eval_success = []
        eval_interactions = []

        for i in range(args.test_num):
            ## eval_policy ##
            if env.seed == None:
                seed = i
                setup_seed(seed)          
            else:
                setup_seed(args.seed)

            if args.eval_type.lower()  == "llm" or args.eval_type.lower()  == "random" or args.eval_type.lower() == "always":
                reward, comm_reward, len, success, call_num = env.eval(trajs=1, seed=seed)
            elif args.eval_type.lower() == "fix":
                reward, comm_reward, len, success, call_num = env.eval_fix(trajs=1, seed=seed)    
            else:
                print("Invalid option '{}'".format(args.eval_type))
            success_rate = (success[0]['stage_0_5_success'], success[0]['stage_1_success'], success[1])
            eval_rewards.append(reward)
            eval_comm_rewards.append(comm_reward)
            eval_lens.append(len)
            eval_success.append(success_rate)
            eval_interactions.append(call_num)

            print("task %s, reward %s, len %s, interaction %s, reward with comm penalty %s" %(i, reward,len, call_num, comm_reward))


        print("Mean reward:", np.mean(eval_rewards))
        print("Mean reward with comm penalty:", np.mean(eval_comm_rewards))
        print("Mean len:", np.mean(eval_lens))
        print("Mean interactions:", np.mean(eval_interactions))
        print("Success rate:", np.mean(eval_success,0))


    else:
        print("Invalid option '{}'".format(sys.argv[1]))
