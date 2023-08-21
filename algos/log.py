#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   log.py
@Time    :   2023/08/09 16:59:47
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
from collections import OrderedDict
import os, pickle
import json

class color:
 BOLD   = '\033[1m\033[48m'
 END    = '\033[0m'
 ORANGE = '\033[38;5;202m'
 BLACK  = '\033[38;5;240m'

 
def create_logger(args):
    from torch.utils.tensorboard import SummaryWriter
    """Use hyperparms to set a directory to output diagnostic files."""

    arg_dict = args.__dict__

    assert "logdir" in arg_dict, \
      "You must provide a 'logdir' key in your command line arguments."
  
    arg_dict = OrderedDict(sorted(arg_dict.items(), key=lambda t: t[0]))
    logdir = str(arg_dict.pop('logdir'))
    output_dir = os.path.join(logdir, args.policy, args.task, args.save_name)

    os.makedirs(output_dir, exist_ok=True)

    # Create a file with all the hyperparam settings in plaintext
    info_path = os.path.join(output_dir, "config.json")

    with open(info_path,'wt') as f:
        json.dump(arg_dict, f, indent=4)

    logger = SummaryWriter(output_dir, flush_secs=0.1)
    print("Logging to " + color.BOLD + color.ORANGE + str(output_dir) + color.END)

    logger.name = args.task
    logger.dir = output_dir
    return logger

if __name__ == '__main__':
   pass

