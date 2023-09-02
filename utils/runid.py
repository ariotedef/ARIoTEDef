"""
Author: maggie
Date:   2023-05-15
Place:  Purdue University
@copyright
"""

import os
import re

def GetRunID(exp_result_dir):
    prev_run_dirs = []
    if os.path.isdir(exp_result_dir):
        prev_run_dirs = [x for x in os.listdir(exp_result_dir) if os.path.isdir(os.path.join(exp_result_dir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1

    return cur_run_id