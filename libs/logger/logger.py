"""
Basic logging functions.
Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import logging
import os
import time
from libs.common import utils

initialized = False

def get_dirs(cfgs):
    root_output_dir = cfgs['dirs']['output']
    dataset_name = cfgs['dataset']['name']
    model_type = cfgs['model_type']
    cfg_name = cfgs['name']
    final_output_dir = [root_output_dir, dataset_name, model_type]    
    final_output_dir = os.path.join(*final_output_dir)
    time_str = time.strftime('%Y-%m-%d %H:%M')
    log_file = '{}_{}.log'.format(cfg_name, time_str)
    final_log_file = os.path.join(final_output_dir, log_file)
    return final_output_dir, final_log_file

def get_logger(cfgs, head = '%(asctime)-15s %(message)s'):
    final_output_dir, final_log_file = get_dirs(cfgs)
    utils.make_dir(final_log_file)
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if len(logger.handlers) == 1:    
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)    
    return logger, final_output_dir
