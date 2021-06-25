"""
Argument parser for command line inputs and experiment configuration file.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import yaml
import argparse

def read_yaml_file(path):
    """
    Read a .yml file.
    """
    try: 
        with open (path, 'r') as file:
            configs = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file: ', e)
    return configs

def parse_args():
    """
    Read a .yml experiment configuration file whose path is provided by the user.
    
    You can add more arguments and modify configs accordingly.
    """
    parser = argparse.ArgumentParser(description='a general parser')
    # path to the configuration file
    parser.add_argument('--cfg',
                        help='experiment configuration file path',
                        type=str
                        )
    parser.add_argument('--visualize',
                        default=False,
                        type=bool
                        )    
    parser.add_argument('--batch_to_show',
                        default=1000000,
                        type=int
                        )    
    args, unknown = parser.parse_known_args()
    configs = read_yaml_file(args.cfg)   
    configs['config_path'] = args.cfg
    configs['visualize'] = args.visualize
    configs['batch_to_show'] = args.batch_to_show
    return configs
