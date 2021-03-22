"""
Argument parser for command line inputs and experiment configuration file.
Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import yaml
import argparse

def read_yaml_file(path):
    try: 
        with open (path, 'r') as file:
            configs = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file: ', e)
    return configs

def parse_args():
    """
    Read an .yml experiment configuration file whose path is provided by the user.
    """
    parser = argparse.ArgumentParser(description='a general parser')
    # path to the configuration file
    parser.add_argument('cfg',
                        help='experiment configuration file path',
                        type=str
                        )
    args, unknown = parser.parse_known_args()
    configs = read_yaml_file(args.cfg)         
    return configs
