"""
Training the sub-network \mathcal{L}() that predicts 3D cuboid 
given 2D screen coordinates as input.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import sys
sys.path.append('../')

import libs.arguments.parse as parse
import libs.logger.logger as liblogger
# Deprecated: Apolloscape dataset
#import libs.dataset.ApolloScape.car_instance as car_instance
# KITTI dataset
import libs.dataset.KITTI.car_instance as car_instance
import libs.trainer.trainer as trainer

import torch
import numpy as np
import os

def main():
    # experiment configurations
    cfgs = parse.parse_args()

    # logging
    logger, final_output_dir = liblogger.get_logger(cfgs)
    
    # Set GPU
    if cfgs['use_gpu'] and torch.cuda.is_available():
        GPUs = cfgs['gpu_id']
    else:
        logger.info("GPU acceleration is disabled.")

    # load datasets
    train_dataset, eval_dataset = car_instance.prepare_data(cfgs, logger)
    logger.info("Finished preparing datasets...")
    
    # training
    if cfgs['train']:
        record = trainer.train_cascade(train_dataset, eval_dataset, cfgs, logger)
        cascade = record['cascade']

    if cfgs['save'] and 'cascade' in locals():
        save_path = os.path.join(cfgs['dirs']['output'], "KITTI")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # save the model and the normalization statistics
        torch.save(cascade[0].cpu().state_dict(), 
                   os.path.join(save_path, 'L.pth')
                   )
        np.save(os.path.join(save_path, 'LS.npy'), train_dataset.statistics)
        logger.info('=> saving final model state to {}'.format(save_path))        
        # save loss history
        #np.save(os.path.join(save_path, 'record.npy'), record['record'])
        
    if cfgs['visualize'] or cfgs['evaluate']:
        # visualize the predictions
        cascade = torch.load(cfgs['load_model_path'])        
        if cfgs['use_gpu']:
            cascade.cuda()
            
    if cfgs['evaluate']:   
        trainer.evaluate_cascade(cascade, eval_dataset, cfgs) 
        
    return record

if __name__ == "__main__":
    record = main()
    torch.cuda.empty_cache()