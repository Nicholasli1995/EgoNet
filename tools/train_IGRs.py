"""
Training the coordinate localization sub-network.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import sys
sys.path.append('../')
import torch
import os

import libs.arguments.parse as parse
import libs.logger.logger as liblogger
import libs.dataset as dataset
# import libs.dataset.ApolloScape.car_instance
import libs.dataset.KITTI.car_instance
import libs.trainer.trainer as trainer
import libs.model as models
import libs.optimizer.optimizer as optimizer
import libs.loss.function as loss_func

from libs.common.utils import get_model_summary
from libs.metric.criterions import get_distance_src, get_angle_error
from libs.metric.criterions import Evaluator

def choose_loss_func(model_settings, cfgs):
    """
    Initialize the loss function used for training. 
    """
    loss_type = model_settings['loss_type']
    if loss_type == 'JointsCompositeLoss':
        spec_list = model_settings['loss_spec_list']
        loss_weights = model_settings['loss_weight_list']
        func = loss_func.JointsCompositeLoss(spec_list=spec_list,
                                             img_size=model_settings['input_size'],
                                             hm_size=model_settings['heatmap_size'],
                                             cr_loss_thres=model_settings['cr_loss_threshold'],
                                             loss_weights=loss_weights
                                             )
    else:
        func = eval('loss_func.' + loss_type)(use_target_weight=cfgs['training_settings']['use_target_weight'])    
    # the order of the points are needed when computing the cross-ratio loss
    if model_settings['loss_spec_list'][2] != 'None':
        func.cr_indices = libs.dataset.KITTI.car_instance.cr_indices_dict['bbox12']
        func.target_cr = 4/3
    return func.cuda()

def train(model, model_settings, GPUs, cfgs, logger, final_output_dir):
    """
    The training method.
    """
    # get model summary
    input_size = model_settings['input_size']
    input_channels = 5 if cfgs['heatmapModel']['add_xy'] else 3
    dump_input = torch.rand((1, input_channels, input_size[1], input_size[0]))
    logger.info(get_model_summary(model, dump_input))
    
    model = torch.nn.DataParallel(model, device_ids=GPUs).cuda()

    # get forward-pass time if you need 
    # import time
    # dump_input = torch.rand((64, input_channels, input_size[1], input_size[0])).cuda()
    # t1 = time.clock()
    # out = model(dump_input)
    # l = out[0].sum()
    # l.backward()
    # torch.cuda.synchronize()
    # print(time.clock() - t1)

    # specify loss function 
    func = choose_loss_func(model_settings, cfgs)
    
    # dataset preparation
    data_cfgs = cfgs['dataset']
    train_dataset, valid_dataset = eval('dataset.' + data_cfgs['name'] + 
                                        '.car_instance').prepare_data(cfgs, logger)
    # get the optimizer and learning rate scheduler    
    optim, sche = optimizer.prepare_optim(model, cfgs)

    # metrics used for training error
    if cfgs['exp_type'] in ['baselinealpha', 'baselinetheta']:
        metric_function = get_angle_error
        save_debug_images = False
    elif cfgs['exp_type'] == 'instanceto2d':
        metric_function = get_distance_src
        save_debug_images = cfgs['training_settings']['debug']['save']
    collate_fn = train_dataset.get_collate_fn()
    trainer.train(train_dataset=train_dataset, 
                  valid_dataset=valid_dataset,
                  model=model,                   
                  loss_func=func,
                  optim=optim, 
                  sche=sche, 
                  metric_func=metric_function,
                  cfgs=cfgs, 
                  logger=logger,
                  collate_fn=collate_fn,
                  save_debug=save_debug_images
                  )

    final_model_state_file = os.path.join(final_output_dir, 'HC.pth')
    logger.info('=> saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.cpu().state_dict(), final_model_state_file)
    return

def evaluate(model, model_settings, GPUs, cfgs, logger, final_output_dir, eval_train=False):
    saved_path = cfgs['dirs']['load_hm_model']
    model.load_state_dict(torch.load(saved_path))
    model = torch.nn.DataParallel(model, device_ids=GPUs).cuda()
    evaluator = Evaluator(cfgs['testing_settings']['eval_metrics'], cfgs)
    # define loss function (criterion) and optimizer
    loss_func = choose_loss_func(model_settings, cfgs)
    # dataset preparation
    data_cfgs = cfgs['dataset']
    train_dataset, valid_dataset = eval('dataset.' + data_cfgs['name'] + 
                                        '.car_instance').prepare_data(cfgs, logger)
    collate_fn = valid_dataset.get_collate_fn()
    logger.info("Evaluation on the validation split:")
    trainer.evaluate(valid_dataset, model, loss_func, cfgs, logger, evaluator, collate_fn=collate_fn)    
    if eval_train:
        logger.info("Evaluation on the training split:")
        trainer.evaluate(train_dataset, model, loss_func, cfgs, logger, evaluator)
    return

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
    
    if len(GPUs) == 1:
        torch.cuda.set_device(GPUs[0])
        
    # cudnn related setting
    torch.backends.cudnn.benchmark = cfgs['cudnn']['benchmark']
    torch.backends.cudnn.deterministic = cfgs['cudnn']['deterministic']
    torch.backends.cudnn.enabled = cfgs['cudnn']['enabled']
    
    # model initialization
    model_settings = cfgs['heatmapModel']
    model_name = model_settings['name']
    method_str = 'models.heatmapModel' + '.' + model_name + '.get_pose_net'
    model = eval(method_str)(cfgs, is_train=cfgs['train'])

    if cfgs['train']:
        train(model, model_settings, GPUs, cfgs, logger, final_output_dir)
    elif cfgs['evaluate']:
        evaluate(model, model_settings, GPUs, cfgs, logger, final_output_dir)
    
if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()