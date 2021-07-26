"""
Utilities for training and validation.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import libs.model.FCmodel as FCmodel
import libs.optimizer.optimizer as optimizer
import libs.loss.function as loss_funcs
import libs.visualization.points as vp

from libs.common.transformation import procrustes_transform, pnp_refine
from libs.visualization.debug import save_debug_images
from libs.common.utils import AverageMeter
from libs.metric.criterions import Evaluator

import torch.nn.functional as F
import torch
import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt

def train_cascade(train_dataset, valid_dataset, cfgs, logger):
    # data statistics
    #stats = train_dataset.stats
    stats = None
    # cascaded model
    cascade = FCmodel.get_cascade()
    stage_record = []
    # train each stage
    for stage_id in range(cfgs['cascade']['num_stages']):
        # initialize the model
        input_size, output_size = train_dataset.get_input_output_size()
        cfgs['FCModel']['input_size'] = input_size
        cfgs['FCModel']['output_size'] = output_size
        stage_model = FCmodel.get_fc_model(stage_id + 1, 
                                           cfgs=cfgs,
                                           input_size=input_size,
                                           output_size=output_size
                                           )
        # train_dataset.set_stage(stage_id+1)
        # eval_dataset.set_stage(stage_id+1)
        if cfgs['use_gpu']:
            stage_model = stage_model.cuda()
        # debug
#        train_dataset.stage_update(stage_model, stats, opt)
        # prepare the optimizer
        optim, sche = optimizer.prepare_optim(stage_model, cfgs)
        loss_type = cfgs['FCModel']['loss_type']
        loss_func = eval('loss_funcs.' + loss_type)(
        reduction=cfgs['FCModel']['loss_reduction']
        ).cuda()
        # TODO: a loss that considers geometric consisitency
        # loss_params = {'reduction':cfgs['FCModel']['loss_reduction'],
        #                'use_target_weight':cfgs['training_settings']['use_target_weight'],
        #                'K':train_dataset.intrinsic,
        #                'mean_3d':train_dataset.statistics,
        #                'std_3d':train_dataset.statistics
        #                }
        # loss_func = eval('loss_funcs.' + loss_type)(loss_params).cuda()        
        # train the model
        record = train(train_dataset=train_dataset,
                       valid_dataset=valid_dataset, 
                       model=stage_model, 
                       loss_func=loss_func,
                       optim=optim, 
                       sche=sche, 
                       stats=stats, 
                       cfgs=cfgs,
                       logger=logger
                       )
        stage_model = record['model']
        stage_record.append((record['batch_idx'], record['loss']))
        # update datasets
        # train_dataset.stage_update(stage_model, stats, opt)
        # eval_dataset.stage_update(stage_model, stats, opt)
        # put into cascade
        cascade.append(stage_model.cpu())     
        # release memory
        del stage_model    
    return {'cascade':cascade, 'record':stage_record}

def evaluate_cascade(cascade, 
                     eval_dataset, 
                     stats, 
                     opt, 
                     save=False, 
                     save_path=None,
                     action_wise=False, 
                     action_eval_list=None, 
                     apply_dropout=False
                     ):
    loss, distance = None, None
    for stage_id in range(len(cascade)):
        # initialize the model
        stage_model = cascade[stage_id]
    
        if opt.cuda:
            stage_model = stage_model.cuda()
        
        # evaluate the model
        loss, distance = evaluate(eval_dataset, 
                                  stage_model, 
                                  stats, 
                                  opt, 
                                  save=save, 
                                  save_path=save_path,
                                  procrustes=False, 
                                  per_joint=True, 
                                  apply_dropout=apply_dropout
                                  )

        # update datasets
        eval_dataset.stage_update(stage_model, stats, opt)
        
        # release memory
        del stage_model       
    return loss, distance

def get_loader(dataset, cfgs, split, collate_fn=None):
    setting = split + '_settings'   
    arg_dic = {'batch_size': cfgs[setting]['batch_size'],
               'num_workers': cfgs[setting]['num_threads'],
               'shuffle': cfgs[setting]['shuffle'],
               }
    if collate_fn is not None:
        arg_dic['collate_fn'] = collate_fn
    loader = torch.utils.data.DataLoader(dataset, **arg_dic)     
    return loader

def train(train_dataset,
          model, 
          loss_func,
          optim, 
          sche, 
          cfgs, 
          logger,
          metric_func=None,
          stats=None, 
          valid_dataset=None,
          collate_fn=None,
          save_debug=False
          ):
    """
    Training with optional validation.
    """
    # training configurations
    total_epochs = cfgs['training_settings']['total_epochs']
    batch_size = cfgs['training_settings']['batch_size']
    report_every = cfgs['training_settings']['report_every']
    eval_during = cfgs['training_settings']['eval_during']
    eval_start_epoch = cfgs['training_settings']['eval_start_epoch'] if \
        'eval_start_epoch' in cfgs['training_settings'] else 0
    # evaluate during training
    if eval_during and valid_dataset is not None:
        eval_every = cfgs['training_settings']['eval_every'] 
        evaluator = Evaluator(cfgs['training_settings']['eval_metrics'], 
                              cfgs,
                              train_dataset.num_joints
                              )
    plot_loss = cfgs['training_settings']['plot_loss'] 
    cuda = cfgs['use_gpu'] and torch.cuda.is_available()
    # debug evaluation code
    # evaluate(valid_dataset, 
    #           model, 
    #           loss_func, 
    #           cfgs, 
    #           logger, 
    #           evaluator, 
    #           collate_fn=collate_fn,
    #           epoch=0
    #           )
    # optional list storing loss curve 
    x_buffer = []
    y_buffer = []
    # online plotting
    if plot_loss:
        ax, lines, x_buffer, y_buffer = initialize_plot()
    # training
    for epoch in range(1, total_epochs + 1):
        # Apply cross-ratio loss after certain epochs
        if epoch > 1:
            loss_func.apply_cr_loss = True
        # initialize training record
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()           
        ## DEBUG
        # if epoch % 20 == 0:
        #     logger.info('eval set')
        #     evaluate(eval_dataset, model, cfgs, logger)
        #     logger.info('train set')
        #     evaluate(train_dataset, model, cfgs, logger)
        ## END DEBUG
        model.train()  
        # modify the learning rate according to the scheduler
        sche.step()
        # data loader
        train_loader = get_loader(train_dataset, cfgs, 'training', collate_fn)   
        total_batches = len(train_loader)
        total_sample = len(train_dataset)
        end = time.time()
        for batch_idx, (data, target, weights, meta) in enumerate(train_loader):
            if cuda:
                data, target, weights = data.cuda(), target.cuda(), weights.cuda()
            # measure data loading time
            data_time.update(time.time() - end)
            # erase all computed gradient        
            optim.zero_grad()
            # forward pass to get prediction
            prediction = model(data)
            # compute loss
            loss = loss_func(prediction, target, weights, meta)
            # compute gradient in the computational graph
            loss.backward()
            # update parameters in the model 
            optim.step()
            losses.update(loss.item(), data.size(0))
            # compute other optional metrics besides the loss value
            if metric_func is not None:
                avg_acc, cnt, others = metric_func(prediction, meta, cfgs)
                acc.update(avg_acc, n=cnt, others=others)
                if batch_idx % report_every == 0:
                    acc.print_content()
            else:
                others = None
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()      
            # logging
            if batch_idx % report_every == 0:
                logger_print(epoch, 
                             batch_idx, 
                             batch_size, 
                             total_sample, 
                             batch_time, 
                             data.size()[0], 
                             data_time, 
                             losses,
                             acc, 
                             logger
                             )
                # optional: save intermediate results for debugging
                if save_debug:
                    save_debug_images(epoch, 
                                      batch_idx, 
                                      cfgs, 
                                      data, 
                                      meta, 
                                      target, 
                                      others, 
                                      prediction, 
                                      'train'
                                      )
                # update loss curve
                x_buffer.append(total_batches * (epoch - 1) + batch_idx)
                y_buffer.append(loss.item())
                if plot_loss:
                    update_curve(ax, lines[0], x_buffer, y_buffer)
            del data, target, weights, meta
            # evaluate model if specified
            if eval_during and epoch> eval_start_epoch and \
                batch_idx and batch_idx % eval_every == 0:
                evaluate(valid_dataset, 
                         model, 
                         loss_func, 
                         cfgs, 
                         logger, 
                         evaluator, 
                         collate_fn=collate_fn,
                         epoch=epoch
                         )
                # back to training mode
                model.train()
    logger.info('Training finished.')
    return {'model':model, 'batch_idx':x_buffer, 'loss':y_buffer}  

def initialize_plot():
    x_buffer, y_buffer = [], []
    ax = plt.subplot(111)
    lines = ax.plot(x_buffer, y_buffer)
    plt.xlabel('batch index')
    plt.ylabel('training loss')    
    return ax, lines, x_buffer, y_buffer

def update_curve(ax, line, x_buffer, y_buffer):
    line.set_xdata(x_buffer)
    line.set_ydata(y_buffer)
    # recompute the ax.dataLim
    ax.relim()
    # update ax.viewLim using the new dataLim
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.05)    
    return

def logger_print(epoch, 
                 batch_idx, 
                 batch_size, 
                 total_sample, 
                 batch_time,
                 length,
                 data_time,
                 losses,
                 acc,
                 logger
                 ):
    msg = 'Training Epoch: [{0}][{1}/{2}]\t' \
          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
          'Speed {speed:.1f} samples/s\t' \
          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
          'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
              epoch, 
              batch_idx * batch_size, 
              total_sample, 
              batch_time=batch_time,
              speed=length / batch_time.val,
              data_time=data_time, 
              loss=losses
              )      
    if acc.val != 0 and acc.avg != 0:
        # acc is a pre-defined metric with positive value
        msg += 'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(acc=acc)          
    logger.info(msg)
    return

def visualize_lifting_results(data, 
                              prediction, 
                              target=None, 
                              sample_num=None,
                              intrinsics=None, 
                              refine=False, 
                              dist_coeffs=np.zeros((4,1)),
                              meta_data=None
                              ):
    # only take the coordinates
    if data.shape[1] > 18:
        data = data[:, :18]
    # use the ground truth translation if provided in the meta_data
    if 'roots' in meta_data:
        target = np.hstack([meta_data['roots'], target])
        prediction = np.hstack([meta_data['roots'], prediction])
    sample_num = sample_num if sample_num else len(prediction) 
    chosen = np.random.choice(len(prediction), sample_num, replace=False)
    if target is not None:
        assert len(target) == len(prediction)
        p3d_gt_sample = target[chosen].reshape(sample_num, -1, 3)
    else:
        p3d_gt_sample = None
    p3d_pred_sample = prediction[chosen].reshape(sample_num, -1, 3)
    data_sample = data[chosen].reshape(sample_num, -1, 2)
    # vp.plot_comparison_relative(p3d_pred_sample[:9, 3:], 
    #                             p3d_gt_sample[:9, 3:])
    ax = vp.plot_scene_3dbox(p3d_pred_sample, p3d_gt_sample)
    if not refine:
        return
    # refine 3D point prediction by minimizing re-projection errors
    assert intrinsics is not None
    for idx in range(sample_num):
        prediction = p3d_pred_sample[idx]
        tempt_box_pred = prediction.copy()
        tempt_box_pred[1:, :] += tempt_box_pred[0, :].reshape(1, 3)
        observation = data_sample[idx]
        # use the predicted 3D bounding box size for refinement
        refined_prediction = pnp_refine(tempt_box_pred, observation, intrinsics, 
                                        dist_coeffs)
        vp.plot_lines(ax, 
                      refined_prediction[:, 1:].T, 
                      vp.plot_3d_bbox.connections, 
                      dimension=3, 
                      c='g'
                      )
        # use the gt 3D box size for refinement
        # first align a box with gt size with the predicted box, then refine
        if target is None:
            continue
        tempt_box_gt = p3d_gt_sample[idx].copy()
        tempt_box_gt[1:, :] += tempt_box_gt[0, :].reshape(1, 3) 
        pseudo_box = procrustes_transform(tempt_box_gt.T, tempt_box_pred.T)
        refined_prediction2 = pnp_refine(pseudo_box.T, observation, intrinsics, 
                                        dist_coeffs)
        vp.plot_lines(ax, 
                      pseudo_box[:, 1:].T, 
                      vp.plot_3d_bbox.connections, 
                      dimension=3, 
                      c='y'
                      )         
        vp.plot_lines(ax, 
                      refined_prediction2[:, 1:].T, 
                      vp.plot_3d_bbox.connections, 
                      dimension=3, 
                      c='b'
                      )        
    return

def evaluate(eval_dataset, 
             model, 
             loss_func,
             cfgs, 
             logger, 
             evaluator,
             save=False, 
             save_path=None,
             collate_fn=None,
             epoch=None,
             sample_num=20
             ):
    # unnormalize the prediction if needed
    if cfgs['testing_settings']['unnormalize']:
        stats = eval_dataset.statistics
    # visualize after certain epoch
    if cfgs['exp_type'] == '2dto3d' and 'vis_epoch' in cfgs['testing_settings']:
        vis_epoch = cfgs['testing_settings']['vis_epoch']
    else:
        vis_epoch = -1
    all_dists = []
    model.eval()
    # optional: enable dropout in testing to produce loss similar to the training loss
    if cfgs['testing_settings']['apply_dropout']:
        def apply_dropout(m):
            if type(m) == torch.nn.Dropout:
                m.train()        
        model.apply(apply_dropout)
    intrinsics = None if not hasattr(eval_dataset, 'intrinsic') else \
        eval_dataset.intrinsic
    refine = False if intrinsics is None else True
    eval_loader = get_loader(eval_dataset, cfgs, 'testing', collate_fn)
    cuda = cfgs['use_gpu'] and torch.cuda.is_available()
    losses = AverageMeter()
    # optional: save intermediate results
    if save:
        pred_list = []
        gt_list = []
    has_plot = False # only plot once
    for batch_idx, (data, target, weights, meta) in enumerate(eval_loader):
        if cuda:
            data, target, weights = data.cuda(), target.cuda(), weights.cuda()        
        # forward pass to get prediction
        prediction = model(data)
#        if save:
#            pred_list.append(prediction.data.cpu().numpy())
        loss = loss_func(prediction, target, weights, meta)
        losses.update(loss.item(), data.size(0))
        if cfgs['testing_settings']['unnormalize']:
            # compute distance of body joints in un-normalized format
            target = eval_dataset.unnormalize(target.data.cpu().numpy(), 
                                              stats['mean_out'], 
                                              stats['std_out']
                                              )    
            prediction = eval_dataset.unnormalize(prediction.data.cpu().numpy(), 
                                                  stats['mean_out'], 
                                                  stats['std_out']
                                                  ) 
        evaluator.update(prediction, ground_truth=target, meta_data=meta)
        ## plot 3D bounding boxes for visualization
        if not has_plot and vis_epoch > 0 and epoch > vis_epoch:
            data_unnorm = eval_dataset.unnormalize(data.data.cpu().numpy(), 
                                                   stats['mean_in'], 
                                                   stats['std_in']
                                                   ) 
            visualize_lifting_results(data_unnorm, 
                                      prediction, 
                                      target, 
                                      sample_num=sample_num, 
                                      intrinsics=intrinsics,
                                      refine=refine,
                                      meta_data=meta
                                      )
            has_plot = True
        if save:
            pred_list.append(prediction)
            gt_list.append(target)
    if save:
        # note the residual update is saved if a cascade is used
        record = {#'data':np.concatenate(data_list, axis=0), 
                  'pred':np.concatenate(pred_list, axis=0), 
                  'error':all_dists, 
                  'gt':np.concatenate(gt_list, axis=0)
                  }
        np.save(save_path, np.array(record))
    evaluator.report(logger)
    return