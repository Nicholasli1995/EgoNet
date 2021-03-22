#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deprecated. Will be deleted in a future version.
Pre-defined accuracy functions.
Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import libs.common.img_proc as lip

import numpy as np

def get_distance(gt, pred):
    # gt: [n_joints, 2 or 3]
    # pred: [n_joints, 2]
    if gt.shape[1] == 2:
        sqerr = (gt - pred)**2
        sqerr = sqerr.sum(axis = 1)
        dist_list = list(np.sqrt(sqerr))
    elif gt.shape[1] == 3:
        dist_list = []
        sqerr = (gt[:, :2] - pred)**2
        sqerr = sqerr.sum(axis = 1)
        indices = np.nonzero(gt[:, 2])[0]
        dist_list = list(np.sqrt(sqerr[indices]))        
    else:
        raise ValueError('Array shape not supported.')
    return dist_list

def accuracy_pixel(output, 
                   meta_data, 
                   cfgs=None,
                   image_size = (256.0, 256.0), 
                   arg_max='hard'
                   ):
    """
    pixel-wise distance computed from predicted heatmaps
    """
    # report distance in terms of pixel in the original image
    if arg_max == 'soft':
        if isinstance(output, np.ndarray):
            pred, max_vals = lip.get_max_preds_soft(output)
        else:
            pred, max_vals = lip.get_max_preds_soft_pt(output)
    elif arg_max == 'hard':
        if not isinstance(output, np.ndarray):
            output = output.data.cpu().numpy()
        pred, max_vals = lip.get_max_preds(output)
    else:
        raise NotImplementedError
    image_size = image_size if cfgs is None else cfgs['heatmapModel']['input_size']
    # TODO: check the target generation and coordinate mapping
    # multiply by down-sample ratio
    if not isinstance(pred, np.ndarray):
        pred = pred.data.cpu().numpy()
        max_vals = max_vals.data.cpu().numpy()
    pred *= image_size[0]/output.shape[3]
    # inverse transform and compare pixel didstance
    centers, scales, rots = meta_data['center'], meta_data['scale'], meta_data['rotation']
    centers = centers.data.cpu().numpy()
    scales = scales.data.cpu().numpy()
    rots = rots.data.cpu().numpy()
    joints_original_batch = meta_data['original_joints'].data.cpu().numpy()
    distance_list = []
    all_src_coordinates = []
    for sample_idx in range(len(pred)):
        trans_inv = lip.get_affine_transform(centers[sample_idx], 
                                             scales[sample_idx], 
                                             rots[sample_idx], 
                                             image_size, 
                                             inv=1)
        joints_original = joints_original_batch[sample_idx]        
        pred_src_coordinates = lip.affine_transform_modified(pred[sample_idx], 
                                                             trans_inv) 
        all_src_coordinates.append(pred_src_coordinates.reshape(1, len(pred_src_coordinates), 2))
        distance_list += get_distance(joints_original, pred_src_coordinates)
    cnt = len(distance_list)
    avg_acc = sum(distance_list)/cnt
    others = {
        'src_coord': np.concatenate(all_src_coordinates, axis=0),
        'joints_pred': pred,
        'max_vals': max_vals
        }
    return avg_acc, cnt, others