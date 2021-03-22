#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset normalization operations.
Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import numpy as np

def get_statistics_1d(data):
    # data of shape [num_sample, vector_length]
    assert len(data.shape) == 2
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    return mean, std

def normalize_1d(data, mean, std, individual=False):
    """
    Normalizes a dictionary of poses
    Args
      data: dictionary where values are
      mean: np vector with the mean of the data
      std: np vector with the standard deviation of the data
      individual: whether to perform normalization independently for each input
    Returns
      data_out: normalized data
    """
    if individual:
        # this representation has the implicit assumption that the representation
        # is translational and scaling invariant
        # Reference: 
        # for data organized as [x1, y1, x2, y2, ...]
        num_data = len(data)
        data = data.reshape(num_data, -1, 2)
        mean_x = np.mean(data[:,:,0], axis=1).reshape(num_data, 1)
        std_x = np.std(data[:,:,0], axis=1)
        mean_y = np.mean(data[:,:,1], axis=1).reshape(num_data, 1)
        std_y = np.std(data[:,:,1], axis=1)
        denominator = (0.5 * (std_x + std_y)).reshape(num_data, 1)
        data[:,:,0] = (data[:,:,0] - mean_x)/denominator
        data[:,:,1] = (data[:,:,1] - mean_y)/denominator
        data_out = data.reshape(num_data, -1)
    else:
        data_out = (data - mean)/std
    return data_out

def unnormalize_1d(normalized_data, mean, std):
    orig_data = normalized_data*std + mean
    return orig_data
