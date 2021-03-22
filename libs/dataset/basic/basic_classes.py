#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic classes for customized dataset classes to inherit.
Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""
import torch.utils.data
import libs.dataset.normalization.operations as nop

class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, cfgs, split, logger=None):
        self.cfgs = cfgs
        self.split = split
        self.logger = logger
        self.root = cfgs['dataset']['root']
        return
    
    def generate_pairs(self, synthetic=True):
        # sub-classes need to override this method to specify the inputs and
        # outputs
        self.input = None
        self.output = None
        self.total_data = 0
        return
    
    def normalize(self, statistics=None):
        # normalize the input-output pairs with optional given statistics
        if statistics is None:
            mean_in, std_in = nop.get_statistics_1d(self.input)
            mean_out, std_out = nop.get_statistics_1d(self.output)
            self.statistics = {'mean_in': mean_in,
                               'mean_out': mean_out,
                               'std_in': std_in,
                               'std_out': std_out
                               }
        else:
            mean_in, std_in = statistics['mean_in'], statistics['std_in']
            mean_out, std_out = statistics['mean_out'], statistics['std_out']
            self.statistics = statistics
        self.input = nop.normalize_1d(self.input, mean_in, std_in)
        self.output = nop.normalize_1d(self.output, mean_out, std_out)
        return
    
    def unnormalize(self, data, mean, std):
        return nop.unnormalize_1d(data, mean, std)
    
    def __len__(self):
        return self.total_data

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]
