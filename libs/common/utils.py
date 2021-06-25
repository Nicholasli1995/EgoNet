"""
Common utilities.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import torch
import torch.nn as nn
import numpy as np

from libs.metric.criterions import PCK_THRES

import os
from os.path import join as pjoin
from collections import namedtuple

def make_dir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            print('make_dir failed.')
            raise exc
    return

def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    torch.save(states, pjoin(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'], pjoin(output_dir, 'model_best.pth'))


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    summarize a model. For now only convolution, batch normalization and 
    linear layers are considered for parameters and FLOPs.
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """
    summary = []
    ModuleDetails = namedtuple(
        "Layer", 
        ["name", "input_size", "output_size", "num_parameters", "multiply_adds"]
        )
    hooks = []
    layer_instances = {}

    def hook(module, input, output):
        class_name = str(module.__class__.__name__)
        instance_index = 1
        if class_name not in layer_instances:
            layer_instances[class_name] = instance_index
        else:
            instance_index = layer_instances[class_name] + 1
            layer_instances[class_name] = instance_index
    
        layer_name = class_name + "_" + str(instance_index)
    
        params = 0
    
        if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
           class_name.find("Linear") != -1:
            for param_ in module.parameters():
                params += param_.view(-1).size(0)
    
        flops = "Not Available"
        if class_name.find("Conv") != -1 and hasattr(module, "weight"):
            flops = (
                torch.prod(
                    torch.LongTensor(list(module.weight.data.size()))) *
                torch.prod(
                    torch.LongTensor(list(output.size())[2:]))).item()
        elif isinstance(module, nn.Linear):
            flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                     * input[0].size(1)).item()
    
        if isinstance(input[0], list):
            input = input[0]
        if isinstance(output, list):
            output = output[0]
    
        summary.append(
            ModuleDetails(
                name=layer_name,
                input_size=list(input[0].size()),
                output_size=list(output.size()),
                num_parameters=params,
                multiply_adds=flops)
        )

    def add_hooks(module):
        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for h in hooks:
        h.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        self.PCK_stats = {}
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return
    
    def update(self, val, n=1, others=None):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
        if others is not None and'correct_cnt' in others:
            if 'sum' not in self.PCK_stats:
                self.PCK_stats['sum'] = np.zeros(len(others['correct_cnt'])) 
            self.PCK_stats['sum'] += others['correct_cnt']
            if 'total' not in self.PCK_stats:
                self.PCK_stats['total'] = 0.
            self.PCK_stats['total'] += n         
        return
    
    def print_content(self):
        if 'sum' in self.PCK_stats:
            for idx, value in enumerate(self.PCK_stats['sum']):
                PCK = value / self.PCK_stats['total']
                print('Average PCK at threshold {:.2f}: {:.3f}'.format(PCK_THRES[idx], PCK))
        return