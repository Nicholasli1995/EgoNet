# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Shichao Li (nicholas.li@connect.ust.hk)
# ------------------------------------------------------------------------------

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn

import numpy as np

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def basicdownsample(in_planes, out_planes):
    downsample = nn.Sequential(
    nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1, 
        stride=2, 
        bias=False
        ),
    nn.BatchNorm2d(
        out_planes
        ),
    )
    return downsample

class BasicLinearModule(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=512):
        super(BasicLinearModule, self).__init__()
        self.l1 = nn.Linear(in_channels, out_channels)
        # self.l1 = nn.Linear(in_channels, mid_channels)
        # self.bn1 = nn.BatchNorm1d(mid_channels, momentum=BN_MOMENTUM)
        # self.relu = nn.ReLU(inplace=True)
        # self.l2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        x = x.view(len(x), -1)

        out = self.l1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        # out = self.l2(out)
        return out
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'basic': BasicBlock,
    'bottleneck': Bottleneck
}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfgs, **kwargs):
        self.inplanes = 64
        self.num_joints = cfgs['heatmapModel']['num_joints']
        extra = cfgs['heatmapModel']['extra']
        super(PoseHighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = cfgs['heatmapModel']['extra']['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block = blocks_dict[self.stage2_cfg['block']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfgs['heatmapModel']['extra']['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block = blocks_dict[self.stage3_cfg['block']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfgs['heatmapModel']['extra']['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block = blocks_dict[self.stage4_cfg['block']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)
        
        self.pretrained_layers = cfgs['heatmapModel']['extra']['pretrained_layers']
        
        # network head
        self.head_type = cfgs['heatmapModel']['head_type']
        self.pixel_shuffle = cfgs['heatmapModel']['pixel_shuffle']
        if self.head_type == 'heatmap':
            self.final_layer = nn.Conv2d(
                in_channels=pre_stage_channels[0],
                out_channels=self.num_joints,
                kernel_size=extra['final_conv_kernel'],
                stride=1,
                padding=1 if extra['final_conv_kernel'] == 3 else 0
            )
            
            if cfgs['heatmapModel']['pixel_shuffle']:
            # Add a pixel shuffle upsampling layer to control the heatmap size
                self.upsamp_fact = int(cfgs['heatmapModel']['heatmap_size'][0]\
                    /cfgs['heatmapModel']['input_size'][0]*4)
                self.upsample_layer = nn.Sequential(
                        nn.Conv2d(self.num_joints, self.num_joints*self.upsamp_fact**2, 
                                  kernel_size=1),
                        nn.BatchNorm2d(self.num_joints*self.upsamp_fact**2),
                        nn.ReLU(inplace=True),                
                        nn.PixelShuffle(self.upsamp_fact)
                        )
        elif self.head_type == 'angleregression':
            num_chan = 256
            self.head = nn.Sequential(
                nn.Conv2d(
                    in_channels=pre_stage_channels[0],
                    out_channels=num_chan,
                    kernel_size=1,
                    stride=1,
                    padding=0
                    ), 
                # produce 8*8*num_joints tensor
                BasicBlock(num_chan, 
                           num_chan, 
                           stride=2,
                           downsample=basicdownsample(num_chan, num_chan)
                           ),
                BasicBlock(num_chan, 
                           num_chan, 
                           stride=2,
                           downsample=basicdownsample(num_chan, num_chan)
                           ),
                BasicBlock(num_chan, 
                           num_chan, 
                           stride=2,
                           downsample=basicdownsample(num_chan, num_chan)
                           ),
                BasicBlock(num_chan, 
                           num_chan, 
                           stride=2,
                           downsample=basicdownsample(num_chan, num_chan)
                           ),
                nn.AvgPool2d(kernel_size=4),                
                )
            self.final_fc = nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 2)
                )
        elif self.head_type == 'coordinates':
            num_chan = 33
            self.head1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=pre_stage_channels[0],
                    out_channels=self.num_joints,
                    kernel_size=1,
                    stride=1,
                    padding=0
                    ), 
                )
            self.head2 = nn.Sequential(
                # produce 8*8*num_joints tensor
                BasicBlock(num_chan+2, 
                           num_chan*2, 
                           stride=2,
                           downsample=basicdownsample(num_chan+2, num_chan*2)
                           ),
                BasicBlock(num_chan*2, 
                           num_chan*2, 
                           stride=2,
                           downsample=basicdownsample(num_chan*2, num_chan*2)
                           ),
                BasicBlock(num_chan*2, 
                           num_chan*2, 
                           stride=2,
                           downsample=basicdownsample(num_chan*2, num_chan*2)
                           ),
                BasicBlock(num_chan*2, 
                           num_chan*2, 
                           stride=2,
                           downsample=basicdownsample(num_chan*2, num_chan*2)
                           ),
                nn.Conv2d(num_chan*2, num_chan*2, kernel_size=4),
                nn.Sigmoid()
                ) 
            # coordinate convolution makes arg-max easier
            map_height, map_width = cfgs['heatmapModel']['heatmap_size']
            x_map = np.tile(np.linspace(0, 1, map_width), (map_height, 1))
            x_map = x_map.reshape(1, 1, map_height, map_width)
            y_map = np.linspace(0, 1, map_height).reshape(map_height, 1)
            y_map = np.tile(y_map, (1, map_width))
            y_map = y_map.reshape(1, 1, map_height, map_width)
            self.coor_maps = np.concatenate([x_map, y_map], axis=1).astype(np.float32)
            self.coor_maps = torch.from_numpy(self.coor_maps)
        else:
            raise NotImplementedError

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = blocks_dict[layer_config['block']]
        fuse_method = layer_config['fuse_method']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        
        if self.head_type == 'heatmap':
            x = self.final_layer(y_list[0])        
            # upsampling
            if self.pixel_shuffle:
                x = self.upsample_layer(x)
        elif self.head_type == 'coordinates':
            maps = self.head1(y_list[0])
            # concatenate coordinate maps
            num_sample = len(maps)
            coor_maps = self.coor_maps.repeat(num_sample, 1, 1, 1).to(maps.device)
            augmented_maps = torch.cat([maps, coor_maps], dim=1)
            coordinates = self.head2(augmented_maps)
            x = (maps, coordinates.view(len(x), -1, 2))
        elif self.head_type == 'angleregression':
            maps = self.head(y_list[0])
            x = self.final_fc(maps.reshape(len(maps), -1))
        else:
            raise NotImplementedError()
        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] == '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} does not exist!'.format(pretrained))
    
    def modify_input_channel(self, num_channels):
        if num_channels == 3:
            return
        new_layer = nn.Conv2d(num_channels, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        # copy the old weights
        with torch.no_grad():
            new_layer.weight[:,:3,:,:] = self.conv1.weight.clone()
        del self.conv1
        self.conv1 = new_layer
        return
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            param = param.data
            own_state[name].copy_(param)
            
def get_pose_net(cfgs, is_train, **kwargs):
    model = PoseHighResolutionNet(cfgs, **kwargs)

    if is_train and cfgs['heatmapModel']['init_weights']:
        model.init_weights(cfgs['heatmapModel']['pretrained'])

    if cfgs['heatmapModel']['add_xy']:
        model.modify_input_channel(5)
    return model
