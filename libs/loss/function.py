"""
Loss functions.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance_matrix

from libs.common.img_proc import soft_arg_max, appro_cr


loss_dict = {'mse': nn.MSELoss(reduction='mean'),
             'sl1': nn.SmoothL1Loss(reduction='mean'),
             'l1': nn.L1Loss(reduction='mean')
             }

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight, meta=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

def get_comp_dict(spec_list = ['None', 'None', 'None'], 
                  loss_weights = [1,1,1]
                  ):
    comp_dict = {}

    if spec_list[0] != 'None':
        comp_dict['hm'] = (loss_dict[spec_list[0]], loss_weights[0])
    if spec_list[1] != 'None':
        comp_dict['coor'] = (loss_dict[spec_list[1]], loss_weights[1])
    if spec_list[2] != 'None':
        comp_dict['cr'] = (loss_dict[spec_list[2]], loss_weights[2])     
    return comp_dict

class JointsCompositeLoss(nn.Module):
    """
    Loss function for 2d screen coordinate regression which consists of 
    multiple terms.
    """
    def __init__(self,
                 spec_list,
                 img_size,
                 hm_size,
                 loss_weights = [1,1,1],
                 target_cr = None,
                 cr_loss_thres = 0.15,
                 use_target_weight=False
                 ):
        """
        comp_dict specify the optional terms used in the loss computation, 
        which is specified with spec_list.
        loss for each component follows the format of [loss_type, weight],
        loss_type speficy the loss type for each component (e.g. L1 or L2) while
        weight gives the weight for this component.
        
        hm: a supervised loss defined with a heatmap target
        coor: a supervised loss defined with 2D coordinates
        cr: a self-supervised loss defined with prior cross-ratio
        """
        super(JointsCompositeLoss, self).__init__()
        self.comp_dict = get_comp_dict(spec_list, loss_weights)
        self.img_size = img_size
        self.hm_size = hm_size
        self.target_cr = target_cr
        self.use_target_weight = use_target_weight
        self.apply_cr_loss = False
        self.cr_loss_thres = cr_loss_thres

    def calc_hm_loss(self, output, target):
        """
        Heatmap loss which corresponds to L_{hm} in the paper.
        
        output: predicted heatmaps of shape [N, K, H, W]
        target: ground truth heatmaps of shape [N, K, H, W]
        """        
        batch_size = output.size(0)
        num_parts = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_parts, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_parts, -1)).split(1, 1)
        loss = 0
        for idx in range(num_parts):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            loss += 0.5 * self.comp_dict['hm'][0](heatmap_pred, heatmap_gt)        
        return loss / num_parts
    
    def calc_cross_ratio_loss(self, pred_coor, target_cr, mask):
        """
        Cross-ratio loss which corresponds to L_{cr} in the paper.
        
        pred_coor: predicted local coordinates
        target_cr: ground truth cross ratio
        """  
        assert hasattr(self, 'cr_indices')
        # this indices is assumed to be initialized by the user
        loss = 0
        mask = mask.to(pred_coor.device)
        if mask.sum() == 0:
            return loss
        for sample_idx in range(len(pred_coor)):
            for line_idx in range(len(self.cr_indices)):
                if mask[sample_idx][line_idx] == 0:
                    continue
                # predicted cross-ratio square
                pred_cr_sqr = appro_cr(pred_coor[sample_idx][self.cr_indices[line_idx]])
                # normalize the predicted cross-ratio square
                pred_cr_sqr /= target_cr**2
                line_loss = self.comp_dict['cr'][0](pred_cr_sqr, torch.ones(1).to(pred_cr_sqr.device))
                loss += line_loss * mask[sample_idx][line_idx][0]
        return loss/mask.sum()
    
    def get_cr_mask(self, coordinates, threshold = 0.15):
        """
        Mask some edges out when computing the cross-ratio loss.
        Ignore the fore-shortened edges since they will produce large and 
        unstable gradient.
        """          
        assert hasattr(self, 'cr_indices')
        mask = torch.zeros(coordinates.shape[0], len(self.cr_indices), 1)
        for sample_idx in range(len(coordinates)):
            for line_idx in range(len(self.cr_indices)):
                pts = coordinates[sample_idx][self.cr_indices[line_idx]]
                dm = distance_matrix(pts, pts)
                minval = np.min(dm[np.nonzero(dm)])
                if minval > threshold:
                    mask[sample_idx][line_idx] = 1.0
        return mask
    
    def calc_colinear_loss(self):
        # DEPRECATED
        return 0.
    
    def calc_coor_loss(self, coordinates_pred, coordinates_gt):
        """
        Coordinate loss which corresponds to L_{2d} in the paper.
        coordinates_pred: [N, K, 2]
        coordinates_gt: [N, K, 2]
        """  
        coordinates_gt[:, :, 0] /= self.img_size[0]
        coordinates_gt[:, :, 1] /= self.img_size[1]   
        loss = self.comp_dict['coor'][0](coordinates_pred, coordinates_gt) 
        return loss
    
    def forward(self, output, target, target_weight=None, meta=None):
        """
        Loss evaluation.
        Output is in the format of (heatmaps, coordinates) where coordinates
        is optional.
        target refers to the ground truth heatmaps.
        """  
        if type(output) is tuple:
            heatmaps_pred, coordinates_pred = output
        else:
            heatmaps_pred, coordinates_pred = output, None
        total_loss = 0
        if 'hm' in self.comp_dict:
            # some heatmaps map be produced by unlabeled data
            if len(heatmaps_pred) != len(target):
                heatmaps_pred = heatmaps_pred[:len(target)]
            total_loss += self.calc_hm_loss(heatmaps_pred, target) * self.comp_dict['hm'][1]
        if 'coor' in self.comp_dict:
            coordinates_gt = meta['transformed_joints'][:, :, :2].astype(np.float32)
            coordinates_gt = torch.from_numpy(coordinates_gt).cuda()           
            if coordinates_pred == None:
                coordinates_pred, max_vals = soft_arg_max(heatmaps_pred)
                coordinates_pred[:, :, 0] /= self.hm_size[1]
                coordinates_pred[:, :, 1] /= self.hm_size[0]     
            if len(coordinates_pred) != len(coordinates_gt):
                coordinates_pred_fs = coordinates_pred[:len(coordinates_gt)]
            else:
                coordinates_pred_fs = coordinates_pred
            total_loss += self.calc_coor_loss(coordinates_pred_fs, coordinates_gt) * self.comp_dict['coor'][1] 
        if 'cr' in self.comp_dict and self.comp_dict['cr'][1] != "None" and self.apply_cr_loss:
            cr_loss_mask = self.get_cr_mask(coordinates_pred.clone().detach().data.cpu().numpy(), self.cr_loss_thres)
            total_loss += self.calc_cross_ratio_loss(coordinates_pred, self.target_cr, cr_loss_mask) * self.comp_dict['cr'][1]
        return total_loss
    
class MSELoss1D(nn.Module):
    """
    Mean squared error loss.
    """     
    def __init__(self, use_target_weight=False, reduction='mean'):
        super(MSELoss1D, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None, meta=None):
        loss = self.criterion(output, target)
        return loss
    
class SmoothL1Loss1D(nn.Module):
    """
    Smooth L1 loss.
    """
    def __init__(self, use_target_weight=False):
        super(SmoothL1Loss1D, self).__init__()
        self.criterion = nn.SmoothL1Loss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None, meta=None):
        loss = self.criterion(output, target)
        return loss

class DecoupledSL1Loss(nn.Module):
    # DEPRECATED
    def __init__(self, use_target_weight=None):
        super(DecoupledSL1Loss, self).__init__()
        self.criterion = F.smooth_l1_loss

    def forward(self, output, target, target_weight=None):
        # balance the loss for translation and rotation regression
        loss_center = self.criterion(output[:, :3], target[:, :3], reduction='mean')
        loss_else = self.criterion(output[:, 3:], target[:, 3:], reduction='mean')
        return loss_center + loss_else
    
class JointsOHKMMSELoss(nn.Module):
    # DEPRECATED
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)

class WingLoss(nn.Module):
    # DEPRECATED
    def __init__(self, use_target_weight, width=5, curvature=0.5, image_size=(384, 288)):
        super(WingLoss, self).__init__()
        self.width = width
        self.curvature = curvature
        self.C = self.width - self.width * np.log(1 + self.width / self.curvature)
        self.image_size = image_size
        
    def forward(self, output, target, target_weight):
        prediction, _ = soft_arg_max(output)
        # normalize the coordinates to 0-1
        prediction[:, :, 0] /= self.image_size[1]
        prediction[:, :, 1] /= self.image_size[0]
        target[:, :, 0] /= self.image_size[1]
        target[:, :, 1] /= self.image_size[0]  
        diff = target - prediction
        diff_abs = diff.abs()
        loss = diff_abs.clone()

        idx_smaller = diff_abs < self.width
        idx_bigger = diff_abs >= self.width

        loss[idx_smaller] = self.width * torch.log(1 + diff_abs[idx_smaller] / self.curvature)
        loss[idx_bigger]  = loss[idx_bigger] - self.C
        loss = loss.mean()
        return loss