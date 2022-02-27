"""
Utilities for saving debugging images.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

from libs.common.img_proc import get_max_preds
from libs.common.utils import make_dir

import math
import numpy as np
import torchvision
import cv2

from os.path import join

def draw_circles(ndarr, 
                 xmaps, 
                 ymaps, 
                 nmaps, 
                 batch_joints, 
                 batch_joints_vis, 
                 width, 
                 height, 
                 padding, 
                 color=[255,0,0],
                 add_idx=True
                 ):
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            for idx, joint in enumerate(joints):
                xpos = x * width + padding + joint[0]
                ypos = y * height + padding + joint[1]
                cv2.circle(ndarr, (int(xpos), int(ypos)), 2, color, 2)
                if add_idx:
                    cv2.putText(ndarr, 
                                str(idx+1), 
                                (int(xpos), int(ypos)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, color, 1
                                )
            k += 1
    return ndarr

# functions used for debugging heatmap-based keypoint localization model      #
def save_batch_image_with_joints(batch_image, 
                                 record_dict, 
                                 file_name, 
                                 nrow=8, 
                                 padding=2
                                 ):
    """
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    """
    grid = torchvision.utils.make_grid(batch_image[:, :3, :, :], nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()
    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    batch_joints, batch_joints_vis = record_dict['pred'] 
    ndarr = draw_circles(ndarr, xmaps, ymaps, nmaps, batch_joints, batch_joints_vis, 
                         width, height, padding)
    if 'gt' in record_dict:
        nmaps = min(nmaps, len(batch_joints_vis))
        xmaps = min(nrow, nmaps)
        ymaps = int(math.ceil(float(nmaps) / xmaps))
        batch_joints_gt, batch_joints_vis_gt = record_dict['gt']
        ndarr = draw_circles(ndarr, xmaps, ymaps, nmaps, batch_joints_gt, batch_joints_vis_gt, 
                             width, height, padding, color=[0,255,255])        
    cv2.imwrite(file_name, cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR))
    return

def save_batch_heatmaps(batch_image, 
                        batch_heatmaps, 
                        file_name,
                        normalize=True
                        ):
    """
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    """
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)
    return

def save_debug_images(epoch, 
                      batch_index, 
                      cfgs, 
                      input, 
                      meta, 
                      target, 
                      others, 
                      output, 
                      split
                      ):
    """
    Save debugging images during training HC.pth.
    """    
    if not cfgs['training_settings']['debug']['save']:
        return
    prefix = join(cfgs['dirs']['output'], 
                  "intermediate_results",
                  split, 
                  '{}_{}'.format(epoch, batch_index)
                  )
    make_dir(prefix)
    joints_pred = others['joints_pred']
    debug_cfgs = cfgs['training_settings']['debug']
    record_dict = {'pred':(joints_pred, meta['joints_vis']),
                   'gt':(meta['transformed_joints'], meta['joints_vis'])}    
    if debug_cfgs['save_images_kpts']:
        save_batch_image_with_joints(
            input[:,:3,:,:], record_dict, '{}_keypoints.jpg'.format(prefix)
        )
    if debug_cfgs['save_hms_gt']:
        save_batch_heatmaps(
            input[:,:3,:,:], target, '{}_hm_gt.jpg'.format(prefix)
        )
    if debug_cfgs['save_hms_pred']:
        output = output[0] if type(output) is tuple else output
        save_batch_heatmaps(
            input[:,:3,:,:], output, '{}_hm_pred.jpg'.format(prefix)
        )
    return