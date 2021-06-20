"""
Visualization utilities for Ego-Net inference.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

import libs.visualization.points as vp

def plot_2d_objects(img_path, record, color_dict):
    if 'plots' in record:
        # update old drawing
        fig = record['plots']['fig2d']
        ax = record['plots']['ax2d']
    else:
        # new drawing
        fig = plt.figure(figsize=(11.3, 9))
        ax = plt.subplot(111)
        record['plots'] = {}
        record['plots']['fig2d'] = fig
        record['plots']['ax2d'] = ax
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[:, :, ::-1]
        height, width, _ = image.shape
        ax.imshow(image) 
        ax.set_xlim([0, width])
        ax.set_ylim([0, height])
        ax.invert_yaxis()
    for idx in range(len(record['kpts_2d_pred'])):
        kpts = record['kpts_2d_pred'][idx].reshape(-1, 2)
        bbox = record['bbox_resize'][idx]
        vp.plot_2d_bbox(ax, bbox, color_dict['bbox_2d'])
        # predicted key-points
        ax.plot(kpts[:, 0], kpts[:, 1], color_dict['kpts'][0])    
    if 'kpts_2d_gt' in record:
        # plot ground truth 2D screen coordinates
        for idx, kpts_gt in enumerate(record['kpts_2d_gt']):
            kpts_gt = kpts_gt.reshape(-1, 3)
            vp.plot_3d_bbox(ax, kpts_gt[1:, :2], color='g', linestyle='-.')
    if 'arrow' in record:
        for idx in range(len(record['arrow'])):
            start = record['arrow'][idx][:,0]
            end = record['arrow'][idx][:,1]
            x, y = start
            dx, dy = end - start
            ax.arrow(x, y, dx, dy, color='r', lw=4, head_width=5, alpha=0.5) 
    # save intermediate results
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
    # hspace = 0, wspace = 0)
    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # img_name = img_path.split('/')[-1]
    # save_dir = './qualitative_results/'
    # plt.savefig(save_dir + img_name, dpi=100, bbox_inches = 'tight', pad_inches = 0)
    return record

def plot_3d_objects(prediction, target, pose_vecs_gt, record, color):
    if target is not None:
        p3d_gt = target.reshape(len(target), -1, 3)
    else:
        p3d_gt = None
    p3d_pred = prediction.reshape(len(prediction), -1, 3)
    if "kpts_3d_before" in record:
        # use predicted translation for visualization
        p3d_pred = np.concatenate([record['kpts_3d_before'][:, [0], :], p3d_pred], axis=1)
    elif p3d_gt is not None and p3d_gt.shape[1] == p3d_pred.shape[1] + 1:
        # use ground truth translation for visualization
        assert len(p3d_pred) == len(p3d_gt)
        p3d_pred = np.concatenate([p3d_gt[:, [0], :], p3d_pred], axis=1) 
    else:
        raise NotImplementedError
    if 'plots' in record and 'ax3d' in record['plots']:
        # update drawing
        ax = record['plots']['ax3d']
        ax = vp.plot_scene_3dbox(p3d_pred, p3d_gt, ax=ax, color=color)
    elif 'plots' in record:
        # plotting a set of 3D boxes
        ax = vp.plot_scene_3dbox(p3d_pred, p3d_gt, color=color)
        ax.set_title("GT: black w/o Ego-Net: magenta w/ Ego-Net: red/yellow")
        vp.draw_pose_vecs(ax, pose_vecs_gt)
        record['plots']['ax3d'] = ax
    else:
        raise NotImplementedError
    # draw pose angle predictions
    translation = p3d_pred[:, 0, :]    
    pose_vecs_pred = np.concatenate([translation, record['euler_angles']], axis=1)
    vp.draw_pose_vecs(ax, pose_vecs_pred, color=color)
    if 'kpts_3d_before' in record and 'plots' in record:
        # plot input 3D bounding boxes before using Ego-Net
        kpts_3d_before = record['kpts_3d_before']
        vp.plot_scene_3dbox(kpts_3d_before, ax=ax, color='m')    
        pose_vecs_before = np.zeros((len(kpts_3d_before), 6))
        for idx in range(len(pose_vecs_before)):
            pose_vecs_before[idx][0:3] = record['raw_txt_format'][idx]['locations']
            pose_vecs_before[idx][4] = record['raw_txt_format'][idx]['rot_y']
        vp.draw_pose_vecs(ax, pose_vecs_before, color='m')
    return record