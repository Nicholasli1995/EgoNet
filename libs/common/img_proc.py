"""
Image processing utilities.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os

SIZE = 200.0

def transform_preds(coords, center, scale, output_size):
    """
    Transform local coordinates within a patch to screen coordinates.
    """      
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def get_affine_transform(center, 
                         scale, 
                         rot, 
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32), 
                         inv=0
                         ):
    """
    Estimate an affine transformation given crop parameters (center, scale and
    rotation) and output resolution.                                                        
    """  
    if isinstance(scale, list):
        scale = np.array(scale)
    if isinstance(center, list):
        center = np.array(center)
    scale_tmp = scale * SIZE
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def affine_transform_modified(pts, t):
    """
    Apply affine transformation with homogeneous coordinates.                                                    
    """ 
    # pts of shape [n, 2]
    new_pts = np.hstack([pts, np.ones((len(pts), 1))]).T
    new_pts = t @ new_pts
    return new_pts[:2, :].T

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def crop(img, center, scale, output_size, rot=0):
    """
    A cropping function implemented as warping.                                                      
    """     
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img, 
                             trans, 
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR
                             )   

    return dst_img

def simple_crop(input_image, center, crop_size):
    """
    A simple cropping function without warping.
    """  
    assert len(input_image.shape) == 3, 'Unsupported image format.'
    channel = input_image.shape[2]
    # crop a rectangular region around the center in the image
    start_x = int(center[0] - crop_size[0])
    end_x = int(center[0] + crop_size[0]) 
    start_y = int(center[1] - crop_size[1])
    end_y = int(center[1] + crop_size[1])
    cropped = np.zeros((end_y - start_y, end_x - start_x, channel), 
                       dtype = input_image.dtype)
    # new bounding box index 
    new_start_x = max(-start_x, 0)
    new_end_x = min(input_image.shape[1], end_x) - start_x
    new_start_y = max(-start_y, 0)
    new_end_y = min(input_image.shape[0], end_y) - start_y
    # clamped old bounding box index
    old_start_x = max(start_x, 0)
    old_end_x = min(end_x, input_image.shape[1])
    old_start_y = max(start_y, 0)
    old_end_y = min(end_y, input_image.shape[0])
    try:
        cropped[new_start_y:new_end_y, new_start_x:new_end_x,:] = input_image[
            old_start_y:old_end_y, old_start_x:old_end_x,:]
    except ValueError:
        print('Error: cropping fails')
    return cropped

def np_random():
    """
    Return a random number sampled uniformly from [-1, 1]
    """
    return np.random.rand()*2 - 1

def jitter_bbox_with_kpts(old_bbox, joints, parameters):
    """
    Randomly shifting and resizeing a bounding box and mask out occluded joints.
    Used as data augmentation to improve robustness to detector noise.
    
    bbox: [x1, y1, x2, y2]
    joints: [N, 3]
    """
    new_joints = joints.copy()
    width, height = old_bbox[2] - old_bbox[0], old_bbox[3] - old_bbox[1]
    old_center = [0.5*(old_bbox[0] + old_bbox[2]), 
                  0.5*(old_bbox[1] + old_bbox[3])]
    horizontal_shift = parameters['shift'][0]*width*np_random()
    vertical_shift = parameters['shift'][1]*height*np_random()
    new_center = [old_center[0] + horizontal_shift,
                  old_center[1] + vertical_shift]
    horizontal_scaling = parameters['scaling'][0]*np_random() + 1
    vertical_scaling = parameters['scaling'][1]*np_random() + 1
    new_width = width*horizontal_scaling
    new_height = height*vertical_scaling
    new_bbox = [new_center[0] - 0.5*new_width, new_center[1] - 0.5*new_height,
                new_center[0] + 0.5*new_width, new_center[1] + 0.5*new_height]
    # predicate from upper left corner
    predicate1 = joints[:, :2] - np.array([[new_bbox[0], new_bbox[1]]])
    predicate1 = (predicate1 > 0.).prod(axis=1)
    # predicate from lower right corner
    predicate2 = joints[:, :2] - np.array([[new_bbox[2], new_bbox[3]]])
    predicate2 = (predicate2 < 0.).prod(axis=1)
    new_joints[:, 2] *= predicate1*predicate2
    return new_bbox, new_joints

def jitter_bbox_with_kpts_no_occlu(old_bbox, joints, parameters):
    """
    Similar to the function above, but does not produce occluded joints
    """
    width, height = old_bbox[2] - old_bbox[0], old_bbox[3] - old_bbox[1]
    old_center = [0.5*(old_bbox[0] + old_bbox[2]), 
                  0.5*(old_bbox[1] + old_bbox[3])]
    horizontal_scaling = parameters['scaling'][0]*np.random.rand() + 1
    vertical_scaling = parameters['scaling'][1]*np.random.rand() + 1
    horizontal_shift = 0.5*(horizontal_scaling-1)*width*np_random()
    vertical_shift = 0.5*(vertical_scaling-1)*height*np_random()
    new_center = [old_center[0] + horizontal_shift,
                  old_center[1] + vertical_shift]
    new_width = width*horizontal_scaling
    new_height = height*vertical_scaling
    new_bbox = [new_center[0] - 0.5*new_width, new_center[1] - 0.5*new_height,
                new_center[0] + 0.5*new_width, new_center[1] + 0.5*new_height]
    return new_bbox, joints

def generate_xy_map(bbox, resolution, global_size):
    """
    Generate the normalized coordinates as 2D maps which encodes location 
    information.
    
    bbox: [x1, y1, x2, y2] the local region
    resolution (height, width): target resolution
    global_size (height, width): the size of original image
    """
    map_height, map_width = resolution
    g_height, g_width = global_size
    x_start, x_end = 2*bbox[0]/g_width - 1, 2*bbox[2]/g_width - 1
    y_start, y_end = 2*bbox[1]/g_height - 1, 2*bbox[3]/g_height - 1
    x_map = np.tile(np.linspace(x_start, x_end, map_width), (map_height, 1))
    x_map = x_map.reshape(map_height, map_width, 1)
    y_map = np.linspace(y_start, y_end, map_height).reshape(map_height, 1)
    y_map = np.tile(y_map, (1, map_width))
    y_map = y_map.reshape(map_height, map_width, 1)
    return np.concatenate([x_map, y_map], axis=2)

def crop_single_instance(data_numpy, bbox, joints, parameters, pth_trans=None):
    """
    Crop an instance from an image given the bounding box and part coordinates.
    """
    reso = parameters['input_size']  
    transformed_joints = joints.copy()
    if parameters['jitter_bbox']:
        bbox, joints = jitter_bbox_with_kpts_no_occlu(bbox, 
                                                      joints,
                                                      parameters['jitter_params']
                                                      )
    joints_vis = joints[:, 2]
    if parameters['resize']:
        ret = resize_bbox(bbox[0], bbox[1], bbox[2], bbox[3], 
                          target_ar=reso[0]/reso[1])
        c, s = ret['c'], ret['s']
    else:
        c, s = bbox2cs(bbox)    
    trans = get_affine_transform(c, s, 0.0, reso)
    input = cv2.warpAffine(data_numpy,
                           trans,
                           (int(reso[0]), int(reso[1])),
                           flags=cv2.INTER_LINEAR
                           )
    # add two more channels to encode object location
    if parameters['add_xy']:
        xymap = generate_xy_map(ret['bbox'], reso, parameters['global_size'])
        input = np.concatenate([input, xymap.astype(np.float32)], axis=2)
    #cv2.imwrite('test.jpg', input)
    #input = torch.from_numpy(input.transpose(2,0,1))
    input = input if pth_trans is None else pth_trans(input)
    for i in range(len(joints)):
        if joints_vis[i] > 0.0:
            transformed_joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)   
    c = c.reshape(1, 2)
    s = s.reshape(1, 2)
    return input.unsqueeze(0), transformed_joints, c, s

def get_tensor_from_img(path, 
                        parameters,
                        sf=0.2, 
                        rf=30., 
                        r_prob=0.6, 
                        aug=False, 
                        rgb=True, 
                        joints=None,
                        global_box=None,
                        pth_trans=None,
                        generate_hm=False,
                        max_cnt=None
                        ):
    """
    Read image and apply data augmentation to obtain a tensor. 
    Keypoints are also transformed if given.
    
    path: image path
    c: cropping center
    s: cropping scale
    r: rotation
    reso: resolution of output image
    sf: scaling factor
    rf: rotation factor
    aug: apply data augmentation
    joints: key-point locations with optional visibility [N_instance, N_joint, 3]
    generate_hm: whether to generate heatmap based on joint locations
    """
#    data_numpy = cv2.imread(
#        path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
#        )
    data_numpy = cv2.imread(
        path, 1 | 128
        )    
    if data_numpy is None:
        raise ValueError('Fail to read {}'.format(path))    
    if rgb:
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
    all_inputs = []
    all_target = []
    all_centers = []
    all_scales = []
    all_target_weight = []
    # the dimension of the image
    parameters['global_size'] = data_numpy.shape[:-1]
    all_transformed_joints = []
    if parameters['reference'] == 'bbox':
        # crop around the given bounding boxes
        # bbox = [0, 0, data_numpy.shape[1] - 1, data_numpy.shape[0] - 1] \
        #     if 'bbox' not in parameters else parameters['bbox']
        bboxes = parameters['boxes'] # [N_instance, 4]
        for idx, bbox in enumerate(bboxes):
            input, transformed_joints, c, s = crop_single_instance(data_numpy,
                                                                   bbox,
                                                                   joints[idx],
                                                                   parameters,
                                                                   pth_trans
                                                                   )
            all_inputs.append(input)
            all_centers.append(c)
            all_scales.append(s)
        # s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        # r = np.clip(np.random.randn() * rf, -rf, rf) if np.random.rand() <= r_prob else 0
            target = target_weight = 1.
            if generate_hm:
                target, target_weight = generate_target(transformed_joints, 
                                                        transformed_joints[:,2], 
                                                        parameters)
                target = torch.unsqueeze(torch.from_numpy(target), 0)
                target_weight = torch.unsqueeze(torch.from_numpy(target_weight), 0)
            all_target.append(target)
            all_target_weight.append(target_weight)
            all_transformed_joints.append(np.expand_dims(transformed_joints,0))
    all_transformed_joints = np.concatenate(all_transformed_joints)
    if max_cnt is not None and max_cnt < len(all_inputs):
        end = max_cnt
    else:
        end = len(all_inputs)
    end_indices = list(range(end))
    meta = {
        'path': path,
        'original_joints': joints[end_indices],
        'transformed_joints': all_transformed_joints[end_indices],
        'center': np.vstack(all_centers[:end]),
        'scale': np.vstack(all_scales[:end]),
        'joints_vis': all_transformed_joints[end_indices][:,:,2]
        # 'rotation': r,
    }
    inputs = torch.cat(all_inputs[:end], dim=0)
    if generate_hm:
        targets = torch.cat(all_target[:end], dim=0)
        target_weights = torch.cat(all_target_weight[:end], dim=0)
    else:
        targets, target_weights = None, None
    return inputs, targets, target_weights, meta

def generate_target(joints, joints_vis, parameters):
    """
    Generate heatmap targets by drawing Gaussian dots.
    
    joints:  [num_joints, 3]
    joints_vis: [num_joints]
    
    return: target, target_weight (1: visible, 0: invisible)
    """
    num_joints = parameters['num_joints']
    target_type = parameters['target_type']
    input_size = parameters['input_size']
    heatmap_size = parameters['heatmap_size']
    sigma = parameters['sigma']
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis

    
    assert target_type == 'gaussian', 'Only support gaussian map now!'

    if target_type == 'gaussian':
        target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]), 
                          dtype=np.float32)

        tmp_size = sigma * 3

        for joint_id in range(num_joints):
            if target_weight[joint_id] <= 0.5:
                continue
            feat_stride = input_size / heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    if parameters['use_different_joints_weight']:
        target_weight = np.multiply(target_weight, parameters['joints_weight'])

    return target, target_weight

def resize_bbox(left, top, right, bottom, target_ar=1.):
    """
    Resize a bounding box to pre-defined aspect ratio.
    """ 
    width = right - left
    height = bottom - top
    aspect_ratio = height/width
    center_x = (left + right)/2
    center_y = (top + bottom)/2
    if aspect_ratio > target_ar:
        new_width = height*(1/target_ar)
        new_left = center_x - 0.5*new_width
        new_right = center_x + 0.5*new_width
        new_top = top
        new_bottom = bottom        
    else:
        new_height = width*target_ar
        new_left = left
        new_right = right
        new_top = center_y - 0.5*new_height
        new_bottom = center_y + 0.5*new_height
    return {'bbox': [new_left, new_top, new_right, new_bottom],
            'c': np.array([center_x, center_y]),
            's': np.array([(new_right - new_left)/SIZE, (new_bottom - new_top)/SIZE])
            }

def enlarge_bbox(left, top, right, bottom, enlarge):
    """
    Enlarge a bounding box.
    """ 
    width = right - left
    height = bottom - top
    new_width = width * enlarge[0]
    new_height = height * enlarge[1]
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2    
    new_left = center_x - 0.5 * new_width
    new_right = center_x + 0.5 * new_width
    new_top = center_y - 0.5 * new_height
    new_bottom = center_y + 0.5 * new_height
    return [new_left, new_top, new_right, new_bottom]

def modify_bbox(bbox, target_ar, enlarge=1.1):
    """
    Modify a bounding box by enlarging/resizing.
    """
    lbbox = enlarge_bbox(bbox[0], bbox[1], bbox[2], bbox[3], [enlarge, enlarge])
    ret = resize_bbox(lbbox[0], lbbox[1], lbbox[2], lbbox[3], target_ar=target_ar)
    return ret
    
def resize_crop(crop_size, target_ar=None):
    """
    Resize a crop size to a pre-defined aspect ratio.
    """    
    if target_ar is None:
        return crop_size
    width = crop_size[0]
    height = crop_size[1]
    aspect_ratio = height / width    
    if aspect_ratio > target_ar:
        new_width = height * (1 / target_ar)
        new_height = height
    else:
        new_height = width*target_ar
        new_width = width
    return [new_width, new_height]

def bbox2cs(bbox):
    """
    Convert bounding box annotation to center and scale.
    """  
    return [(bbox[0] + bbox[2]/2), (bbox[1] + bbox[3]/2)], \
        [(bbox[2] - bbox[0]/SIZE), (bbox[3] - bbox[1]/SIZE)]

def cs2bbox(center, size):
    """
    Convert center/scale to a bounding box annotation.
    """  
    x1 = center[0] - size[0]
    y1 = center[1] - size[1]
    x2 = center[0] + size[0]
    y2 = center[1] + size[1]
    return [x1, y1, x2, y2]

def kpts2cs(keypoints, 
            enlarge=1.1, 
            method='boundary', 
            target_ar=None, 
            use_visibility=True
            ):
    """
    Convert instance screen coordinates to cropping center and size
    
    keypoints of shape [n_joints, 2/3]
    """   
    assert keypoints.shape[1] in [2, 3], 'Unsupported input.'
    if keypoints.shape[1] == 2:
        visible_keypoints = keypoints
        vis_rate = 1.0
    elif keypoints.shape[1] == 3 and use_visibility:
        visible_indices = keypoints[:, 2].nonzero()[0]
        visible_keypoints = keypoints[visible_indices, :2]
        vis_rate = len(visible_keypoints)/len(keypoints)
    else:
        visible_keypoints = keypoints[:, :2]
        visible_indices = np.array(range(len(keypoints)))
        vis_rate = 1.0
    if method == 'centroid':
        center = np.ceil(visible_keypoints.mean(axis=0, keepdims=True))
        dif = np.abs(visible_keypoints - center).max(axis=0, keepdims=True)
        crop_size = np.ceil(dif*enlarge).squeeze()
        center = center.squeeze()
    elif method == 'boundary':
        left_top = visible_keypoints.min(axis=0, keepdims=True)
        right_bottom = visible_keypoints.max(axis=0, keepdims=True)
        center = ((left_top + right_bottom) / 2).squeeze()
        crop_size = ((right_bottom - left_top)*enlarge/2).squeeze()
    else:
        raise NotImplementedError
    # resize the bounding box to a specified aspect ratio
    crop_size = resize_crop(crop_size, target_ar)
    x1, y1, x2, y2 = cs2bbox(center, crop_size)

    new_origin = np.array([[x1, y1]], dtype=keypoints.dtype)
    new_keypoints = keypoints.copy()
    if keypoints.shape[1] == 2:
        new_keypoints = visible_keypoints - new_origin
    elif keypoints.shape[1] == 3: 
        new_keypoints[visible_indices, :2] = visible_keypoints - new_origin
    return center, crop_size, new_keypoints, vis_rate

def draw_bboxes(img_path, bboxes_dict, save_path=None):
    """
    Draw bounding boxes with OpenCV.
    """
    data_numpy = cv2.imread(img_path, 1 | 128)  
    for name, (color, bboxes) in bboxes_dict.items():
        for bbox in bboxes:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            cv2.rectangle(data_numpy, start_point, end_point, color, 2)
    if save_path is not None:
        cv2.imwrite(save_path, data_numpy)
    return data_numpy

def imread_rgb(img_path):
    """
    Read image with OpenCV.
    """    
    data_numpy = cv2.imread(img_path, 1 | 128)  
    data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
    return data_numpy

def save_cropped_patches(img_path, 
                         keypoints, 
                         save_dir="./", 
                         threshold=0.25,
                         enlarge=1.4, 
                         target_ar=None
                         ):
    """
    Crop instances from a image given part screen coordinates and save them.
    """
#    data_numpy = cv2.imread(
#        img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
#        )   
    data_numpy = cv2.imread(img_path, 1 | 128)  
    # data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
    # debug
    # import matplotlib.pyplot as plt
    # plt.imshow(data_numpy[:,:,::-1])
    # plt.plot(keypoints[0][:,0], keypoints[0][:,1], 'ro')
    # plt.pause(0.1)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    new_paths = []
    all_new_keypoints = []
    all_bbox = []
    for i in range(len(keypoints)):
        center, crop_size, new_keypoints, vis_rate = kpts2cs(keypoints[i], 
                                                             enlarge, 
                                                             target_ar=target_ar)
        all_bbox.append(list(map(int, cs2bbox(center, crop_size))))
        if vis_rate < threshold:
            continue
        all_new_keypoints.append(new_keypoints.reshape(1, keypoints.shape[1], -1))
        cropped = simple_crop(data_numpy, center, crop_size)
        save_path = os.path.join(save_dir, "instance_{:d}.jpg".format(i))
        new_paths.append(save_path)
        cv2.imwrite(save_path, cropped)
        del cropped
    if len(new_paths) == 0:
        # No instances cropped
        return new_paths, np.zeros((0, keypoints.shape[1], 3)), all_bbox
    else:
        return new_paths, np.concatenate(all_new_keypoints, axis=0), all_bbox
    
def get_max_preds(batch_heatmaps):
    """
    Get predictions from heatmaps with hard arg-max.
    
    batch_heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def soft_arg_max_np(batch_heatmaps):
    """
    Soft-argmax instead of hard-argmax considering quantization errors.
    """
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    height = batch_heatmaps.shape[2]
    width = batch_heatmaps.shape[3]
    # get score/confidence for each joint
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    maxvals = np.amax(heatmaps_reshaped, 2)
    maxvals = maxvals.reshape((batch_size, num_joints, 1))    
    # normalize the heatmaps so that they sum to 1
    #assert batch_heatmaps.min() >= 0.0
    batch_heatmaps = np.clip(batch_heatmaps, a_min=0.0, a_max=None)
    temp_sum = heatmaps_reshaped.sum(axis = 2, keepdims=True)
    heatmaps_reshaped /= temp_sum
    ## another normalization method: softmax
    # spatial soft-max
    #heatmaps_reshaped = softmax(heatmaps_reshaped, axis=2)
    ##
    batch_heatmaps = heatmaps_reshaped.reshape(batch_size, num_joints, height, width)
    x = batch_heatmaps.sum(axis = 2)
    y = batch_heatmaps.sum(axis = 3)
    x_indices = np.arange(width).astype(np.float32).reshape(1,1,width)
    y_indices = np.arange(height).astype(np.float32).reshape(1,1,height)
    x *= x_indices
    y *= y_indices
    x = x.sum(axis = 2, keepdims=True)
    y = y.sum(axis = 2, keepdims=True)
    preds = np.concatenate([x, y], axis=2)
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask
    return preds, maxvals

def soft_arg_max(batch_heatmaps):
    """
    A pytorch version of soft-argmax
    """
    assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim'
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    height = batch_heatmaps.shape[2]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.view((batch_size, num_joints, -1))
    # get score/confidence for each joint    
    maxvals = heatmaps_reshaped.max(dim=2)[0]
    maxvals = maxvals.view((batch_size, num_joints, 1))       
    # normalize the heatmaps so that they sum to 1
    heatmaps_reshaped = F.softmax(heatmaps_reshaped, dim=2)
    batch_heatmaps = heatmaps_reshaped.view(batch_size, num_joints, height, width)
    x = batch_heatmaps.sum(dim = 2)
    y = batch_heatmaps.sum(dim = 3)
    x_indices = torch.arange(width).type(torch.cuda.FloatTensor)
    x_indices = torch.cuda.comm.broadcast(x_indices, devices=[x.device.index])[0]
    x_indices = x_indices.view(1, 1, width)
    y_indices = torch.arange(height).type(torch.cuda.FloatTensor)
    y_indices = torch.cuda.comm.broadcast(y_indices, devices=[y.device.index])[0]
    y_indices = y_indices.view(1, 1, height)    
    x *= x_indices
    y *= y_indices
    x = x.sum(dim = 2, keepdim=True)
    y = y.sum(dim = 2, keepdim=True)
    preds = torch.cat([x, y], dim=2)
    return preds, maxvals

def appro_cr(coordinates):
    """
    Approximate the square of cross-ratio along four ordered 2D points using 
    inner-product
    
    coordinates: PyTorch tensor of shape [4, 2]
    """
    AC = coordinates[2] - coordinates[0]
    BD = coordinates[3] - coordinates[1]
    BC = coordinates[2] - coordinates[1]
    AD = coordinates[3] - coordinates[0]
    return (AC.dot(AC) * BD.dot(BD)) / (BC.dot(BC) * AD.dot(AD))

def to_npy(tensor):
    """
    Convert PyTorch tensor to numpy array.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    else:
        return tensor.data.cpu().numpy()