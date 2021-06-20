"""
A PyTorch implementation of Ego-Net.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import math

from scipy.spatial.transform import Rotation

import libs.model as models
import libs.model.FCmodel as FCmodel
import libs.dataset.normalization.operations as nop
import libs.visualization.points as vp
import libs.visualization.egonet_utils as vego
import libs.common.transformation as ltr

from libs.common.img_proc import to_npy, get_affine_transform, generate_xy_map, modify_bbox
from libs.common.img_proc import affine_transform_modified
from libs.common.format import save_txt_file, get_pred_str
from libs.dataset.KITTI.car_instance import interp_dict

class EgoNet(nn.Module):
    def __init__(self,
                 cfgs,
                 pre_trained=False
                 ):
        """
        Initialization method of Ego-Net.
        """
        super(EgoNet, self).__init__()
        # initialize a fully-convolutional heatmap regression model
        # this model corresponds to H and C in Equation (2)
        hm_model_settings = cfgs['heatmapModel']
        hm_model_name = hm_model_settings['name']
        # this implementation uses a HR-Net backbone, yet you can use other 
        # backbones as well
        method_str = 'models.heatmapModel.' + hm_model_name + '.get_pose_net'
        self.HC = eval(method_str)(cfgs, is_train=False)
        self.resolution = cfgs['heatmapModel']['input_size']
        # optional channel augmentation
        if 'add_xy' in cfgs['heatmapModel']:
            self.xy_dict = {'flag':cfgs['heatmapModel']['add_xy']}
        else:
            self.xy_dict = None
        # initialize a lifing model
        # this corresponds to L in Equation (2) 
        self.L = FCmodel.get_fc_model(stage_id=1, 
                                      cfgs=cfgs, 
                                      input_size=cfgs['FCModel']['input_size'],
                                      output_size=cfgs['FCModel']['output_size']
                                      )
        if pre_trained:
            # load pre-trained checkpoints
            self.HC.load_state_dict(torch.load(cfgs['dirs']['load_hm_model']))
            # the statistics used by the lifter for normalizing inputs
            self.LS = np.load(cfgs['dirs']['load_stats'], allow_pickle=True).item()
            self.L.load_state_dict(torch.load(cfgs['dirs']['load_lifter']))
    
    def crop_single_instance(self, 
                             img, 
                             bbox, 
                             resolution, 
                             pth_trans=None, 
                             xy_dict=None
                             ):
        """
        Crop a single instance given an image and bounding box.
        """
        bbox = to_npy(bbox)
        target_ar = resolution[0] / resolution[1]
        ret = modify_bbox(bbox, target_ar)
        c, s, r = ret['c'], ret['s'], 0.
        # xy_dict: parameters for adding xy coordinate maps
        trans = get_affine_transform(c, s, r, resolution)
        instance = cv2.warpAffine(img,
                                  trans,
                                  (int(resolution[0]), int(resolution[1])),
                                  flags=cv2.INTER_LINEAR
                                  )
        #cv2.imwrite('instance.jpg', instance)
        if xy_dict is not None and xy_dict['flag']:
            xymap = generate_xy_map(ret['bbox'], resolution, img.shape[:-1])
            instance = np.concatenate([instance, xymap.astype(np.float32)], axis=2)        
        instance = instance if pth_trans is None else pth_trans(instance)
        return instance
    
    def load_cv2(self, path, rgb=True):
        data_numpy = cv2.imread(path, 1 | 128)    
        if data_numpy is None:
            raise ValueError('Fail to read {}'.format(path))    
        if rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)         
        return data_numpy
    
    def crop_instances(self, 
                       annot_dict, 
                       resolution, 
                       pth_trans=None, 
                       rgb=True,
                       xy_dict=None
                       ):
        """
        Crop input instances given an annotation dictionary.
        """
        all_instances = []
        # each record stores attributes of one instance
        all_records = []
        target_ar = resolution[0] / resolution[1]
        for idx, path in enumerate(annot_dict['path']):
            data_numpy = self.load_cv2(path)
            boxes = annot_dict['boxes'][idx]
            if 'labels' in annot_dict:
                labels = annot_dict['labels'][idx]
            else:
                labels = -np.ones((len(boxes)), dtype=np.int64)
            if 'scores' in annot_dict:
                scores = annot_dict['scores'][idx]
            else:
                scores = -np.ones((len(boxes)))
            if len(boxes) == 0:
                continue
            for idx, bbox in enumerate(boxes):
                # crop an instance with required aspect ratio
                instance = self.crop_single_instance(data_numpy,
                                                     bbox, 
                                                     resolution, 
                                                     pth_trans=pth_trans,
                                                     xy_dict=xy_dict
                                                     )
                bbox = to_npy(bbox)
                ret = modify_bbox(bbox, target_ar)
                c, s, r = ret['c'], ret['s'], 0.
                all_instances.append(torch.unsqueeze(instance, dim=0))
                all_records.append({
                    'path': path,
                    'center': c,
                    'scale': s,
                    'bbox': bbox,
                    'bbox_resize': ret['bbox'],
                    'rotation': r,
                    'label': labels[idx],
                    'score': scores[idx]
                    }
                    )
        return torch.cat(all_instances, dim=0), all_records

    def add_orientation_arrow(self, record):
        """
        Generate an arrow for each predicted orientation for visualization.
        """      
        pred_kpts = record['kpts_3d_pred']
        gt_kpts = record['kpts_3d_gt']
        K = record['K']
        arrow_2d = np.zeros((len(pred_kpts), 2, 2))
        for idx in range(len(pred_kpts)):
            vector_3d = (pred_kpts[idx][1] - pred_kpts[idx][5])
            arrow_3d = np.concatenate([gt_kpts[idx][0].reshape(3, 1), 
                                      (gt_kpts[idx][0] + vector_3d).reshape(3, 1)],
                                      axis=1)
            projected = K @ arrow_3d
            arrow_2d[idx][0] = projected[0, :] / projected[2, :]
            arrow_2d[idx][1] = projected[1, :] / projected[2, :]
            # fix the arrow length if not fore-shortened
            vector_2d = arrow_2d[idx][:,1] - arrow_2d[idx][:,0]
            length = np.linalg.norm(vector_2d)
            if length > 50:
                vector_2d = vector_2d/length * 60
            arrow_2d[idx][:,1] = arrow_2d[idx][:,0] + vector_2d
        return arrow_2d

    def write_annot_dict(self, annot_dict, records):
        for idx, path in enumerate(annot_dict['path']):
            if 'boxes' in annot_dict:
                records[path]['boxes'] = to_npy(annot_dict['boxes'][idx])
            if 'kpts' in annot_dict:
                records[path]['kpts_2d_gt'] = to_npy(annot_dict['kpts'][idx])   
            if 'kpts_3d_gt' in annot_dict:
                records[path]['kpts_3d_gt'] = to_npy(annot_dict['kpts_3d_gt'][idx])   
            if 'pose_vecs_gt' in annot_dict:            
                records[path]['pose_vecs_gt'] = to_npy(annot_dict['pose_vecs_gt'][idx])  
            if 'kpts_3d_before' in annot_dict:
                records[path]['kpts_3d_before'] = to_npy(annot_dict['kpts_3d_before'][idx])  
            if 'raw_txt_format' in annot_dict:
                # list of annotation dictionary for each instance
                records[path]['raw_txt_format'] = annot_dict['raw_txt_format'][idx]
            if 'K' in annot_dict:
                # list of annotation dictionary for each instance
                records[path]['K'] = annot_dict['K'][idx]
            if 'kpts_3d_gt' in annot_dict and 'K' in annot_dict:
                records[path]['arrow'] = self.add_orientation_arrow(records[path])
        return records

    def get_observation_angle_trans(self, euler_angles, translations):
        """
        Convert orientation in camera coordinate into local coordinate system
        utilizing known object location (translation)
        """ 
        alphas = euler_angles[:,1].copy()
        for idx in range(len(euler_angles)):
            ry3d = euler_angles[idx][1] # orientation in the camera coordinate system
            x3d, z3d = translations[idx][0], translations[idx][2]
            alpha = ry3d - math.atan2(-z3d, x3d) - 0.5 * math.pi
            #alpha = ry3d - math.atan2(x3d, z3d)# - 0.5 * math.pi
            while alpha > math.pi: alpha -= math.pi * 2
            while alpha < (-math.pi): alpha += math.pi * 2
            alphas[idx] = alpha
        return alphas
    
    def get_observation_angle_proj(self, euler_angles, kpts, K):
        """
        Convert orientation in camera coordinate into local coordinate system
        utilizing the projection of object on the image plane
        """ 
        f = K[0,0]
        cx = K[0,2]
        kpts_x = [kpts[i][0,0] for i in range(len(kpts))]
        alphas = euler_angles[:,1].copy()
        for idx in range(len(euler_angles)):
            ry3d = euler_angles[idx][1] # orientation in the camera coordinate system
            x3d, z3d = kpts_x[idx] - cx, f
            alpha = ry3d - math.atan2(-z3d, x3d) - 0.5 * math.pi
            #alpha = ry3d - math.atan2(x3d, z3d)# - 0.5 * math.pi
            while alpha > math.pi: alpha -= math.pi * 2
            while alpha < (-math.pi): alpha += math.pi * 2
            alphas[idx] = alpha
        return alphas

    def get_template(self, prediction, interp_coef=[0.332, 0.667]):
        """
        Construct a template 3D cuboid used for computing a rigid transformation.
        """ 
        parents = prediction[interp_dict['bbox12'][0] - 1]
        children = prediction[interp_dict['bbox12'][1] - 1]
        lines = parents - children
        lines = np.sqrt(np.sum(lines**2, axis=1))
        # averaged over the four parallel line segments
        h, l, w = np.sum(lines[:4])/4, np.sum(lines[4:8])/4, np.sum(lines[8:])/4
        x_corners = [l, l, l, l, 0, 0, 0, 0]
        y_corners = [0, h, 0, h, 0, h, 0, h]
        z_corners = [w, w, 0, 0, w, w, 0, 0]
        x_corners += - np.float32(l) / 2
        y_corners += - np.float32(h)
        #y_corners += - np.float32(h/2)
        z_corners += - np.float32(w) / 2
        corners_3d = np.array([x_corners, y_corners, z_corners])    
        if len(prediction) == 32:
            pidx, cidx = interp_dict['bbox12']
            parents, children = corners_3d[:, pidx - 1], corners_3d[:, cidx - 1]
            lines = children - parents
            new_joints = [(parents + interp_coef[i]*lines) for i in range(len(interp_coef))]
            corners_3d = np.hstack([corners_3d, np.hstack(new_joints)])    
        return corners_3d

    def kpts_to_euler(self, template, prediction):
        """
        Convert the predicted cuboid representation to euler angles.
        """    
        # estimate roll, pitch, yaw of the prediction by comparing with a 
        # reference bounding box
        # prediction and template of shape [3, N_points]
        R, T = ltr.compute_rigid_transform(template, prediction)
        # in the order of yaw, pitch and roll
        angles = Rotation.from_matrix(R).as_euler('yxz', degrees=False)
        # re-order in the order of x, y and z
        angles = angles[[1,0,2]]
        return angles, T

    def get_6d_rep(self, predictions, ax=None, color="black"):
        """
        Get the 6DoF representation of a 3D prediction.
        """    
        predictions = predictions.reshape(len(predictions), -1, 3)
        all_angles = []
        for instance_idx in range(len(predictions)):
            prediction = predictions[instance_idx]
            # templates are 3D boxes with no rotation
            # the prediction is estimated as the rotation between prediction and template
            template = self.get_template(prediction)
            instance_angle, instance_trans = self.kpts_to_euler(template, prediction.T)        
            all_angles.append(instance_angle.reshape(1, 3))
        angles = np.concatenate(all_angles)
        # the first point is the predicted point center
        translation = predictions[:, 0, :]    
        return angles, translation

    def gather_lifting_results(self,
                               record,
                               data,
                               prediction, 
                               target=None,
                               pose_vecs_gt=None,
                               intrinsics=None, 
                               refine=False, 
                               visualize=False,
                               template=None,
                               dist_coeffs=np.zeros((4,1)),
                               color='r',
                               get_str=False,
                               alpha_mode='trans'
                               ):
        """
        Convert network outputs to pose angles.
        """
        # prepare the prediction strings for submission
        # compute the roll, pitch and yaw angle of the predicted bounding box
        record['euler_angles'], record['translation'] = \
            self.get_6d_rep(record['kpts_3d_pred'])
        if alpha_mode == 'trans':
            record['alphas'] = self.get_observation_angle_trans(record['euler_angles'], 
                                                                record['translation']
                                                                )
        elif alpha_mode == 'proj':
            record['alphas'] = self.get_observation_angle_proj(record['euler_angles'],
                                                               record['kpts_2d_pred'],
                                                               record['K']
                                                               )        
        else:
             raise NotImplementedError   
        if get_str:
            record['pred_str'] = get_pred_str(record)      
        if visualize:
            record = vego.plot_3d_objects(prediction, 
                                          target, 
                                          pose_vecs_gt, 
                                          record, 
                                          color
                                          )
        return record

    def plot_one_image(self, 
                       img_path, 
                       record, 
                       visualize=False,
                       color_dict={'bbox_2d':'r',
                                   'bbox_3d':'r',
                                   'kpts':['rx', 'b']
                                   },
                       save_dict={'flag':False,
                                  'save_dir':None
                                  },
                       alpha_mode='trans'
                       ):
        """
        Post-process and plot the predictions from one image.
        """
        if visualize:
            # plot 2D predictions 
            vego.plot_2d_objects(img_path, record, color_dict)
        # plot 3d bounding boxes
        all_kpts_2d = np.concatenate(record['kpts_2d_pred'])
        all_kpts_3d_pred = record['kpts_3d_pred'].reshape(len(record['kpts_3d_pred']), -1)
        if 'kpts_3d_gt' in record:
            all_kpts_3d_gt = record['kpts_3d_gt']
            all_pose_vecs_gt = record['pose_vecs_gt']
        else:
            all_kpts_3d_gt = None
            all_pose_vecs_gt = None
        # refine and gather the prediction strings
        record = self.gather_lifting_results(record,
                                             all_kpts_2d,
                                             all_kpts_3d_pred, 
                                             all_kpts_3d_gt,
                                             all_pose_vecs_gt,
                                             color=color_dict['bbox_3d'],
                                             alpha_mode=alpha_mode,
                                             visualize=visualize,
                                             get_str=save_dict['flag']
                                             )

        # save KITTI-style prediction file in .txt format
        save_txt_file(img_path, record, save_dict)
        return record

    def post_process(self, 
                     records,
                     visualize=False, 
                     color_dict={'bbox_2d':'r',
                                 'kpts':['ro', 'b'],
                                 },
                     save_dict={'flag':False,
                                'save_dir':None
                                },
                     alpha_mode='trans'
                     ):
        """
        Save save and visualize them optionally.
        """   
        for img_path in records.keys():
            print("Processing {:s}".format(img_path))
            records[img_path] = self.plot_one_image(img_path, 
                                                    records[img_path],
                                                    visualize=visualize,
                                                    color_dict=color_dict,
                                                    save_dict=save_dict,
                                                    alpha_mode=alpha_mode
                                                    )      
        return records
    
    def new_img_dict(self):
        """
        An empty dictionary for image-level records.
        """
        img_dict = {'center':[], 
                    'scale':[], 
                    'rotation':[], 
                    'bbox_resize':[], # resized bounding box 
                    'kpts_2d_pred':[], 
                    'label':[], 
                    'score':[] 
                    }        
        return img_dict
    
    def get_keypoints(self,
                      instances, 
                      records, 
                      is_cuda=True
                      ):
        """
        Foward pass to obtain the screen coordinates.
        """
        if is_cuda:
            instances = instances.cuda()
        output = self.HC(instances)
        # local part coordinates
        local_coord = output[1].data.cpu().numpy()
        local_coord *= self.resolution[0]
        # transform local part coordinates to screen coordinates
        centers = [records[i]['center'] for i in range(len(records))]
        scales = [records[i]['scale'] for i in range(len(records))]
        rots = [records[i]['rotation'] for i in range(len(records))]    
        for instance_idx in range(len(local_coord)):
            trans_inv = get_affine_transform(centers[instance_idx],
                                             scales[instance_idx], 
                                             rots[instance_idx], 
                                             self.resolution, 
                                             inv=1
                                             )
            screen_coord = affine_transform_modified(local_coord[instance_idx], 
                                                     trans_inv
                                                     ) 
            records[instance_idx]['kpts'] = screen_coord
        # assemble a dictionary where each key corresponds to one image
        ret = {}
        for record in records:
            path = record['path']
            if path not in ret:
                ret[path] = self.new_img_dict()
            ret[path]['kpts_2d_pred'].append(record['kpts'].reshape(1, -1))
            ret[path]['center'].append(record['center'])
            ret[path]['scale'].append(record['scale'])
            ret[path]['bbox_resize'].append(record['bbox_resize'])
            ret[path]['label'].append(record['label'])
            ret[path]['score'].append(record['score'])
            ret[path]['rotation'].append(record['rotation'])
        return ret

    def lift_2d_to_3d(self, records, cuda=True):
        """
        Foward-pass of the lifter sub-model.
        """      
        for path in records.keys():
            data = np.concatenate(records[path]['kpts_2d_pred'], axis=0)
            data = nop.normalize_1d(data, self.LS['mean_in'], self.LS['std_in'])
            data = data.astype(np.float32)
            data = torch.from_numpy(data)
            if cuda:
                data = data.cuda()
            prediction = self.L(data)  
            prediction = nop.unnormalize_1d(prediction.data.cpu().numpy(),
                                            self.LS['mean_out'], 
                                            self.LS['std_out']
                                            )
            records[path]['kpts_3d_pred'] = prediction.reshape(len(prediction), -1, 3)
        return records
    
    def forward(self, annot_dict):
        """
        Process a batch of images.
        
        annot_dict is a Python dictionary storing the following keys: 
            path: list of image paths
            boxes: list of bounding boxes for each image
        """
        all_instances, all_records = self.crop_instances(annot_dict, 
                                                         resolution=self.resolution,
                                                         pth_trans=self.pth_trans,
                                                         xy_dict=self.xy_dict
                                                         )
        # all_records stores records for each instance
        records = self.get_keypoints(all_instances, all_records)
        # records stores records for each image
        records = self.lift_2d_to_3d(records)
        # write the annotation dictionary
        records = self.write_annot_dict(annot_dict, records)
        return records