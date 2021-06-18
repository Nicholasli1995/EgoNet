"""
Python class for KITTI dataset. 
Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import libs.dataset.basic.basic_classes as bc
import libs.visualization.points as vp
import libs.common.img_proc as lip
from libs.common.utils import make_dir
from libs.common.img_proc import get_affine_transform

import numpy as np
import torch
import cv2
import csv

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import copy

from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data.dataloader import default_collate
from os.path import join as pjoin
from os.path import sep as osep
from os.path import exists
from os import listdir 

# maximum number of inputs to the network depending on your GPU memory
MAX_INS_CNT = 140
#MAX_INS_CNT = 64
TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
}

FIELDNAMES = ['type', 
              'truncated', 
              'occluded', 
              'alpha', 
              'xmin', 
              'ymin', 
              'xmax', 
              'ymax', 
              'dh', 
              'dw',
              'dl', 
              'lx', 
              'ly', 
              'lz', 
              'ry']

# the format of prediction has one more field: confidence score
FIELDNAMES_P = FIELDNAMES.copy() + ['score']

# indices used for performing interpolation
# key->value: style->index arrays
interp_dict = {
    'bbox12':(np.array([1,3,5,7,# h direction
                        1,2,3,4,# l direction
                        1,2,5,6]), # w direction
              np.array([2,4,6,8,
                        5,6,7,8,
                        3,4,7,8])
              ),
    'bbox12l':(np.array([1,2,3,4,]), # w direction
              np.array([5,6,7,8])
              ),
    'bbox12h':(np.array([1,3,5,7]), # w direction
              np.array([2,4,6,8])
              ),
    'bbox12w':(np.array([1,2,5,6]), # w direction
              np.array([3,4,7,8])
              ),    
    }

# indices used for computing the cross ratio
cr_indices_dict = {
    'bbox12':np.array([[ 1,  9, 21,  2],
                       [ 3, 10, 22,  4],
                       [ 5, 11, 23,  6],
                       [ 7, 12, 24,  8],
                       [ 1, 13, 25,  5],
                       [ 2, 14, 26,  6],
                       [ 3, 15, 27,  7],
                       [ 4, 16, 28,  8],
                       [ 1, 17, 29,  3],
                       [ 2, 18, 30,  4],
                       [ 5, 19, 31,  7],
                       [ 6, 20, 32,  8]]
                      )
    }

def get_cr_indices():
    # helper function to define the indices used in computing the cross-ratio
    num_base_pts = 9
    num_lines = 12
    parents, children = interp_dict['bbox12']
    cr_indices = []
    for line_idx in range(num_lines):
        parent_idx = parents[line_idx] # first point
        child_idx = children[line_idx] # last point
        second_point_idx = num_base_pts + line_idx
        third_point_idx = num_base_pts + num_lines + line_idx
        cr_indices.append(np.array([parent_idx, 
                                   second_point_idx, 
                                   third_point_idx,
                                   child_idx]
                                  ).reshape(1,4)
                         )
    cr_indices = np.vstack(cr_indices)
    return cr_indices

class KITTI(bc.SupervisedDataset):
    def __init__(self, cfgs, split, logger, scale=1.0, use_stereo=False):
        super().__init__(cfgs, split, logger)
        self.logger = logger
        self.logger.info("Initializing KITTI {:s} set, Please wait...".format(split))
        self.exp_type = cfgs['exp_type'] # exp_type: experiment type 
        self._data_dir = cfgs['dataset']['root'] # root directory
        self._classes = cfgs['dataset']['detect_classes'] # used object classes
        self._get_data_parameters(cfgs) # initialize hyper-parameters
        self._set_paths() # initialize paths
        self._inference_mode = False 
        self.car_sizes = [] # dimension of cars
        if use_stereo:
            raise NotImplementedError
        self._load_image_list()
        if self.split in ['train', 'valid', 'trainvalid'] and \
            self.exp_type in ['instanceto2d', 'baselinealpha', 'baselinetheta']:
            self._prepare_key_points(cfgs)
            # save cropped car instances for debugging
            # cropped_path = pjoin(self._data_config['cropped_dir'], self.kpts_style,
            #                      self.split)
            # if not exists(cropped_path) and cfgs['dataset']['pre-process']:
            #     self._save_cropped_instances()            
        # prepare data used for future loading
        self.generate_pairs()
        # self.visualize()
        if self.split in ['train', 'trainvalid'] and self.exp_type in ['2dto3d']:
            # 2dto3d means the data is used a the sub-network that predicts 3D 
            # cuboid based on 2D screen coordinates 
            self.normalize() # normalization for 2d-to-3d pose estimation
        # use unlabeled images for weak self-supervision
        if 'ss' in cfgs and cfgs['ss']['flag']:
            self.use_ss = True
            self.ss_settings = cfgs['ss']
            self._initialize_unlabeled_data(cfgs)
        self.logger.info("Initialization finished for KITTI {:s} set".format(split))
        # self.show_statistics()
        # debugging code if you need
        # test = self[10]
        # test = self.extract_ss_sample(1)
    
    def _get_image_path_list(self):
        """
        Prepare list of image paths for the used split.
        """
        assert 'image_name_list' in self._data_config
        image_path_list = []
        for name in self._data_config['image_name_list']:
            img_path = pjoin(self._data_config['image_dir'], name)
            image_path_list.append(img_path)
        self._data_config['image_path_list'] = image_path_list        
        return
    
    def _initialize_unlabeled_data(self, cfgs):
        """
        Initialize unlabeled data for self-supervision experiment.
        """
        self.ss_record = np.load(cfgs['ss']['record_path'], allow_pickle=True).item()
        self.logger.info('Found prepared self-supervision record at: ' + cfgs['ss']['record_path'])
        return
    
    def _load_image_list(self):
        """
        Prepare list of image names for the used split.
        """
        path = self._data_config[self.split + '_list']       
        with open(path, "r") as f:
            image_name_list = f.read().splitlines()
        for idx, line in enumerate(image_name_list):
            base_name = line.replace("\n", "")
            image_name = base_name + ".png"
            image_name_list[idx] = image_name
        self._data_config['image_name_list'] = image_name_list
        self._get_image_path_list()
        return
    
    def _check_precomputed_file(self, path, name):
        """
        Check if a pre-computed numpy file exist or not.
        """
        if exists(path):
            self.logger.info('Found prepared {0:s} at {1:s}'.format(name, path))
            value = np.load(path, allow_pickle=True).item()
            setattr(self, name, value)
            return True
        else:
            return False
        
    def _save_precomputed_file(self, data_dic, pre_computed_path, name):
        """
        Save a pre-computed numpy file.
        """
        setattr(self, name, data_dic)
        make_dir(pre_computed_path)
        np.save(pre_computed_path, data_dic)
        self.logger.info('Save prepared {0:s} at {1:s}'.format(name, pre_computed_path))        
        return
    
    def _prepare_key_points_custom(self, style, interp_params, vis_thresh=0.25):
        # Define the 2d key-points as the projected 3D points on the image plane.
        assert 'keypoint_dir' in self._data_config
        kpt_dir = self._data_config['keypoint_dir']
        if interp_params['flag']:
            style += str(interp_params['coef'])
        pre_computed_path_kpts = pjoin(kpt_dir, '{0:s}_{1:s}_{2:s}.npy'.format(style, self.split, str(self._classes)))
        pre_computed_path_ids = pjoin(kpt_dir, '{0:s}_{1:s}_{2:s}_ids.npy'.format(style, self.split, str(self._classes)))        
        pre_computed_path_rots = pjoin(kpt_dir, '{0:s}_{1:s}_{2:s}_rots.npy'.format(style, self.split, str(self._classes)))   
        if self._check_precomputed_file(pre_computed_path_kpts, 'keypoints'):
            pass
        if self._check_precomputed_file(pre_computed_path_ids, 'instance_ids'):
            pass     
        if self._check_precomputed_file(pre_computed_path_rots, 'rotations'):
            return    
        path_list = self._data_config['image_path_list']
        data_dic_kpts = {}
        data_dic_ids = {}
        data_dic_rots = {}
        for path in path_list:
            image_name = path.split(osep)[-1]
            # instances that lie out of the image plane will be discarded 
            list_2d, _, list_id, _, list_rots = self.get_2d_3d_pair(path, 
                                                                    style=style, 
                                                                    augment=False,
                                                                    add_visibility=True,
                                                                    filter_outlier=True,
                                                                    add_rotation=True
                                                                    )  
            if len(list_2d) == 0:
                continue
            for idx, kpts in enumerate(list_2d):
                list_2d[idx] = kpts.reshape(1, -1, 3)
            data_dic_kpts[image_name] = np.concatenate(list_2d, axis=0)
            data_dic_ids[image_name] = list_id
            data_dic_rots[image_name] = np.concatenate(list_rots, axis=0)
        self._save_precomputed_file(data_dic_kpts, pre_computed_path_kpts, 'keypoints')
        self._save_precomputed_file(data_dic_ids, pre_computed_path_ids, 'instance_ids')  
        self._save_precomputed_file(data_dic_rots, pre_computed_path_rots, 'rotations') 
        return
    
    def _prepare_key_points(self, cfgs):
        self.kpts_style = cfgs['dataset']['2d_kpt_style']
        self._prepare_key_points_custom(self.kpts_style, cfgs['dataset']['interpolate'])
        self.enlarge_factor = 1.1
        return
    
    def _save_cropped_instances(self):
        # DEPRECATED
        """ crop and save car instance images with given 2d key-points
        """        
        assert hasattr(self, 'keypoints')
        all_save_paths = []
        all_keypoints = []
        all_bbox = []
        target_ar = self.hm_para['target_ar']
        for image_name in self.keypoints.keys():
            image_path = pjoin(self._data_config['image_dir'], image_name)
            save_dir = pjoin(self._data_config['cropped_dir'], self.kpts_style,
                             self.split, image_name[:-4])
            keypoints = self.keypoints[image_name]
            new_paths, new_keypoints, bboxes = lip.save_cropped_patches(image_path, 
                                                                keypoints, 
                                                                save_dir, 
                                                                enlarge=self.enlarge_factor,
                                                                target_ar=target_ar)
            all_save_paths += new_paths
            all_keypoints.append(new_keypoints)
            all_bbox += bboxes
        annot_save_name = pjoin(self._data_config['cropped_dir'], 
                                self.kpts_style, self.split, 'annot.npy')
        np.save(annot_save_name, {'paths': all_save_paths,
                                  'kpts': np.concatenate(all_keypoints, axis=0),
                                  'global_box': all_bbox
                                  })
        return
    
    def _prepare_2d_pose_annot(self, threshold=4):
        all_paths = []
        all_boxes = []
        all_rotations = []
        all_keypoints = []
        all_keypoints_raw = []
        for image_name in self.keypoints.keys():
            image_path = pjoin(self._data_config['image_dir'], image_name)
            # raw keypoints using camera projection
            keypoints = self.keypoints[image_name]
            rotations = self.rotations[image_name]
            boxes_img = []
            rots_img = []
            visible_kpts_img = []
            for i in range(len(keypoints)):
                # Note here severely-occluded instances are ignored in the trainign data
                visible_cnt = np.sum(keypoints[i][:, 2])
                if visible_cnt < threshold:
                    continue
                else:
                    # now set all keypoints as visible
                    tempt_kpts = keypoints[i][:,:2]
                    visible_kpts_img.append(np.expand_dims(tempt_kpts, 0))
                center, crop_size, new_keypoints, vis_rate = lip.kpts2cs(tempt_kpts, enlarge=self.enlarge_factor)
                bbox_instance = np.array((list(map(int, lip.cs2bbox(center, crop_size)))))
                boxes_img.append(bbox_instance.reshape(1,4))
                rots_img.append(rotations[i].reshape(1,2))
            if len(boxes_img) == 0:
                continue
            all_paths.append(image_path)            
            all_boxes.append(np.concatenate(boxes_img))
            all_rotations.append(np.concatenate(rots_img))
            all_keypoints.append(np.concatenate(visible_kpts_img))
            all_keypoints_raw.append(keypoints)
        return {'paths':all_paths, 
                'boxes':all_boxes, 
                'rots':all_rotations,
                'kpts':all_keypoints,
                'raw_kpts':all_keypoints_raw
                }
    
    def _prepare_detection_records(self, save=False, threshold = 0.1):
        # DEPRECATED UNTIL FURTHER UPDATE
        raise ValueError

    def gather_annotations(self, 
                           threshold=0.1, 
                           use_raw_bbox=False, 
                           add_gt=True,
                           filter_outlier=False
                           ):
        path_list = self._data_config['image_path_list'] 
        record_dict = {}
        for img_path in path_list:
            image_name = img_path.split(osep)[-1]
            if self.split != 'test':
                # default: use gt label and calibration
                label_path = pjoin(self._data_config['label_dir'], 
                                   image_name[:-4] + '.txt'
                                   )
                self.read_single_file(image_name, 
                                      record_dict, 
                                      label_path=label_path,
                                      fieldnames=FIELDNAMES,
                                      add_gt=add_gt,
                                      use_raw_bbox=use_raw_bbox,
                                      filter_outlier=filter_outlier
                                      )
            else:
                record_dict[image_name] = {}
        self.annot_dict = record_dict
        return     
    
    def read_single_file(self, 
                         image_name, 
                         record_dict, 
                         label_path=None,
                         calib_path=None,
                         threshold=0.1,
                         fieldnames=FIELDNAMES_P,
                         add_gt=False,
                         use_raw_bbox=True,
                         filter_outlier=False,
                         bbox_only=False
                         ):
        style = self._data_config['3d_kpt_sample_style']
        image_path = pjoin(self._data_config['image_dir'], image_name)
        if label_path is None:
            # default is ground truth annotation
            label_path = pjoin(self._data_config['label_dir'], image_name[:-3] + 'txt')
        if calib_path is None:
            calib_path = pjoin(self._data_config['calib_dir'], image_name[:-3] + 'txt')
        list_2d, list_3d, list_id, pv, raw_bboxes = self.get_2d_3d_pair(image_path,
                                                                        label_path=label_path,
                                                                        calib_path=calib_path,
                                                                        style=style,
                                                                        augment=False,
                                                                        add_raw_bbox=True,
                                                                        bbox_only=bbox_only,
                                                                        filter_outlier=filter_outlier,
                                                                        fieldnames=fieldnames # also load the confidence score
                                                                        )  
        if len(raw_bboxes) == 0:
            return False        
        if image_name not in record_dict:
            record_dict[image_name] = {}
        raw_annot, P = self.load_annotations(label_path, calib_path, fieldnames=fieldnames)
        # use different (slightly) intrinsic parameters for different images
        K = P[:, :3]  
        if len(list_2d) != 0:
            for idx, kpts in enumerate(list_2d):
                list_2d[idx] = kpts.reshape(1, -1, 3)
                list_3d[idx] = list_3d[idx].reshape(1, -1, 3)
            all_keypoints_2d = np.concatenate(list_2d, axis=0)
            all_keypoints_3d = np.concatenate(list_3d, axis=0)                       
            # compute 2D bounding box based on the projected 3D boxes
            bboxes_kpt = []
            for idx, keypoints in enumerate(all_keypoints_2d):
                # relatively tight bounding box: use enlarge = 1.0
                # delete invisible instances
                center, crop_size, _, _ = lip.kpts2cs(keypoints[:,:2],
                                                      enlarge=1.01)
                bbox = np.array(lip.cs2bbox(center, crop_size))             
                bboxes_kpt.append(np.array(bbox).reshape(1, 4))
            record_dict[image_name]['kpts_3d'] = all_keypoints_3d
            if add_gt:
                # special key name representing ground truth
                record_dict[image_name]['kpts'] = all_keypoints_2d
                record_dict[image_name]['kpts_3d_gt'] = all_keypoints_3d
        if use_raw_bbox:
            bboxes = np.vstack(raw_bboxes)
        elif len(bboxes_kpt) != 0:
            bboxes = np.vstack(bboxes_kpt)
            
        record_dict[image_name]['bbox_2d'] = bboxes
        record_dict[image_name]['raw_txt_format'] = raw_annot
        record_dict[image_name]['K'] = K
        # add some key-value pairs as ground truth annotation
        if add_gt:         
            pvs = np.vstack(pv) if len(pv) != 0 else []
            tempt_dic = {'boxes': bboxes,
                         'pose_vecs_gt':pvs
                         }
            record_dict[image_name] = {**record_dict[image_name], **tempt_dic}              
        return True
    
    def read_predictions(self, path):
        """
        Read the prediction files in the same format as the ground truth.
        """
        self.logger.info("Reading predictions from {:s}".format(path))
        file_list = listdir(path)  
        record_dict = {}
        use_raw_bbox = True if self.split == 'test' else False
        for file_name in file_list:
            if not file_name.endswith(".txt"):
                continue
            image_name = file_name[:-4] + ".png"
            label_path = pjoin(path, file_name)            
            self.read_single_file(image_name, 
                                  record_dict, 
                                  label_path=label_path,
                                  use_raw_bbox=use_raw_bbox
                                  )
        self.logger.info("Reading predictions finished.")
        return record_dict
    
    def _get_data_parameters(self, cfgs):
        """
        Initialize dataset-relevant parameters.
        """
        self._data_config = {}
        self._data_config['image_size_raw'] = NotImplemented
        if self.exp_type in ['2dto3d', 'inference', 'finetune']:
            # parameters relevant to input/output representation
            for key in ['3d_kpt_sample_style', 'lft_in_rep', 'lft_out_rep']:
                self._data_config[key] = cfgs['dataset'][key] 
        if self.exp_type in ['2dto3d']:  
            # parameters relevant to data augmentation              
            for key in ['lft_aug','lft_aug_times']:
                self._data_config[key] = cfgs['training_settings'][key]
        # parameters relevant to cuboid interpolation
        self.interp_params = cfgs['dataset']['interpolate']
        # parameters relevant to heatmap regression model and image data augmentation
        if 'heatmapModel' in cfgs:
            hm = cfgs['heatmapModel']
            self.hm_para = {'reference': 'bbox',
                            'resize': True,
                            'add_xy': hm['add_xy'],
                            'jitter_bbox': hm['jitter_bbox'] and self.split=='train',
                            'jitter_params': hm['jitter_params'],
                            'input_size': np.array([hm['input_size'][1],
                                             hm['input_size'][0]]),
                            'heatmap_size': np.array([hm['heatmap_size'][1],
                                               hm['heatmap_size'][0]]),
                            'target_ar': hm['heatmap_size'][1]/hm['heatmap_size'][0],
                            'augment': hm['augment_input'],
                            'sf': cfgs['dataset']['scaling_factor'],
                            'rf': cfgs['dataset']['rotation_factor'],
                            'num_joints': hm['num_joints'],
                            'sigma': hm['sigma'],
                            'target_type': hm['target_type'],
                            'use_different_joints_weight': 
                                hm['use_different_joints_weight']                               
                              }
            self.num_joints = hm['num_joints']
        # parameters relevant to PyTorch image transformation operations
        if 'pth_transform' in cfgs['dataset']:
            pth_transform = cfgs['dataset']['pth_transform']
            normalize = transforms.Normalize(
                mean=pth_transform['mean'], 
                std=pth_transform['std']
                )
            transform_list = [transforms.ToTensor(), normalize]
            if self.exp_type == 'detect2D' and self.split == 'train':
                transform_list.append(transforms.RandomHorizontalFlip(0.5))
            self.pth_trans = transforms.Compose(transform_list)           

    def _set_paths(self):
        """
        Initialize relevant directories.
        """
        ROOT = self.root
        split = self.split
        # validation set is a sub-set of the official training split
        # train/val/test: 3712/3769/7518
        split = 'train' if self.split == 'valid' else split
        split += 'ing'
        self._data_config['image_dir'] = pjoin(ROOT, split, 'image_2')
        self._data_config['cropped_dir'] = pjoin(ROOT, split, 'cropped')
        self._data_config['drawn_dir'] = pjoin(ROOT, split, 'drawn')
        self._data_config['label_dir'] = pjoin(ROOT, split, 'label_2')
        self._data_config['calib_dir'] = pjoin(ROOT, split, 'calib')
        self._data_config['keypoint_dir'] = pjoin(ROOT, split, 'keypoints')
        self._data_config['stats_dir'] = pjoin(ROOT, 'instance_stats.npy')
        # list of images for each sub-set
        self._data_config['train_list'] = pjoin(ROOT, 'training/ImageSets/train.txt')
        self._data_config['valid_list'] = pjoin(ROOT, 'training/ImageSets/val.txt')
        self._data_config['test_list'] = pjoin(ROOT, 'testing/ImageSets/test.txt')
        self._data_config['trainvalid_list'] = pjoin(ROOT, 'training/ImageSets/trainval.txt')        
        return
    
    def project_3d_to_2d(self, points, K):
        # get 2D projections 
        projected = K @ points.T
        projected[:2, :] /= projected[2, :]
        return projected
    
    def render_car(self, ax, K, obj_class, rot_y, locs, dimension, shift):
        cam_cord = []
        self.get_cam_cord(cam_cord, shift, rot_y, dimension, locs)
        # get 2D projections 
        projected = self.project_3d_to_2d(cam_cord[0], K)
        ax.plot(projected[0, :], projected[1, :], 'ro')
        vp.plot_3d_bbox(ax, projected[:2, 1:].T)
        return
    
    def show_statistics(self):
        # DEPRECATED
        path = self._data_config['stats_dir']       
        if self._check_precomputed_file(path, 'instance_stats') or self.split != 'train':
            return
        self.instance_statistics = {}
        if hasattr(self, 'car_sizes') and len(self.car_sizes) != 0:
            all_sizes = np.concatenate(self.car_sizes)
            fig, axes = plt.subplots(3,1)
            names = ['x', 'y', 'z']
            for axe_id in range(3):
                axes[axe_id].hist(all_sizes[:, axe_id])
                axes[axe_id].set_xlabel('Car size in {:s} direction'.format(names[axe_id]))
                axes[axe_id].set_ylabel('Counts')
            mean_size = all_sizes.mean(axis=0)
            std_size = all_sizes.std(axis=0)
            self.instance_statistics['size'] = {'mean':mean_size,
                                                'std': std_size}
            # prepare a reference 3D bounding box for inference purpose
            xmax, xmin = mean_size[0], -mean_size[0]
            ymax, ymin = mean_size[1], -mean_size[1]
            zmax, zmin = mean_size[2], -mean_size[2]
            bbox = np.array([[xmax, ymin, zmax],
                             [xmax, ymax, zmax],
                             [xmax, ymin, zmin],
                             [xmax, ymax, zmin],
                             [xmin, ymin, zmax],
                             [xmin, ymax, zmax],
                             [xmin, ymin, zmin],
                             [xmin, ymax, zmin]])
            bbox = np.vstack([np.array([[0., 0., 0.]]), bbox])            
            self.instance_statistics['ref_box3d'] = bbox
        self._save_precomputed_file(self.instance_statistics, path, 'instance_stats')            
        return
    
    def augment_pose_vector(self, 
                            locs,
                            rot_y,
                            obj_class,
                            dimension,
                            augment,
                            augment_times,
                            std_rot = np.array([15., 50., 15.])*np.pi/180.,
                            std_trans = np.array([0.2, 0.01, 0.2]),
                            ):
        """
        std_rot: standard deviation of rotation around x, y and z axis
        std_trans: standard deviation of translation along x, y and z axis
        """
        aug_ids, aug_pose_vecs = [], []
        aug_ids.append((obj_class, dimension))
        # KITTI only annotates rotation around y-axis
        pose_vec = np.concatenate([locs, np.array([0., rot_y, 0.])]).reshape(1, 6)
        aug_pose_vecs.append(pose_vec)
        if not augment:
            return aug_ids, aug_pose_vecs
        rots_random = np.random.randn(augment_times, 3) * std_rot.reshape(1, 3)
        # y-axis
        rots_random[:, 1] += rot_y
        trans_random = 1 + np.random.randn(augment_times, 3) * std_trans.reshape(1, 3)
        trans_random *= locs.reshape(1, 3)
        for i in range(augment_times):
            # augment 6DoF pose
            aug_ids.append((obj_class, dimension))
            pose_vec = np.concatenate([trans_random[i], rots_random[i]]).reshape(1, 6)
            aug_pose_vecs.append(pose_vec)
        return aug_ids, aug_pose_vecs
    
    def get_representation(self, p2d, p3d, in_rep, out_rep):
        # get input-output representations based on 3d point cloud and its 
        # 2d projection
        # input representation
        if len(p2d) > 0:
            num_kpts = len(p2d[0])
        if in_rep == 'coordinates2d':
            input_list = [points.reshape(1, num_kpts, -1) for points in p2d]
        elif in_rep == 'coordinates2d+area' and self._data_config['3d_kpt_sample_style'] == 'bbox9':
            # indices: [corner, neighbour1, neighbour2]
            indices = self.area_indices
            input_list = [vp.get_area(points, indices, True) for points in p2d]
        else:
            raise NotImplementedError('Undefined input representation.')
        # output representation
        if out_rep == 'R3d+T':
            # R3D stands for relative 3D shape, T stands for translation
            # center the camera coordinates to remove depth
            output_list = []
            for i in range(len(p3d)):
                # format: the root should be pre-computed as the first 3d point 
                root = p3d[i][[0], :]
                relative_shape = p3d[i][1:, :] - root
                output = np.concatenate([root, relative_shape], axis=0)
                output_list.append(output.reshape(1, -1)) 
        elif out_rep == 'R3d': # relative 3D shape
            output_list = []
            # save a copy of the 3D object roots
            if not hasattr(self, 'root_list'):
                self.root_list = []
            for i in range(len(p3d)):
                # format: the root should be pre-computed as the first 3d point 
                root = p3d[i][[0], :]
                self.root_list.append(root)
                relative_shape = p3d[i][1:, :] - root
                output_list.append(relative_shape.reshape(1, -1)) 
        else:
            raise NotImplementedError('undefined output representation.')
        return input_list, output_list
    
    def get_input_output_size(self):
        """
        get the input-output size for 2d-to-3d lifting
        """
        num_joints = self.num_joints
        if self._data_config['lft_in_rep'] == 'coordinates2d':
             input_size = num_joints*2
        else:
             raise NotImplementedError
        if self._data_config['lft_out_rep'] in ['R3d+T']:
             output_size = num_joints*3
        elif self._data_config['lft_out_rep'] in ['R3d']:
             output_size = (num_joints - 1) * 3             
        else:
             raise NotImplementedError        
        return input_size, output_size
    
    def interpolate(self, 
                    bbox_3d, 
                    style, 
                    interp_coef=[0.5], 
                    dimension=None, 
                    strings=['l','h','w']
                    ):
        # interpolate 3d points on a 3D bounding box with specified style
        if dimension is not None:
            # size-encoded representation
            l = dimension[0]
            if l < 3.5:
                style += 'l'
            elif l < 4.5:
                style += 'h'
            else:
                style += 'w'       
        pidx, cidx = interp_dict[style]
        parents, children = bbox_3d[:, pidx], bbox_3d[:, cidx]
        lines = children - parents
        new_joints = [(parents + interp_coef[i]*lines) for i in range(len(interp_coef))]
        return np.hstack([bbox_3d, np.hstack(new_joints)])
    
    def construct_box_3d(self, l, h, w, interp_params):
        # add radom noise to length
        # l *= (1 + np.random.randn()*0.1)
        x_corners = [0.5*l, l, l, l, l, 0, 0, 0, 0]
        y_corners = [0.5*h, 0, h, 0, h, 0, h, 0, h]
        z_corners = [0.5*w, w, w, 0, 0, w, w, 0, 0]
        x_corners += - np.float32(l) / 2
        y_corners += - np.float32(h)
        z_corners += - np.float32(w) / 2
        corners_3d = np.array([x_corners, y_corners, z_corners])     
        if interp_params['flag']:
            corners_3d = self.interpolate(corners_3d, 
                                          interp_params['style'],
                                          interp_params['coef'],
                                          #dimension=np.array([l,h,w]) # dimension aware
                                          )
        return corners_3d
    
    def get_cam_cord(self, cam_cord, shift, ids, pose_vecs, rot_xz=False):
        # does not augment the dimension for now
        dims = ids[0][1]
        l, h, w = dims[0], dims[1], dims[2]
        corners_3d_fixed = self.construct_box_3d(l, h, w, self.interp_params)
        for pose_vec in pose_vecs:
            # translation
            locs = pose_vec[0, :3]
            rots = pose_vec[0, 3:]
            x, y, z = locs[0], locs[1], locs[2] # bottom center of the labeled 3D box
            rx, ry, rz = rots[0], rots[1], rots[2]
            # TEMPORAL TESTING: rotation and translation augmentation
            # This purturbation turns out to work well for rotation estimation
#            x *= (1 + np.random.randn()*0.1)
#            y *= (1 + np.random.randn()*0.05)
#            z *= (1 + np.random.randn()*0.1)
            if self.split == 'train' and self.exp_type == '2dto3d' and not self._inference_mode:
                ry += np.random.randn()*np.pi # random perturbation
            # END TEMPORAL TESTING
            rot_maty = np.array([[np.cos(ry), 0, np.sin(ry)],
                                [0, 1, 0],
                                [-np.sin(ry), 0, np.cos(ry)]])
            if rot_xz:
                # rotation. Only yaw angle is considered in KITTI dataset
                rot_matx = np.array([[1, 0, 0],
                                    [0, np.cos(rx), -np.sin(rx)],
                                    [0, np.sin(rx), np.cos(rx)]])        
    
                rot_matz = np.array([[np.cos(rz), -np.sin(rz), 0],
                                    [np.sin(rz), np.cos(rz), 0],
                                    [0, 0, 1]])        
                # TODO: correct here
                rot_mat = rot_matz @ rot_maty @ rot_matx     
            else:
                rot_mat = rot_maty
            corners_3d = np.matmul(rot_mat, corners_3d_fixed)
            # translation
            corners_3d += np.array([x, y, z]).reshape([3, 1])
            camera_coordinates = corners_3d + shift
            cam_cord.append(camera_coordinates.T)
        return 
    
    def csv_read_annot(self, file_path, fieldnames):
        """
        Read instance attributes in the KITTI format. Instances not in the 
        selected class will be ignored. A list of python dictionary is returned
        where each dictionary represents one instsance.
        """        
        annotations = []
        with open(file_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
            for line, row in enumerate(reader):
                if row["type"] in self._classes:
                    annot_dict = {
                        "class": row["type"],
                        "label": TYPE_ID_CONVERSION[row["type"]],
                        "truncation": float(row["truncated"]),
                        "occlusion": float(row["occluded"]),
                        "alpha": float(row["alpha"]),
                        "dimensions": [float(row['dl']), 
                                       float(row['dh']), 
                                       float(row['dw'])
                                       ],
                        "locations": [float(row['lx']), 
                                      float(row['ly']), 
                                      float(row['lz'])
                                      ],
                        "rot_y": float(row["ry"]),
                        "bbox": [float(row["xmin"]),
                                 float(row["ymin"]),
                                 float(row["xmax"]),
                                 float(row["ymax"])
                                 ]
                    }
                    if "score" in fieldnames:
                        annot_dict["score"] = float(row["score"])
                    annotations.append(annot_dict)        
        return annotations
    
    def csv_read_calib(self, file_path):
        """
        Read camera projection matrix in the KITTI format.
        """  
        # get camera intrinsic matrix K
        with open(file_path, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for line, row in enumerate(reader):
                if row[0] == 'P2:':
                    P = row[1:]
                    P = [float(i) for i in P]
                    P = np.array(P, dtype=np.float32).reshape(3, 4)
                    break        
        return P
    
    def load_annotations(self, label_path, calib_path, fieldnames=FIELDNAMES):        
        if self.split in ['train', 'valid', 'trainvalid', 'test']:
            annotations = self.csv_read_annot(label_path, fieldnames)

        # get camera intrinsic matrix K
        P = self.csv_read_calib(calib_path)

        return annotations, P
    
    def add_visibility(self, joints, img_width=1242, img_height=375):
        assert joints.shape[1] == 2
        visibility = np.ones((len(joints), 1))
        # predicate from upper left corner
        predicate1 = joints - np.array([[0., 0.]])
        predicate1 = (predicate1 > 0.).prod(axis=1)
        # predicate from lower right corner
        predicate2 = joints - np.array([[img_width, img_height]])
        predicate2 = (predicate2 < 0.).prod(axis=1)
        visibility[:, 0] *= predicate1*predicate2      
        return np.hstack([joints, visibility])
    
    def get_inlier_indices(self, p_2d, threshold=0.3):
        indices = []
        num_joints = p_2d[0].shape[0]
        for idx, kpts in enumerate(p_2d):
            if p_2d[idx][:, 2].sum() / num_joints >= threshold:
                indices.append(idx)        
        return indices
    
    def filter_outlier(self, p_2d, p_3d, threshold=0.3):
        p_2d_filtered, p_3d_filtered, indices = [], [], []
        num_joints = p_2d[0].shape[0]
        for idx, kpts in enumerate(p_2d):
            if p_2d[idx][:, 2].sum() / num_joints >= threshold:
                p_2d_filtered.append(p_2d[idx])
                p_3d_filtered.append(p_3d[idx])
                indices.append(idx)
        return p_2d_filtered, p_3d_filtered
    
    def get_img_size(self, path):
        """
        Get the resolution of an image without loading it.
        """
        with Image.open(path) as image:
            size = image.size 
        return size
    
    def get_2d_3d_pair(self, 
                       image_path, 
                       label_path=None,
                       calib_path=None,
                       style='null',
                       in_rep = 'coordinates2d',
                       out_rep = 'R3d+T',
                       augment=False, 
                       augment_times=1,
                       add_visibility=True,
                       add_raw_bbox=False, # add original bbox annotation from KITTI
                       add_rotation=False, # add orientation angles
                       bbox_only=False, # only returns raw bounding box
                       filter_outlier=True,
                       fieldnames=FIELDNAMES
                       ):
        image_name = image_path.split(osep)[-1]
        if label_path is None:
            # default is ground truth annotation
            label_path = pjoin(self._data_config['label_dir'], image_name[:-3] + 'txt')
        if calib_path is None:
            calib_path = pjoin(self._data_config['calib_dir'], image_name[:-3] + 'txt')
        anns, P = self.load_annotations(label_path, calib_path, fieldnames=fieldnames)
        # The intrinsics may vary slightly for different images
        # Yet one may convert them to a fixed one by applying a homography
        K = P[:, :3]
        # Debug: use pre-defined intrinsic parameters
        # K = np.array([[707.0493,   0.    , 604.0814],
        #               [  0.    , 707.0493, 180.5066],
        #               [  0.    ,   0.    ,   1.    ]], dtype=np.float32)
        shift = np.linalg.inv(K) @ P[:, 3].reshape(3,1)      
        # P containes intrinsics and extrinsics, we factorize P to K[I|K^-1t] 
        # and use extrinsics to compute the camera coordinate
        # here the extrinsics represent the shift between current camera to
        # the reference grayscale camera        
        # For more calibration details, refer to "Vision meets Robotics: The KITTI Dataset"
        camera_coordinates = []
        pose_vecs = []
        # id includes the class and size of the object
        ids = []
        if add_raw_bbox:
            bboxes = []
        if add_rotation:
            rotations = []
        for i, a in enumerate(anns):
            a = a.copy()
            obj_class = a["label"]
            dimension = a["dimensions"]
            locs = np.array(a["locations"])
            rot_y = np.array(a["rot_y"])
            if add_raw_bbox:
                bboxes.append(np.array(a["bbox"]).reshape(1,4))
            if add_rotation:
                rotations.append(np.array([a["alpha"], a["rot_y"]]).reshape(1,2))
            # apply data augmentation to represent a larger variation of
            # 3D pose and translation 
            if bbox_only:
                continue
            aug_ids, aug_pose_vecs = self.augment_pose_vector(locs,
                                                              rot_y,
                                                              obj_class,
                                                              dimension,
                                                              augment,
                                                              augment_times
                                                              )
            self.get_cam_cord(camera_coordinates, 
                              shift, 
                              aug_ids, 
                              aug_pose_vecs
                              )                    
            ids += aug_ids
            pose_vecs += aug_pose_vecs
        num_instances = len(camera_coordinates)
        # get 2D projections 
        if len(camera_coordinates) != 0:
            camera_coordinates = np.vstack(camera_coordinates)
            projected = self.project_3d_to_2d(camera_coordinates, K)[:2, :].T
            # target is camera coordinates
            p_2d = np.split(projected, num_instances, axis=0) 
            p_3d = np.split(camera_coordinates, num_instances, axis=0) 
            # set visibility to 0 if the projected keypoints lie out of the image plane
            if add_visibility:
                width, height = self.get_img_size(image_path)
                for idx, joints in enumerate(p_2d):
                    p_2d[idx] = self.add_visibility(joints, width, height)
            # filter out the instances that lie outside of the image
            if filter_outlier:
                indices = self.get_inlier_indices(p_2d)
                p_2d = [p_2d[idx] for idx in indices]
                p_3d = [p_3d[idx] for idx in indices]
                # p_2d, p_3d = self.filter_outlier(p_2d, p_3d)
            if filter_outlier and add_raw_bbox:
                bboxes = [bboxes[idx] for idx in indices]
            if filter_outlier and add_rotation:
                rotations = [rotations[idx] for idx in indices]            
            list_2d, list_3d = self.get_representation(p_2d, p_3d, in_rep, out_rep)

        else:
            list_2d, list_3d, ids, pose_vecs = [], [], [], []
        ret = list_2d, list_3d, ids, pose_vecs
        if add_raw_bbox:
            ret = ret + (bboxes, )
        if add_rotation:
            ret = ret + (rotations, )
        return ret            
    
    def show_annot(self, image_path, label_file=None, calib_file=None, save_dir=None):
        """
        Show the annotations of an image.
        """      
        image_name = image_path.split(osep)[-1]
        if label_file is None:
            label_file = pjoin(self._data_config['label_dir'], image_name[:-3] + 'txt')
        if calib_file is None:
            calib_file = pjoin(self._data_config['calib_dir'], image_name[:-3] + 'txt')
        anns, P = self.load_annotations(label_file, calib_file)
        K = P[:, :3]
        shift = np.linalg.inv(K) @ P[:, 3].reshape(3,1)        
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)[:, :, ::-1]
        fig1 = plt.figure(figsize=(11.3, 9))
        ax = plt.subplot(111)
        ax.imshow(image)
        fig2 = plt.figure(figsize=(11.3, 9))
        ax = plt.subplot(111)
        ax.imshow(image)        
        for i, a in enumerate(anns):
            a = a.copy()
            obj_class = a["label"]
            dimension = a["dimensions"]
            locs = np.array(a["locations"])
            rot_y = np.array(a["rot_y"])
            self.render_car(ax, K, obj_class, rot_y, locs, dimension, shift) 
        if save_dir is not None:
            output_path1 =  pjoin(save_dir, image_name + '_original.png')
            output_path2 = pjoin(save_dir, image_name + '_annotated.png')
            make_dir(output_path1)
            fig1.savefig(output_path1, dpi=300)
            fig2.savefig(output_path2, dpi=300)
        return
    
    def _generate_2d_3d_paris(self):
        """
        Prepare pair of 2D screen coordinates and 3D cuboid representation.
        
        """
        path_list = self._data_config['image_path_list']
        kpt_3d_style = self._data_config['3d_kpt_sample_style']
        in_rep = self._data_config['lft_in_rep']
        out_rep = self._data_config['lft_out_rep'] # R3d (Relative 3D shape) encodes 3D rotation
        input_list = []
        output_list = []
        id_list = []
        augment = self._data_config['lft_aug'] if self.split == 'train' else False
        augment_times = self._data_config['lft_aug_times']
        for path in path_list:
            list_2d, list_3d, ids, _ = self.get_2d_3d_pair(path, 
                                                           style=kpt_3d_style,
                                                           in_rep = in_rep,
                                                           out_rep = out_rep,
                                                           augment=augment,
                                                           augment_times=augment_times,
                                                           add_visibility=True
                                                           )            
            input_list += list_2d
            output_list += list_3d
            id_list += ids
        # does not use visibility as input
        num_instance = len(input_list)
        self.input = np.vstack(input_list)[:, :, :2].reshape(num_instance, -1)
        # use visibility as input
        # self.input = np.vstack(input_list).reshape(num_instance, -1)
        self.output = np.vstack(output_list) 
        if hasattr(self, 'root_list'):
            self.root_list = np.vstack(self.root_list)
        self.num_joints = int(self.input.shape[1]/2)      
        return
    
    def generate_pairs(self):
        """
        Prepare data (e.g., input-output pairs and metadata) that will be used 
        depending on the type of experiment.
        """
        if self.exp_type == '2dto3d':           
            # generate 2D screen coordinates and 3D cuboid
            self._generate_2d_3d_paris()
        elif self.exp_type in ['instanceto2d', 'baselinealpha', 'baselinetheta']:
            # # load the annotations containing cropped car instances 
            # path = pjoin(self._data_config['cropped_dir'], 
            #              self.kpts_style, self.split, 'annot.npy')
            # assert exists(path), 'Please prepare instance annotation first.'
            # self.annot_2dpose = np.load(path, allow_pickle=True).item() 
            self.annot_2dpose = self._prepare_2d_pose_annot()
        elif self.exp_type in ['detection2d']:
            self._prepare_detection_records()
            self.total_data = len(self.detection_records)
        elif self.exp_type == 'inference':
            self.gather_annotations()
            self.total_data = len(self.annot_dict)
            self.annoted_img_paths = list(self.annot_dict.keys())
        elif self.exp_type == 'finetune':
            self.gather_annotations(use_raw_bbox=False, 
                                    add_gt=True, 
                                    filter_outlier=True
                                    )
            self.total_data = len(self.annot_dict)
            self.annoted_img_paths = list(self.annot_dict.keys())            
        else:
            raise NotImplementedError('Unknown experiment type.')
        # count of total data
        if self.exp_type == '2dto3d':
            self.input = self.input.astype(np.float32())
            self.output = self.output.astype(np.float32())
            self.total_data = len(self.input)
        elif self.exp_type in ['instanceto2d', 'baselinealpha', 'baselinetheta']:
            self.total_data = len(self.annot_2dpose['paths'])
        return
    
    def visualize(self, plot_num = 1, save_dir=None):
        # show some random images with annotations
        path_list = self._data_config['image_path_list']
        chosen = np.random.choice(len(path_list), plot_num, replace=False)
        for img_idx in chosen:
            self.show_annot(path_list[img_idx], save_dir=save_dir)
        return
    
    def get_collate_fn(self):
        return my_collate_fn
    
    def inference(self, flags=[True, True]):
        self._inference_mode = flags[0]
        self._read_img_during_inference = flags[1]
    
    def extract_ss_sample(self, cnt):
        # cnt: number of fully supervised samples
        extract_cnt = self.ss_settings['max_per_img'] - cnt
        if extract_cnt <= 0:
            num_channel = 5 if self.hm_para['add_xy'] else 3
            return torch.zeros(0, num_channel, 256, 256), None, None, None
        idx = np.random.randint(0, len(self.ss_record['paths']))
        parameters = self.hm_para
        parameters['boxes'] = self.ss_record['boxes'][idx]
        joints = self.ss_record['kpts'][idx]
        img_name = self.ss_record['paths'][idx].split(osep)[-1]
        img_path = pjoin(self.ss_settings['img_root'], img_name)
        image, target, weights, meta = lip.get_tensor_from_img(img_path, 
                                                               parameters, 
                                                               joints=joints,
                                                               pth_trans=self.pth_trans,
                                                               rf=parameters['rf'],
                                                               sf=parameters['sf'],
                                                               generate_hm=False,
                                                               max_cnt=extract_cnt
                                                               )        
        return image, target, weights, meta
    
    def prepare_ft_dict(self, idx):
        img_name = self.annoted_img_paths[idx]
        img_annot = self.annot_dict[img_name]
        ret = {}
        img_path = pjoin(self._data_config['image_dir'], img_name)
        kpts = img_annot['kpts']
        # the croping bounding box in the original image
        # global_box = self.annot_2dpose['global_box'][idx]
        parameters = self.hm_para
        parameters['boxes'] = img_annot['bbox_2d']
        # fs: fully-supervised ss: self-supervised
        images_fs, heatmaps_fs, weights_fs, meta_fs = lip.get_tensor_from_img(img_path, 
                                                                              parameters, 
                                                                              joints=kpts,
                                                                              pth_trans=self.pth_trans,
                                                                              rf=parameters['rf'],
                                                                              sf=parameters['sf'],
                                                                              generate_hm=True)
        ret['path'] = img_path
        ret['images_fs'] = images_fs
        ret['heatmaps_fs'] = heatmaps_fs
        # ret['meta_fs'] = meta_fs
        ret['kpts_3d'] = img_annot['kpts_3d']
        ret['crop_center'] = meta_fs['center']
        ret['crop_scale'] = meta_fs['scale']
        ret['kpts_local'] = meta_fs['transformed_joints']
        # prepare the affine transformation matrices so map local coordinates
        # back to global screen coordinates
        ret['af_mats'] = []
        for idx in range(len(ret['crop_center'])):
            trans_inv = get_affine_transform(ret['crop_center'][idx],
                                             ret['crop_scale'][idx], 
                                             0., 
                                             self.hm_para['input_size'], 
                                             inv=1)  
            ret['af_mats'].append(trans_inv)
        # use random unlabeled images for data augmentation
        if self.split == 'train' and self.use_ss:
            images_ss, heatmaps_ss, weights_ss, meta_ss = self.extract_ss_sample(len(images_fs))
            ret['images_ss'] = images_ss
            ret['meta_ss'] = meta_ss
        return ret
    
    def __getitem__(self, idx):
        # only return testing images during inference
        if self.split == 'test' or self._inference_mode:
            #TODO: consider classes except for cars in the future
            img_name = self.annoted_img_paths[idx]
            # debug: use a specified image for visualization
            # img_name = "006658.png"
            # end debug
            img_path = pjoin(self._data_config['image_dir'], img_name)
            if self._read_img_during_inference:
                image = lip.imread_rgb(img_path)
            else:
                image = None
            if self._read_img_during_inference and hasattr(self, 'pth_trans'):
                # pytorch transformation if provided
                image = self.pth_trans(image)
            record = {'path':img_path}
            # add other available annotations
            if hasattr(self, 'annot_dict'):
                record = {**record, **self.annot_dict[img_name]}
            return image, record
        # for training and validation splits
        if self.exp_type == '2dto3d':
            # the 2D-3D pairs are stored in RAM
            meta_data = {}
            # the 3D global position
            if hasattr(self, 'root_list'):
                meta_data['roots'] = self.root_list[idx]
            return self.input[idx], self.output[idx], np.zeros((0,1)), meta_data
        elif self.exp_type in ['baselinealpha', 'baselinetheta']:
            img_path = self.annot_2dpose['paths'][idx]
            rots = self.annot_2dpose['rots'][idx]
            kpts = self.annot_2dpose['kpts'][idx]
            if kpts.shape[2] == 2:
                kpts = np.concatenate([kpts, np.ones((kpts.shape[0], kpts.shape[1], 1))], axis=2)            
            parameters = self.hm_para
            parameters['boxes'] = self.annot_2dpose['boxes'][idx]
            images_fs, heatmaps_fs, weights_fs, meta_fs = lip.get_tensor_from_img(img_path, 
                                                                                  parameters, 
                                                                                  joints=kpts,
                                                                                  pth_trans=self.pth_trans,
                                                                                  rf=parameters['rf'],
                                                                                  sf=parameters['sf'],
                                                                                  generate_hm=False
                                                                                  )
            if self.exp_type == 'baselinealpha':
                targets = [np.array([[np.cos(rots[idx][0]), np.sin(rots[idx][0])]])  for idx in range(len(rots))]
                meta_fs['angles_gt'] = rots[:, 0]
            elif self.exp_type == 'baselinetheta':
                targets = [np.array([[np.cos(rots[idx][1]), np.sin(rots[idx][1])]]) for idx in range(len(rots))]
                meta_fs['angles_gt'] = rots[:, 1]
            targets = torch.from_numpy(np.concatenate(targets).astype(np.float32))
            return images_fs, targets, weights_fs, meta_fs
        elif self.exp_type == 'instanceto2d':
            # the input images and target heatmaps are produced online
            img_path = self.annot_2dpose['paths'][idx]
            kpts = self.annot_2dpose['kpts'][idx]
            # the croping bounding box in the original image
            # global_box = self.annot_2dpose['global_box'][idx]
            if kpts.shape[2] == 2:
                kpts = np.concatenate([kpts, np.ones((kpts.shape[0], kpts.shape[1], 1))], axis=2)
            parameters = self.hm_para
            parameters['boxes'] = self.annot_2dpose['boxes'][idx]
            # fs: fully-supervised ss: self-supervised
            images_fs, heatmaps_fs, weights_fs, meta_fs = lip.get_tensor_from_img(img_path, 
                                                                   parameters, 
                                                                   joints=kpts,
                                                                   pth_trans=self.pth_trans,
                                                                   rf=parameters['rf'],
                                                                   sf=parameters['sf'],
                                                                   generate_hm=True)
            # use random unlabeled images for data augmentation
            if self.split == 'train' and hasattr(self, 'use_ss') and self.use_ss:
                images_ss, heatmaps_ss, weights_ss, meta_ss = self.extract_ss_sample(len(images_fs))
                images = [images_fs, images_ss]
                targets = heatmaps_fs
                weights = weights_fs
                meta = meta_fs
            else:
                images = images_fs
                targets = heatmaps_fs
                weights = weights_fs
                meta = meta_fs
            return images, targets, weights, meta
        elif self.exp_type == 'detection2d':
            record = copy.deepcopy(self.detection_records[idx])
            path = record['path']
            image = lip.imread_rgb(path)
            target = record['target']
            if hasattr(self, 'pth_trans'):
                # pytorch transformation if provided
                image = self.pth_trans(image)
            return image, target
        elif self.exp_type == 'finetune':
            # prepare images, 2D and 3D annotations as a dictionary for finetuning 
            ret = self.prepare_ft_dict(idx)
            return ret
        else:
            raise NotImplementedError

def prepare_data(cfgs, logger):
    train_set = KITTI(cfgs, 'train', logger)
    valid_set = KITTI(cfgs, 'valid', logger)
    if cfgs['exp_type'] == '2dto3d':
        # normalize 2D keypoints
        valid_set.normalize(train_set.statistics)
    return train_set, valid_set

def get_dataset(cfgs, logger, split):
    return KITTI(cfgs, split, logger)

def collate_dict(dict_list):
    ret = {}
    ret['path'] = [item['path'] for item in dict_list]
    for key in dict_list[0]:
        if key == 'path':
            continue
        ret[key] = np.concatenate([d[key] for d in dict_list], axis=0)
    return ret

def length_limit(instances, targets, target_weights, meta):
    if len(instances) > MAX_INS_CNT and len(instances) == len(targets):
        # normal training
        chosen = np.random.choice(len(instances), MAX_INS_CNT, replace=False)
        ins, tar, tw, = instances[chosen], targets[chosen], target_weights[chosen]
        m = {'path':meta['path']}
        for key in meta:
            if key != 'path':
                m[key] = meta[key][chosen]
    elif len(instances) > MAX_INS_CNT and len(instances) > len(targets) and meta['fs_instance_cnt'] > MAX_INS_CNT:
        # mixed training: fully-supervised instances are too many
        chosen = np.random.choice(meta['fs_instance_cnt'], MAX_INS_CNT, replace=False)
        ins, tar, tw, = instances[chosen], targets[chosen], target_weights[chosen]
        m = {'path':meta['path']}
        for key in meta:
            if key != 'path' and key != 'fs_instance_cnt':
                m[key] = meta[key][chosen]
    elif len(instances) > MAX_INS_CNT and len(instances) > len(targets) and meta['fs_instance_cnt'] <= MAX_INS_CNT:
        # mixed training: self-supervised instances are too many
        ins, tar, tw, m = instances[:MAX_INS_CNT], targets, target_weights, meta
    else:
        ins, tar, tw, m = instances, targets, target_weights, meta
    return ins, tar, tw, m

def my_collate_fn(batch):
    # the collate function for 2d pose training
    instances, targets, target_weights, meta = list(zip(*batch))
    if isinstance(instances[0], list):
        # each batch comes in the format of (fs_instances, ss_instances)
        fs_instances, ss_instances = list(zip(*instances))
        fs_instances = torch.cat(fs_instances)
        ss_instances = torch.cat(ss_instances)
        instances = torch.cat([fs_instances, ss_instances])
        targets = torch.cat(targets, dim=0)
        # target_weights = torch.cat(target_weights, dim=0)
        meta = collate_dict(meta)
        meta['fs_instance_cnt'] = len(fs_instances)
    else:
        instances = torch.cat(instances, dim=0)
        targets = torch.cat(targets, dim=0)
        # target_weights = torch.cat(target_weights, dim=0)
        meta = collate_dict(meta)
    if target_weights[0] is not None:
        target_weights = torch.cat(target_weights, dim=0)
    else:
        #dummy weight
        target_weights = torch.ones(1)
    return length_limit(instances, targets, target_weights, meta)
