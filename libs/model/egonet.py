"""
A PyTorch implementation of Ego-Net.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import torch
import torch.nn as nn
import numpy as np
import cv2

import libs.model as models
import libs.model.FCmodel as FCmodel
import libs.dataset.normalization.operations as nop

from libs.common.img_proc import to_npy, resize_bbox, get_affine_transform, get_max_preds, generate_xy_map
from libs.common.img_proc import affine_transform_modified, cs2bbox, simple_crop, enlarge_bbox

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
    
    def modify_bbox(self, bbox, target_ar, enlarge=1.1):
        """
        Enlarge a bounding box so that occluded parts may be included.
        """
        lbbox = enlarge_bbox(bbox[0], bbox[1], bbox[2], bbox[3], [enlarge, enlarge])
        ret = resize_bbox(lbbox[0], lbbox[1], lbbox[2], lbbox[3], target_ar=target_ar)
        return ret

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
        ret = self.modify_bbox(bbox, target_ar)
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
        # each record describes one instance
        all_records = []
        target_ar = resolution[0] / resolution[1]
        for idx, path in enumerate(annot_dict['path']):
            #print(path)
            data_numpy = cv2.imread(path, 1 | 128)    
            if data_numpy is None:
                raise ValueError('Fail to read {}'.format(path))    
            if rgb:
                data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB) 
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
                ret = self.modify_bbox(bbox, target_ar)
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
            if 'kpts_3d_SMOKE' in annot_dict:
                records[path]['kpts_3d_SMOKE'] = to_npy(annot_dict['kpts_3d_SMOKE'][idx])  
            if 'raw_txt_format' in annot_dict:
                # list of annotation dictionary for each instance
                records[path]['raw_txt_format'] = annot_dict['raw_txt_format'][idx]
            if 'K' in annot_dict:
                # list of annotation dictionary for each instance
                records[path]['K'] = annot_dict['K'][idx]
            if 'kpts_3d_gt' in annot_dict and 'K' in annot_dict:
                records[path]['arrow'] = self.add_orientation_arrow(records[path])
        return records

    def get_keypoints(self,
                      instances, 
                      records,
                      image_size=(256,256), 
                      arg_max='hard',
                      is_cuda=True
                      ):
        """
        Foward pass to obtain the screen coordinates.
        """
        if is_cuda:
            instances = instances.cuda()
        output = self.HC(instances)
        if type(output) is tuple:
            pred, max_vals = output[1].data.cpu().numpy(), None  
            
        elif arg_max == 'hard':
            if not isinstance(output, np.ndarray):
                output = output.data.cpu().numpy()
            pred, max_vals = get_max_preds(output)
        else:
            raise NotImplementedError
        if type(output) is tuple:
            pred *= image_size[0]
        else:
            pred *= image_size[0]/output.shape[3]
        centers = [records[i]['center'] for i in range(len(records))]
        scales = [records[i]['scale'] for i in range(len(records))]
        rots = [records[i]['rotation'] for i in range(len(records))]    
        for sample_idx in range(len(pred)):
            trans_inv = get_affine_transform(centers[sample_idx],
                                             scales[sample_idx], 
                                             rots[sample_idx], 
                                             image_size, 
                                             inv=1)
            pred_src_coordinates = affine_transform_modified(pred[sample_idx], 
                                                                 trans_inv) 
            record = records[sample_idx]
            # pred_src_coordinates += np.array([[record['bbox'][0], record['bbox'][1]]])
            records[sample_idx]['kpts'] = pred_src_coordinates
        # assemble a dictionary where each key corresponds to one image
        ret = {}
        for record in records:
            path = record['path']
            if path not in ret:
                ret[path] = {'center':[], 
                             'scale':[], 
                             'rotation':[], 
                             'bbox_resize':[], # resized bounding box
                             'kpts_2d_pred':[], 
                             'label':[], 
                             'score':[]
                             }
            ret[path]['kpts_2d_pred'].append(record['kpts'].reshape(1, -1))
            ret[path]['center'].append(record['center'])
            ret[path]['scale'].append(record['scale'])
            ret[path]['bbox_resize'].append(record['bbox_resize'])
            ret[path]['label'].append(record['label'])
            ret[path]['score'].append(record['score'])
            ret[path]['rotation'].append(record['rotation'])
        return ret

    def lift_2d_to_3d(self, records, template, cuda=True):
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
    
    def forward(self, 
                images, 
                template,
                annot_dict,
                pth_trans=None, 
                is_cuda=True, 
                threshold=None,
                xy_dict=None
                ):
        """
        Process a batch of images.
        
        annot_dict is a Python dictionary storing the following keys: 
            path: list of image paths
            boxes: list of bounding boxes for each image
        """
        all_instances, all_records = self.crop_instances(annot_dict, 
                                                         resolution=(256, 256),
                                                         pth_trans=pth_trans,
                                                         xy_dict=xy_dict
                                                         )
        # all_records stores records for each instance
        records = self.get_keypoints(all_instances, all_records)
        # records stores records for each image
        records = self.lift_2d_to_3d(records)
        # write the annotation dictionary
        records = self.write_annot_dict(annot_dict, records)
        return records