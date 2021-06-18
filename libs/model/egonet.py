"""
A PyTorch implementation of Ego-Net.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

import libs.model as models
import libs.model.FCmodel as FCmodel
import libs.dataset.normalization.operations as nop
import libs.visualization.points as vp

from libs.common.img_proc import to_npy, resize_bbox, get_affine_transform, get_max_preds, generate_xy_map
from libs.common.img_proc import affine_transform_modified, cs2bbox, simple_crop, enlarge_bbox
from libs.common.format import save_txt_file

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

    def gather_lifting_results(self,
                               record,
                               data,
                               prediction, 
                               target=None,
                               pose_vecs=None,
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
        Lift Screen coordinates to 3D representation and a optimization-based 
        refinement is optional.
        """
        if target is not None:
            p3d_gt = target.reshape(len(target), -1, 3)
        else:
            p3d_gt = None
        p3d_pred = prediction.reshape(len(prediction), -1, 3)
        # only for visualizing the prediciton of shape using gt bboxes
        if "kpts_3d_SMOKE" in record:
            p3d_pred = np.concatenate([record['kpts_3d_SMOKE'][:, [0], :], p3d_pred], axis=1)
        elif p3d_gt is not None and p3d_gt.shape[1] == p3d_pred.shape[1] + 1:
            if len(p3d_pred) != len(p3d_gt):
                print('debug')
            assert len(p3d_pred) == len(p3d_gt)
            p3d_pred = np.concatenate([p3d_gt[:, [0], :], p3d_pred], axis=1) 
        else:
            raise NotImplementedError
        # this object will be updated if one prediction is refined 
        p3d_pred_refined = p3d_pred.copy()
        refine_flags = [False for i in range(len(p3d_pred_refined))]
        # similar object but using a different refinement strategy
        p3d_pred_refined2 = p3d_pred.copy()
        refine_flags2 = [False for i in range(len(p3d_pred_refined2))]
        # input 2D keypoints
        data = data.reshape(len(data), -1, 2)
        if visualize:
            if 'plots' in record and 'ax3d' in record['plots']:
                ax = record['plots']['ax3d']
                ax = vp.plot_scene_3dbox(p3d_pred, p3d_gt, ax=ax, color=color)
            elif 'plots' in record:
            # plotting the 3D scene
                ax = vp.plot_scene_3dbox(p3d_pred, p3d_gt, color=color)
                vp.draw_pose_vecs(ax, pose_vecs)
                record['plots']['ax3d'] = ax
            else:
                raise ValueError
        else:
            ax = None
        if refine:
            assert intrinsics is not None         
            # refine 3D point prediction by minimizing re-projection errors        
            refine_solution(p3d_pred, 
                            data, 
                            intrinsics, 
                            dist_coeffs, 
                            refine_with_predicted_bbox, 
                            p3d_pred_refined, 
                            refine_flags,
                            ax=ax
                            )
            if target is not None:
                # refine with ground truth bounding box size for debugging purpose
                refine_solution(p3d_pred, 
                                data, 
                                intrinsics, 
                                dist_coeffs, 
                                refine_with_perfect_size, 
                                p3d_pred_refined2, 
                                refine_flags2,
                                gts=p3d_gt,
                                ax=ax
                                )        
        record['kpts_3d_refined'] = p3d_pred_refined  
        # prepare the prediction string for submission
        # compute the roll, pitch and yaw angle of the predicted bounding box
        record['euler_angles'], record['translation'] = \
            get_6d_rep(record['kpts_3d_refined'], ax, color=color) # the predicted pose vectors are also drawn here
        if alpha_mode == 'trans':
            record['alphas'] = get_observation_angle_trans(record['euler_angles'], 
                                                           record['translation'])
        elif alpha_mode == 'proj':
            record['alphas'] = get_observation_angle_proj(record['euler_angles'],
                                                          record['kpts_2d_pred'],
                                                          record['K'])        
        else:
             raise NotImplementedError   
        if get_str:
            record['pred_str'] = get_pred_str(record)      
        return record

    def plot_one_image(self,
                         img_path, 
                         record, 
                         add_3d_bbox=True, 
                         camera=None, 
                         template=None,
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
        Refine the predictions from a single image.
        """
        # plot 2D predictions 
        if visualize:
            if 'plots' in record:
                fig = record['plots']['fig2d']
                ax = record['plots']['ax2d']
            else:
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
                
            num_instances = len(record['kpts_2d_pred'])
            for idx in range(num_instances):
                kpts = record['kpts_2d_pred'][idx].reshape(-1, 2)
                # kpts_3d = record['kpts_3d'][idx]
                bbox = record['bbox_resize'][idx]
                label = record['label'][idx]
                score = record['score'][idx]
                vp.plot_2d_bbox(ax, bbox, color_dict['bbox_2d'], score, label)
                # predicted key-points
                ax.plot(kpts[:, 0], kpts[:, 1], color_dict['kpts'][0])
                # if add_3d_bbox:
                #     vp.plot_3d_bbox(ax, kpts[1:,], color_dict['kpts'][1])  
                # bbox_3d_projected = project_3d_to_2d(kpts_3d)
                # vp.plot_3d_bbox(ax, bbox_3d_projected[:2, :].T)      
            # plot ground truth
            if 'kpts_2d_gt' in record:
                for idx, kpts_gt in enumerate(record['kpts_2d_gt']):
                    kpts_gt = kpts_gt.reshape(-1, 3)
                    # ax.plot(kpts_gt[:, 0], kpts_gt[:, 1], 'gx')
                    vp.plot_3d_bbox(ax, kpts_gt[1:, :2], color='g', linestyle='-.')
            if 'arrow' in record:
                for idx in range(len(record['arrow'])):
                    start = record['arrow'][idx][:,0]
                    end = record['arrow'][idx][:,1]
                    x, y = start
                    dx, dy = end - start
                    ax.arrow(x, y, dx, dy, color='r', lw=4, head_width=5, alpha=0.5)         
                # save intermediate results
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                img_name = img_path.split('/')[-1]
                save_dir = './qualitative_results/'
                plt.savefig(save_dir + img_name, dpi=100, bbox_inches = 'tight', pad_inches = 0)
        # plot 3d bounding boxes
        all_kpts_2d = np.concatenate(record['kpts_2d_pred'])
        all_kpts_3d_pred = record['kpts_3d_pred'].reshape(len(record['kpts_3d_pred']), -1)
        if 'kpts_3d_gt' in record:
            all_kpts_3d_gt = record['kpts_3d_gt']
            all_pose_vecs_gt = record['pose_vecs_gt']
        else:
            all_kpts_3d_gt = None
            all_pose_vecs_gt = None
        refine_args = {'visualize':visualize, 'get_str':save_dict['flag']}
        if camera is not None:
            refine_args['intrinsics'] = camera
            refine_args['refine'] = True
            refine_args['template'] = template
        # refine and gather the prediction strings
        record = self.gather_lifting_results(record,
                                        all_kpts_2d,
                                        all_kpts_3d_pred, 
                                        all_kpts_3d_gt,
                                        all_pose_vecs_gt,
                                        color=color_dict['bbox_3d'],
                                        alpha_mode=alpha_mode,
                                        **refine_args
                                        )
        # plot 3D bounding box generated by SMOKE
        if 'kpts_3d_SMOKE' in record:
            kpts_3d_SMOKE = record['kpts_3d_SMOKE']
            if 'plots' in record:
                # update drawings
                ax = record['plots']['ax3d']
                vp.plot_scene_3dbox(kpts_3d_SMOKE, ax=ax, color='m')    
                pose_vecs = np.zeros((len(kpts_3d_SMOKE), 6))
                for idx in range(len(pose_vecs)):
                    pose_vecs[idx][0:3] = record['raw_txt_format'][idx]['locations']
                    pose_vecs[idx][4] = record['raw_txt_format'][idx]['rot_y']
                # plot pose vectors
                vp.draw_pose_vecs(ax, pose_vecs, color='m')
        # save KITTI-style prediction file in .txt format
        save_txt_file(img_path, record, save_dict)
        return record

    def post_process(self, 
                     records, 
                     camera=None, 
                     template=None, 
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
            print(img_path)
            records[img_path] = self.plot_one_image(img_path, 
                                                      records[img_path], 
                                                      camera=camera,
                                                      template=template,
                                                      visualize=visualize,
                                                      color_dict=color_dict,
                                                      save_dict=save_dict,
                                                      alpha_mode=alpha_mode
                                                      )      
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