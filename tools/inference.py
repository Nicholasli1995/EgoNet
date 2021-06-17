"""
Inference of Ego-Net on KITTI dataset.

The user can provide the 3D bounding boxes predicted by other 3D object detectors
and run Ego-Net to refine the orientation of these instances.
"""
import sys
sys.path.append('../')

import libs.arguments.parse as parse
import libs.logger.logger as liblogger
import libs.dataset as dataset
import libs.dataset.KITTI.car_instance
import libs.dataset.normalization.operations as nop
import libs.visualization.points as vp
import libs.common.transformation as ltr

from libs.common.img_proc import resize_bbox, get_affine_transform, get_max_preds, generate_xy_map
from libs.common.img_proc import affine_transform_modified, cs2bbox, simple_crop, enlarge_bbox
from libs.trainer.trainer import visualize_lifting_results, get_loader
from libs.dataset.KITTI.car_instance import interp_dict
from libs.model.egonet import EgoNet

import shutil
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.spatial.transform import Rotation

def kpts_to_euler(template, prediction):
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

def get_template(prediction, interp_coef=[0.332, 0.667]):
    """
    Construct a template 3D cuboid used for computing regid transformation.
    """ 
    parents = prediction[interp_dict['bbox12'][0]]
    children = prediction[interp_dict['bbox12'][1]]
    lines = parents - children
    lines = np.sqrt(np.sum(lines**2, axis=1))
    h = np.sum(lines[:4])/4 # averaged over the four parallel line segments
    l = np.sum(lines[4:8])/4
    w = np.sum(lines[8:])/4
    x_corners = [0.5*l, l, l, l, l, 0, 0, 0, 0]
    y_corners = [0.5*h, 0, h, 0, h, 0, h, 0, h]
    z_corners = [0.5*w, w, w, 0, 0, w, w, 0, 0]
    x_corners += - np.float32(l) / 2
    y_corners += - np.float32(h)
    #y_corners += - np.float32(h/2)
    z_corners += - np.float32(w) / 2
    corners_3d = np.array([x_corners, y_corners, z_corners])    
    if len(prediction) == 33:
        pidx, cidx = interp_dict['bbox12']
        parents, children = corners_3d[:, pidx], corners_3d[:, cidx]
        lines = children - parents
        new_joints = [(parents + interp_coef[i]*lines) for i in range(len(interp_coef))]
        corners_3d = np.hstack([corners_3d, np.hstack(new_joints)])    
    return corners_3d

def get_observation_angle_trans(euler_angles, translations):
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

def get_observation_angle_proj(euler_angles, kpts, K):
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

def get_6d_rep(predictions, ax=None, color="black"):
    """
    Get the 6DoF representation of a 3D prediction.
    """    
    predictions = predictions.reshape(len(predictions), -1, 3)
    all_angles = []
    for instance_idx in range(len(predictions)):
        prediction = predictions[instance_idx]
        # templates are 3D boxes with no rotation
        # the prediction is estimated as the rotation between prediction and template
        template = get_template(prediction)
        instance_angle, instance_trans = kpts_to_euler(template, prediction.T)        
        all_angles.append(instance_angle.reshape(1, 3))
    angles = np.concatenate(all_angles)
    # the first point is the predicted point center
    translation = predictions[:, 0, :]    
    if ax is not None:
        pose_vecs = np.concatenate([translation, angles], axis=1)
        draw_pose_vecs(ax, pose_vecs, color=color)
    return angles, translation

def filter_detection(detected, thres=0.7):
    """
    Filter predictions based on a confidence threshold.
    """      
    # detected: list of dict
    filtered = []
    for detection in detected:
        tempt_dict = {}
        indices = detection['scores'] > thres
        for key in ['boxes', 'labels', 'scores']:
            tempt_dict[key] = detection[key][indices]
        filtered.append(tempt_dict)
    return filtered

def refine_solution(est_3d, 
                    est_2d, 
                    K, 
                    dist_coeffs, 
                    refine_func, 
                    output_arr, 
                    output_flags, 
                    gts=None, 
                    ax=None
                    ):
    """
    Refine 3D prediction by minimizing re-projection error.
    est: estimates [N, 9, 3]
    K: intrinsics    
    """      
    for idx in range(len(est_3d)):
        success, refined_prediction = refine_func(est_3d[idx],
                                                  est_2d[idx],
                                                  K,
                                                  dist_coeffs,
                                                  gts=gts,
                                                  ax=ax)
        if success:
            # update the refined solution
            output_arr[idx] = refined_prediction.T
            output_flags[idx] = True
            # # convert to the center-relative shape representation
            # p3d_pred_refined[idx][1:, :] -= p3d_pred_refined[idx][[0]]    
    return

def gather_lifting_results(record,
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
            draw_pose_vecs(ax, pose_vecs)
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

def save_txt_file(img_path, prediction, params):
    """
    Save a txt file for predictions of an image.
    """    
    if not params['flag']:
        return
    file_name = img_path.split('/')[-1][:-3] + 'txt'
    save_path = os.path.join(params['save_dir'], file_name) 
    with open(save_path, 'w') as f:
        f.write(prediction['pred_str'])
    return

def refine_one_image(img_path, 
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
            save_dir = '/media/nicholas/Database/datasets/KITTI/qualitative_results/'
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
    record = gather_lifting_results(record,
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
            draw_pose_vecs(ax, pose_vecs, color='m')
    # save KITTI-style prediction file in .txt format
    save_txt_file(img_path, record, save_dict)
    return record

def post_process(records, 
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
    for img_path in records.keys():
        print(img_path)
        records[img_path] = refine_one_image(img_path, 
                                             records[img_path], 
                                             camera=camera,
                                             template=template,
                                             visualize=visualize,
                                             color_dict=color_dict,
                                             save_dict=save_dict,
                                             alpha_mode=alpha_mode
                                             )      
    return records

def merge(dict_a, dict_b):
    for key in dict_b.keys():
        dict_a[key] = dict_b[key]
    return

def collate_dict(dict_list):
    ret = {}
    for key in dict_list[0]:
        ret[key] = [d[key] for d in dict_list]
    return ret

def my_collate_fn(batch):
    # the collate function for 2d pose training
    imgs, meta = list(zip(*batch))
    meta = collate_dict(meta)
    return imgs, meta

def filter_conf(record, thres=0.0):
    """
    Filter the proposals with a confidence threshold.
    """
    annots = record['raw_txt_format']
    indices = [i for i in range(len(annots)) if annots[i]['score'] >= thres]
    if len(indices) == 0:
        return False, record
    filterd_record = {
        'bbox_2d': record['bbox_2d'][indices],
        'kpts_3d': record['kpts_3d'][indices],
        'raw_txt_format': [annots[i] for i in indices],
        'scores': [annots[i]['score'] for i in indices],
        'K':record['K']
        }
    return True, filterd_record

def gather_dict(request, references, filter_c=True):
    """
    Gather a dict from reference as requsted.
    """
    assert 'path' in request
    ret = {'path':[], 
           'boxes':[], 
           'kpts_3d_SMOKE':[], 
           'raw_txt_format':[],
           'scores':[],
           'K':[]}
    for img_path in request['path']:
        img_name = img_path.split('/')[-1]
        if img_name not in references:
            print('Warning: ' + img_name + ' not included in detected images!')
            continue
        ref = references[img_name]
        if filter_c:
            success, ref = filter_conf(ref)
        if filter_c and not success:
            continue
        ret['path'].append(img_path)
        # ret['boxes'].append(ref['bbox_2d'])
        # temporary hack: enlarge the bounding box from the stage 1 model
        bbox = ref['bbox_2d']
        for instance_id in range(len(bbox)):
            bbox[instance_id] = np.array(modify_bbox(bbox[instance_id], target_ar=1, enlarge=1.2)['bbox'])
        # temporary hack 2: use the gt bounding box for analysis
        ret['boxes'].append(bbox)
        # 3D bounding box produced by SMOKE
        ret['kpts_3d_SMOKE'].append(ref['kpts_3d'])
        ret['raw_txt_format'].append(ref['raw_txt_format'])
        ret['scores'].append(ref['scores'])
        ret['K'].append(ref['K'])
    #ret['kpts_3d_gt'] = request['kpts_3d_gt']
    if 'pose_vecs_gt' in request:
        ret['pose_vecs_gt'] = request['pose_vecs_gt']
    return ret
    
@torch.no_grad()
def inference(testset, model, results, cfgs):
    """
    The main inference function.
    """
    # visualize to plot the 2D detection and 3D scene reconstruction
    data_loader = get_loader(testset, cfgs, 'testing', collate_fn=my_collate_fn)          
    #template = testset.instance_stats['ref_box3d']
    template = None
    pth_trans = testset.pth_trans
    if 'add_xy' in cfgs['heatmapModel']:
        xy_dict = {'flag':cfgs['heatmapModel']['add_xy']}
    else:
        xy_dict = None
    all_records = {}
    camera = None
    flags = results['flags']
    visualize = cfgs['visualize']
    batch_to_show = cfgs['batch_to_show']
    for batch_idx, (images, meta) in enumerate(data_loader):
        if flags['gt']:
            save_dir = os.path.join(cfgs['dirs']['output'], 'gt_box_test', 'data')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)            
            # ground truth bounding box for comparison
            record = model(images, 
                           model,
                                   annot_dict=meta,
                                   pth_trans=pth_trans, 
                                   threshold=None,
                                   xy_dict=xy_dict
                                   )
            record = post_process(record, 
                                  camera, 
                                  template, 
                                  visualize=visualize,
                                  color_dict={'bbox_2d':'y',
                                              'bbox_3d':'y',
                                              'kpts':['yx', 'y'],
                                              },
                                  save_dict={
                                      'flag':True,
                                      'save_dir':save_dir
                                      }
                                  )
            merge(all_records, record)
        if flags['pred']:
            # use detected bounding box from an anchor-free model
            annot_dict = gather_dict(meta, results['pred'])
            if len(annot_dict['path']) == 0:
                continue
            record2 = process_batch(images,
                                    hm_regressor, 
                                    lifter, 
                                    stats, 
                                    template,
                                    annot_dict,
                                    pth_trans=pth_trans, 
                                    threshold=None,
                                    xy_dict=xy_dict
                                    )
            for key in record2:
                if 'record' in locals() and 'plots' in record[key]:
                    record2[key]['plots'] = record[key]['plots']
            save_dir = os.path.join(cfgs['dirs']['output'], 'submission', 'data')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            record2 = post_process(record2,
                                   camera, 
                                   template, 
                                   visualize=visualize,
                                   color_dict={'bbox_2d':'r',
                                               'bbox_3d':'r',
                                               'kpts':['rx', 'r'],
                                               },
                                   save_dict={'flag':True,
                                              'save_dir':save_dir
                                              },
                                  alpha_mode=cfgs['testing_settings']['alpha_mode']
                                  )   
            del images, record2, meta
        if batch_idx >= batch_to_show - 1:
            break
    # produce a csv file
    # csv_output_path = cfgs['dirs']['csv_output']
    # save_csv(all_records, csv_output_path)        
    return

def generate_empty_file(output_dir, label_dir):
    """
    Generate empty files for images without any predictions.
    """    
    all_files = os.listdir(label_dir)
    detected = os.listdir(os.path.join(output_dir, 'data'))
    for file_name in all_files:
        if file_name[:-4] + ".txt" not in detected:
            file = open(os.path.join(output_dir, 'data', file_name[:-4] + '.txt'), 'w')
            file.close()
    return

def main():
    # experiment configurations
    cfgs = parse.parse_args()
    
    # logging
    logger, final_output_dir = liblogger.get_logger(cfgs)   
    shutil.copyfile(cfgs['config_path'], os.path.join(final_output_dir, 'saved_config.yml'))
    
    # Set GPU
    if cfgs['use_gpu'] and torch.cuda.is_available():
        GPUs = cfgs['gpu_id']
    else:
        raise ValueError('CPU-based inference is not maintained.')
        
    # cudnn related setting
    torch.backends.cudnn.benchmark = cfgs['cudnn']['benchmark']
    torch.backends.cudnn.deterministic = cfgs['cudnn']['deterministic']
    torch.backends.cudnn.enabled = cfgs['cudnn']['enabled']
    
    # configurations related to the KITTI dataset
    data_cfgs = cfgs['dataset']
    
    # which split to show
    split = 'valid' # default: KITTI val split
    dataset_inf = eval('dataset.' + data_cfgs['name']  
                        + '.car_instance').get_dataset(cfgs, logger, split)
    
    # set to inference mode but does not read image
    dataset_inf.inference([True, False])
    
    # read annotations
    input_file_path = cfgs['dirs']['load_prediction_file']
    # the record for 2D and 3D predictions
    # key->value: name of the approach->dictionary storing the predictions
    results = {}
    confidence_thres = cfgs['conf_thres']
    
    # flags: the user can choose to use which type of input bounding boxes to use
    # use_gt_box can be used to re-produce the experiments simulating perfect 2D detection
    results['flags'] = {}
    results['flags']['pred'] = cfgs['use_pred_box']
    if results['flags']['pred']:
        # read the predicted boxes as specified by the path
        results['pred'] = dataset_inf.read_predictions(input_file_path)
    results['flags']['gt'] = cfgs['use_gt_box']
    
    # Initialize Ego-Net and load the pre-trained checkpoint
    model = EgoNet(cfgs)
    
    # inference and update prediction
    inference(dataset_inf, model, results, cfgs)       
    
    # then you can run kitti-eval for evaluation
    evaluator = cfgs['dirs']['kitti_evaluator']
    label_dir = cfgs['dirs']['kitti_label']
    output_dir = os.path.join(cfgs['dirs']['output'], 'submission')
    
    # When generating submission files for the test split,
    # if no detections are produced for one image, generate an empty file
    #generate_empty_file(output_dir, label_dir)
    command = "{} {} {}".format(evaluator, label_dir, output_dir)
    # e.g.
    # ~/Documents/Github/SMOKE/smoke/data/datasets/evaluation/kitti/kitti_eval/evaluate_object_3d_offline /home/nicholas/Documents/Github/SMOKE/datasets/kitti/training/label_2 /media/nicholas/Database/experiments/3DLearning/0826
    # /media/nicholas/Database/Github/M3D-RPN/data/kitti_split1/devkit/cpp/evaluate_object /home/nicholas/Documents/Github/SMOKE/datasets/kitti/training/label_2 /media/nicholas/Database/Github/M3D-RPN/output/tmp_results
    return

if __name__ == "__main__":
    main()
