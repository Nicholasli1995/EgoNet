"""
Inference of Ego-Net on KITTI dataset.

The user can provide the 3D bounding boxes predicted by other 3D object detectors
and run Ego-Net to refine the orientation of these instances.

The user can also visualize the intermediate results.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import sys
sys.path.append('../')

import libs.arguments.parse as parse
import libs.logger.logger as liblogger
import libs.dataset.KITTI.car_instance as libkitti

from libs.common.img_proc import modify_bbox
from libs.trainer.trainer import get_loader
from libs.model.egonet import EgoNet

import shutil
import torch
import numpy as np
import os

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
    Filter the object detections with a confidence threshold.
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

def gather_dict(request, references, filter_c=True, larger=True):
    """
    Gather a annotation dictionary from the prepared detections as requsted.
    """
    assert 'path' in request
    ret = {'path':[], 
           'boxes':[], 
           'kpts_3d_before':[], 
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
        bbox = ref['bbox_2d']
        if larger:
            # enlarge the input bounding box if needed            
            for instance_id in range(len(bbox)):
                bbox[instance_id] = np.array(modify_bbox(bbox[instance_id], 
                                                         target_ar=1, 
                                                         enlarge=1.2
                                                         )['bbox']
                                             )
        ret['boxes'].append(bbox)
        # 3D key-points from the detections before using Ego-Net
        ret['kpts_3d_before'].append(ref['kpts_3d'])
        # raw prediction strings used for later saving
        ret['raw_txt_format'].append(ref['raw_txt_format'])
        ret['scores'].append(ref['scores'])
        ret['K'].append(ref['K'])
    if 'pose_vecs_gt' in request:
        ret['pose_vecs_gt'] = request['pose_vecs_gt']
    return ret
    
def make_output_dir(cfgs, name):
    save_dir = os.path.join(cfgs['dirs']['output'], name, 'data')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    return save_dir

@torch.no_grad()
def inference(testset, model, results, cfgs):
    """
    The inference loop.
    
    Set cfgs['visualize'] to True if you want to view the results.
    color_dict stores plotting parameters used by Matplotlib.
    save_dict stores parameters relevant to result saving.
    """
    # data loader
    data_loader = get_loader(testset, cfgs, 'testing', collate_fn=my_collate_fn)          
    # transformation statistics
    model.pth_trans = testset.pth_trans 
    all_records = {}
    for batch_idx, (_, meta) in enumerate(data_loader):
        if cfgs['use_gt_box']:
            save_dir = make_output_dir(cfgs, 'gt_box_test')         
            # use ground truth bounding box to crop RoIs
            record = model(meta)
            record = model.post_process(record,
                                        visualize=cfgs['visualize'],
                                        color_dict={'bbox_2d':'y',
                                                    'bbox_3d':'y',
                                                    'kpts':['yx', 'y']
                                                    },
                                        save_dict={
                                            'flag':True,
                                            'save_dir':save_dir
                                            }
                                        )
            merge(all_records, record)
        if cfgs['use_pred_box']:
            # use detected bounding box from any 2D/3D detector
            annot_dict = gather_dict(meta, results['pred'])
            if len(annot_dict['path']) == 0:
                continue
            record2 = model(annot_dict)
            # update drawings
            for key in record2:
                if 'record' in locals() and 'plots' in record[key]:
                    record2[key]['plots'] = record[key]['plots']
            save_dir = make_output_dir(cfgs, 'submission')   
            record2 = model.post_process(record2, 
                                         visualize=cfgs['visualize'],
                                         color_dict={'bbox_2d':'r',
                                                     'bbox_3d':'r',
                                                     'kpts':['rx', 'r'],
                                                     },
                                         save_dict={'flag':True,
                                                    'save_dir':save_dir
                                                    },
                                         alpha_mode=cfgs['testing_settings']['alpha_mode']
                                         )   
        # set batch_to_show to a small number if you need to visualize 
        if batch_idx >= cfgs['batch_to_show'] - 1:
            break     
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
    
    # save a copy of the experiment configuration
    shutil.copyfile(cfgs['config_path'], os.path.join(final_output_dir, 'saved_config.yml'))
    
    # set GPU
    if cfgs['use_gpu'] and torch.cuda.is_available():
        logger.info('Using GPU:{}'.format(cfgs['gpu_id']))
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, cfgs['gpu_id'])))
    else:
        raise ValueError('CPU-based inference is not maintained.')
        
    # cudnn related setting
    torch.backends.cudnn.benchmark = cfgs['cudnn']['benchmark']
    torch.backends.cudnn.deterministic = cfgs['cudnn']['deterministic']
    torch.backends.cudnn.enabled = cfgs['cudnn']['enabled']
    
    # configurations related to the KITTI dataset
    data_cfgs = cfgs['dataset']
    
    # which split to show
    split = data_cfgs['split'] # default: KITTI val split
    dataset_inf = libkitti.get_dataset(cfgs, logger, split)
    
    # set the dataset to inference mode
    dataset_inf.inference([True, False])
    
    # read annotations
    input_file_path = cfgs['dirs']['load_prediction_file']
    # the record for 2D and 3D predictions
    # key->value: name of the approach->dictionary storing the predictions
    results = {}
    # confidence_thres = cfgs['conf_thres']
    
    # flags: the user can choose to use which type of input bounding boxes to use
    # use_gt_box can be used to re-produce the experiments simulating perfect 2D detection
    results['flags'] = {}
    if cfgs['use_pred_box']:
        # read the predicted boxes as specified by the path
        results['pred'] = dataset_inf.read_predictions(input_file_path)
    
    # Initialize Ego-Net and load the pre-trained checkpoint
    model = EgoNet(cfgs, pre_trained=True)
    model = model.eval().cuda()
    
    # perform inference and save the (updated) predictions
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