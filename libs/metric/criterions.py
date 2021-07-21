"""
Metric functions used for validation.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import libs.common.transformation as ltr
import libs.common.img_proc as lip
from libs.common.transformation import compute_similarity_transform

import numpy as np
import torch
from scipy.spatial.transform import Rotation

PCK_THRES = np.array([0.1, 0.2, 0.3])

def get_distance(gt, pred):
    """
    2D Euclidean distance of two groups of points with visibility considered. 
    
    gt: [n_joints, 2 or 3]
    pred: [n_joints, 2]
    """    
    if gt.shape[1] == 2:
        sqerr = (gt - pred)**2
        sqerr = sqerr.sum(axis = 1)
        dist_list = list(np.sqrt(sqerr))
    elif gt.shape[1] == 3:
        dist_list = []
        sqerr = (gt[:, :2] - pred)**2
        sqerr = sqerr.sum(axis = 1)
        indices = np.nonzero(gt[:, 2])[0]
        dist_list = list(np.sqrt(sqerr[indices]))        
    else:
        raise ValueError('Array shape not supported.')
    return dist_list

def get_angle_error(pred, meta_data, cfgs=None):
    """
    Compute error for angle prediction.
    """    
    if not isinstance(pred, np.ndarray):
        pred = pred.data.cpu().numpy()    
    angles_pred = np.arctan2(pred[:,1], pred[:,0])
    angles_gt = meta_data['angles_gt']
    dif = np.abs(angles_gt - angles_pred) * 180 / np.pi
    # add or minus 2*pi
    indices = dif > 180
    dif[indices] = 360 - dif[indices]
    cnt = len(pred)
    avg_acc = dif.sum()/cnt
    others = None
    return avg_acc, cnt, others

def get_PCK(pred, gt):
    """
    Get percentage of correct key-points
    """
    distance = np.array(get_distance(gt, pred))
    denominator = (gt[:, 1].max() - gt[:, 1].min()) * 1/3
    correct_cnt = np.zeros((len(PCK_THRES)))
    for idx, thres in enumerate(PCK_THRES):
        correct_cnt[idx] = (distance < thres * denominator).sum()
    return correct_cnt

def get_distance_src(output,
                     meta_data,
                     cfgs=None,
                     image_size = (256.0, 256.0),
                     arg_max='hard'
                     ):
    """
    From predicted heatmaps, obtain key point locations and transform them back
    into the source images based on metadata. Error is then evaluated on the
    source image.
    """
    # report distance in terms of pixel in the original image
    if type(output) is tuple:
        pred, max_vals = output[1].data.cpu().numpy(), None
    elif isinstance(output, np.ndarray) and arg_max == 'soft':
        pred, max_vals = lip.soft_arg_max_np(output)
    elif isinstance(output, torch.Tensor) and arg_max == 'soft': 
        pred, max_vals = lip.soft_arg_max(output)
    elif isinstance(output, np.ndarray) or isinstance(output, torch.Tensor) and arg_max == 'hard':
        if not isinstance(output, np.ndarray):
            output = output.data.cpu().numpy()        
        pred, max_vals = lip.get_max_preds(output)
    else:
        raise NotImplementedError
    image_size = image_size if cfgs is None else cfgs['heatmapModel']['input_size']
    # TODO: check the target generation and coordinate mapping
    # multiply by down-sample ratio
    if not isinstance(pred, np.ndarray):
        pred = pred.data.cpu().numpy()
    if (max_vals is not None) and (not isinstance(max_vals, np.ndarray)):
        max_vals = max_vals.data.cpu().numpy()
    # the coordinates need to be rescaled for different cases
    if type(output) is tuple:
        pred *= image_size[0]
    else:
        pred *= image_size[0]/output.shape[3]
    # inverse transform and compare pixel didstance
    centers, scales = meta_data['center'], meta_data['scale']
    # some predictions are generated for unlabeled data
    if len(pred) != len(centers):
        pred_used = pred[:len(centers)]
    else:
        pred_used = pred
    if 'rotation' in meta_data:
        rots = meta_data['rotation']
    else:
        rots = [0. for i in range(len(centers))]
    joints_original_batch = meta_data['original_joints']
    distance_list = []
    correct_cnt_sum = np.zeros((len(PCK_THRES)))
    all_src_coordinates = []
    for sample_idx in range(len(pred_used)):
        trans_inv = lip.get_affine_transform(centers[sample_idx], 
                                             scales[sample_idx], 
                                             rots[sample_idx], 
                                             image_size, 
                                             inv=1)
        joints_original = joints_original_batch[sample_idx]        
        pred_src_coordinates = lip.affine_transform_modified(pred_used[sample_idx], 
                                                             trans_inv) 
        all_src_coordinates.append(pred_src_coordinates.reshape(1, len(pred_src_coordinates), 2))
        distance_list += get_distance(joints_original, pred_src_coordinates)
        correct_cnt_sum += get_PCK(pred_src_coordinates, joints_original)
    cnt = len(distance_list)
    avg_acc = sum(distance_list)/cnt
    others = {
        'src_coord': np.concatenate(all_src_coordinates, axis=0),
        'joints_pred': pred,
        'max_vals': max_vals,
        'correct_cnt': correct_cnt_sum,
        'PCK_batch': correct_cnt_sum/cnt
        }
    return avg_acc, cnt, others

class AngleError():
    """
    angle error in degrees. 
    """  
    def __init__(self, cfgs, num_joints=None):
        self.name = 'Angle error in degrees'
        self.num_joints = num_joints
        self.count = 0
        self.mean = 0.
        return
  
    def update(self, prediction, meta_data, ground_truth=None, logger=None):
        """
        the prediction and transformation parameters in meta_data are used.
        """    
        avg_acc, cnt, others = get_angle_error(prediction, meta_data)
        self.mean = (self.mean * self.count + cnt * avg_acc) / (self.count + cnt)
        self.count += cnt
        return 
    
    def report(self, logger):
        msg = 'Error type: {error_type:s}\t' \
              'Error: {Error}\t'.format(
                      error_type = self.name,
                      Error = self.mean)     
        logger.info(msg)        
        return

class JointDistance2DSIP():
    """
    joint distance in the source image plane (SIP). 
    """  
    def __init__(self, cfgs, num_joints=None):
        self.name = 'Joint distance in the source image plane'
        if num_joints is not None:
            self.num_joints = num_joints
        else:
            self.num_joints = cfgs['heatmapModel']['num_joints']
        self.image_size = cfgs['heatmapModel']['input_size']
        self.arg_max = cfgs['testing_settings']['arg_max']
        self.count = 0
        self.mean = 0.
        self.PCK_counts = np.zeros(len(PCK_THRES))
        return
  
    def update(self, prediction, meta_data, ground_truth=None, logger=None):
        """
        the prediction and transformation parameters in meta_data are used.
        """    
        avg_acc, cnt, others = get_distance_src(prediction, 
                                                meta_data,
                                                arg_max=self.arg_max,
                                                image_size=self.image_size)       
        self.mean = (self.mean * self.count + cnt * avg_acc) / (self.count + cnt)
        self.count += cnt
        self.PCK_counts += others['correct_cnt']
        return 
    
    def report(self, logger):
        logger.info("Ealuaton Results:")
        msg = 'Error type: {error_type:s}\t' \
              'MPJPE: {MPJPE}\t'.format(
                      error_type = self.name,
                      MPJPE = self.mean)     
        logger.info(msg)        
        for idx, value in enumerate(self.PCK_counts):
            PCK = value / self.count
            logger.info('PCK at threshold {:.2f}: {:.3f}'.format(PCK_THRES[idx], PCK))        
        return

def update_statistics(self, update, num_data, name_str):
    old_count = getattr(self, 'count'+name_str)
    old_mean = getattr(self, 'mean'+name_str)
    old_max = getattr(self, 'max'+name_str)
    old_min = getattr(self, 'min'+name_str)
    new_mean = (old_count * old_mean + np.sum(update, axis=0)) / (old_count + num_data) 
    new_count = old_count + num_data
    new_max = np.maximum(old_max, update.max(axis=0))
    new_min = np.minimum(old_min, update.min(axis=0))
    setattr(self, 'mean'+name_str, new_mean)
    setattr(self, 'count'+name_str, new_count)
    setattr(self, 'max'+name_str, new_max)
    setattr(self, 'min'+name_str, new_min)    
    return

def update_rotation_error(self, 
                          prediction, 
                          ground_truth, 
                          meta_data=None, 
                          logger=None,
                          name_str='',
                          style='euler'):
    """
    get rotation error between two point clouds 
    """    
    num_data = len(prediction)
    prediction = prediction.reshape(num_data, -1, 3)
    ground_truth = ground_truth.reshape(num_data, -1, 3)
    if style == 'euler':
        results = -np.ones((num_data, 3))
    for data_idx in range(num_data):
        R, T = ltr.compute_rigid_transform(prediction[data_idx].T, 
                                           ground_truth[data_idx].T
                                           )
        if style == 'euler':
            results[data_idx] = np.abs(Rotation.from_matrix(R).as_euler('xyz', 
                                                                        degrees=True
                                                                        )
                                       )
        else:
            raise NotImplementedError
    update_statistics(self, results, num_data, name_str)
    return

def update_joints_3d_error(self, 
                           prediction, 
                           ground_truth, 
                           meta_data=None, 
                           logger=None,
                           name_str='',
                           style='direct'
                           ):
    # squared error between prediction and expected output
    ground_truth = ground_truth.reshape(len(ground_truth), -1, 3)
    prediction = prediction.reshape(len(prediction), -1, 3)
    num_joints = prediction.shape[1]
    if style == 'procrustes':
        # Apply procrustes alignment if asked to do so
        for j in range(len(prediction)):
            gt  = ground_truth[j]
            out = prediction[j]
            _, Z, T, b, c = compute_similarity_transform(gt, out, compute_optimal_scale=True)
            out = (b * out.dot(T)) + c
            prediction[j] = np.reshape(out, [num_joints, 3])
    sqerr = (ground_truth - prediction)**2 
    distance = np.sqrt(np.sum(sqerr, axis=2))        
    num_data = len(prediction)
    update_statistics(self, distance, num_data, name_str)    
    # provide detailed L1 errors if there is only one joint
    if num_joints == 1:
        error_xyz = np.abs(ground_truth - prediction)
        update_statistics(self, error_xyz, num_data, name_str + '_xyz')
    return    

class RotationError3D():
    def __init__(self, cfgs):
        self.name = 'Rotation error'
        self.style = cfgs['metrics']['R3D']['style']
        self.count = 0
        if self.style == 'euler':
            self.mean = np.zeros((3))
            self.max = -np.ones((3))
            self.min = np.ones((3))*1e16
        return
    
    def update(self, prediction, ground_truth, meta_data=None, logger=None):
        """
        get rotation error between two point clouds 
        """    
        update_rotation_error(self, 
                              prediction, 
                              ground_truth, 
                              meta_data=meta_data, 
                              logger=logger,
                              style=self.style
                              )
        return 
    
    def report(self, logger):
        msg = 'Error type: {error_type:s}\t' \
              'Mean error: {mean_error}\t' \
              'Max error: {max_error}\t' \
              'Min error: {min_error}\t'.format(
                      error_type = self.name,
                      mean_error= self.mean, 
                      max_error= self.max,
                      min_error= self.min
                      )     
        logger.info(msg)        
        return
    
class JointDistance3D():
    def __init__(self, cfgs):
        self.name = 'Joint distance'
        self.style = cfgs['metrics']['JD3D']['style']
        self.num_joints = int(cfgs['FCModel']['output_size']/3)
        self.count = 0
        if self.style in ['direct', 'procrustes']:
            self.mean = np.zeros((self.num_joints))
            self.max = -np.ones((self.num_joints))
            self.min = np.ones((self.num_joints))*1e16
        else:
            raise NotImplementedError
        return
  
    def update(self, prediction, ground_truth, meta_data=None, logger=None):
        """
        get Euclidean distance between two point clouds 
        """    
        update_joints_3d_error(self, 
                               prediction,
                               ground_truth,
                               meta_data=meta_data,
                               logger=logger,
                               name_str='',
                               style=self.style
                               )        
        return 
    
    def report(self, logger):
        MPJPE = self.mean.sum() / self.num_joints
        msg = 'Error type: {error_type:s}\t' \
              'MPJPE: {MPJPE}\t' \
              'Mean error for each joint: {mean_error}\t' \
              'Max error for each joint: {max_error}\t' \
              'Min error for each joint: {min_error}\t'.format(
                      error_type = self.name,
                      MPJPE = MPJPE,
                      mean_error= self.mean, 
                      max_error= self.max,
                      min_error= self.min
                      )     
        logger.info(msg)        
        return

class RError3D():
    def __init__(self, cfgs, num_joints):
        """
        Relative shape error
        The point cloud should have a format [shape_relative_to_root]
        """           
        self.name = 'RError3D'
        self.T_style = cfgs['metrics']['R3D']['T_style']
        self.R_style = cfgs['metrics']['R3D']['R_style']
        if cfgs['dataset']['3d_kpt_sample_style'] == 'bbox9': 
            self.num_joints = num_joints - 1 # discount the root joint
        else:
            raise NotImplementedError
        self.count_rT = self.count_R = 0
        # translation error of the shape relative to the root
        self.mean_rT = np.zeros((self.num_joints))
        self.max_rT = -np.ones((self.num_joints))
        self.min_rT = np.ones((self.num_joints))*1e16            
        # relative rotation between the ground truth shape and predicted shape
        self.mean_R = np.zeros((3))
        self.max_R = -np.ones((3))
        self.min_R = np.ones((3))*1e16            
        return
  
    def update(self, prediction, ground_truth, meta_data=None, logger=None):
        update_joints_3d_error(self, 
                               prediction=prediction,
                               ground_truth=ground_truth,
                               meta_data=meta_data,
                               logger=logger,
                               name_str='_rT',
                               style=self.T_style
                               )
        update_rotation_error(self,
                              prediction=prediction,
                              ground_truth=ground_truth,
                              meta_data=meta_data,
                              logger=logger,
                              name_str='_R',
                              style=self.R_style
                              )        
        return 
    
    def report(self, logger):
        MPJPE = self.mean_rT.sum() / self.num_joints
        msg = 'Error type: {error_type:s}\t' \
              'MPJPE of the shape relative to the root:\t' \
              'MPJPE: {MPJPE}\t' \
              'Rotation error of the shape relative to the root:\t' \
              'Mean error: {mean_R}\t' \
              'Max error: {max_R}\t' \
              'Min error: {min_R}\t'.format(
                  error_type = self.name,
                  MPJPE = MPJPE,
                  mean_R = self.mean_R,
                  max_R = self.max_R,
                  min_R = self.min_R
                  )     
        logger.info(msg)        
        return
    
class RTError3D():
    def __init__(self, cfgs, num_joints):
        """
        Rotation and translation error combined.
        The point cloud should have a format [root, shape_relative_to_root]
        """           
        self.name = 'RTError3D'
        self.T_style = cfgs['metrics']['RTError3D']['T_style']
        self.R_style = cfgs['metrics']['RTError3D']['R_style']
        if cfgs['dataset']['3d_kpt_sample_style'] == 'bbox9': 
            self.num_joints = num_joints - 1 # discount the root joint
        else:
            raise NotImplementedError
        self.count_T = self.count_T_xyz = self.count_rT = self.count_R = 0
        if self.T_style in ['direct', 'procrustes']:
            # translation error of the root vector
            self.mean_T = np.zeros((1))
            # L1 error for each component
            self.mean_T_xyz = np.zeros((3))
            self.max_T = -np.ones((1))
            self.max_T_xyz = -np.ones((3))
            self.min_T = np.ones((1))*1e16
            self.min_T_xyz = np.ones((3))*1e16
            # translation error of the shape relative to the root
            self.mean_rT = np.zeros((self.num_joints))
            self.max_rT = -np.ones((self.num_joints))
            self.min_rT = np.ones((self.num_joints))*1e16            
        else:
            raise NotImplementedError
        # relative rotation between the ground truth shape and predicted shape
        self.mean_R = np.zeros((3))
        self.max_R = -np.ones((3))
        self.min_R = np.ones((3))*1e16            
        return
  
    def update(self, prediction, ground_truth, meta_data=None, logger=None):
        update_joints_3d_error(self, 
                               prediction=prediction[:, :3],
                               ground_truth=ground_truth[:, :3],
                               meta_data=meta_data,
                               logger=logger,
                               name_str='_T',
                               style=self.T_style
                               )
        update_joints_3d_error(self, 
                               prediction=prediction[:, 3:],
                               ground_truth=ground_truth[:, 3:],
                               meta_data=meta_data,
                               logger=logger,
                               name_str='_rT',
                               style=self.T_style
                               )
        update_rotation_error(self,
                              prediction=prediction[:, 3:],
                              ground_truth=ground_truth[:, 3:],
                              meta_data=meta_data,
                              logger=logger,
                              name_str='_R',
                              style=self.R_style
                              )        
        return 
    
    def report(self, logger):
        MPJPE = self.mean_rT.sum() / self.num_joints
        msg = 'Error type: {error_type:s}\t' \
              'Translation error of the root:\t' \
              'Mean error: {mean_T}\t' \
              'Max error: {max_T}\t' \
              'Min error: {min_T}\t' \
              'Translation error of the root in three directions:\t' \
              'Mean error (L1): {mean_T_xyz}\t' \
              'MPJPE of the shape relative to the root:\t' \
              'MPJPE: {MPJPE}\t' \
              'Rotation error of the shape relative to the root:\t' \
              'Mean error: {mean_R}\t' \
              'Max error: {max_R}\t' \
              'Min error: {min_R}\t'.format(
                  error_type = self.name,
                  MPJPE = MPJPE,
                  mean_T = self.mean_T, 
                  max_T = self.max_T,
                  min_T = self.min_T,
                  mean_T_xyz = self.mean_T_xyz,
                  mean_R = self.mean_R,
                  max_R = self.max_R,
                  min_R = self.min_R)     
        logger.info(msg)        
        return
    
class Evaluator():
    def __init__(self, metrics, cfgs=None, num_joints=9):
        """
        metrics is a list of strings specifying what metrics to use
        """
        self.metrics = []
        for metric in metrics:
            self.metrics.append(eval(metric + '(cfgs=cfgs, num_joints=num_joints)'))
        return
    
    def update(self, 
               prediction, 
               ground_truth=None,
               meta_data=None,
               logger=None
               ):
        """
        update evaluation with a new batch of prediction and ground truth
        """        
        for metric in self.metrics:
            metric.update(prediction, 
                          ground_truth=ground_truth,
                          meta_data=meta_data,
                          logger=logger
                          )
        return 
    
    def report(self, logger):
        for metric in self.metrics:
            metric.report(logger)       
        return