"""
Coordinate transformation functions.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import numpy as np
import cv2

def move_to(points, xyz=np.zeros((1,3))):
    # points of shape [n_points, 3]
    centroid = points.mean(axis=0, keepdims=True)
    return points - (centroid - xyz)

def world_to_camera_frame(P, R, T):
    """
    Convert points from world to camera coordinates
    
    P: Nx3 3d points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    
    Returns
    X_cam: Nx3 3d points in camera coordinates
    """
    assert len(P.shape) == 2
    assert P.shape[1] == 3
    X_cam = R.dot( P.T - T ) # rotate and translate
    return X_cam.T

def camera_to_world_frame(P, R, T):
    """
    Inverse of world_to_camera_frame

    P: Nx3 points in camera coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    
    Returns
    X_cam: Nx3 points in world coordinates
    """
    assert len(P.shape) == 2
    assert P.shape[1] == 3
    X_cam = R.T.dot( P.T ) + T # rotate and translate
    return X_cam.T

def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420
    
    Args
      X: array NxM of targets, with N number of points and M point dimensionality
      Y: array NxM of inputs
      compute_optimal_scale: whether we compute optimal scale or force it to be 1
    
    Returns:
      d: squared error after transformation
      Z: transformed Y
      T: computed rotation
      b: scaling
      c: translation
    """
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()
    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY
    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:,-1] *= np.sign( detT )
    s[-1]   *= np.sign( detT )
    T = np.dot(V, U.T)
    traceTA = s.sum()
    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX
    c = muX - b*np.dot(muY, T)
    return d, Z, T, b, c

def compute_rigid_transform(X, Y, W=None, verbose=False):
    """
    A least-sqaure estimate of rigid transformation by SVD.
    
    Reference: https://content.sakai.rutgers.edu/access/content/group/
    7bee3f05-9013-4fc2-8743-3c5078742791/material/svd_ls_rotation.pdf
    
    X, Y: [d, N] N data points of dimention d
    W: [N, ] optional weight (importance) matrix for N data points
    """    
    assert len(X) == len(Y)
    assert (W is None) or (len(W.shape) in [1, 2])
    # find mean column wise
    centroid_X = np.mean(X, axis=1, keepdims=True)
    centroid_Y = np.mean(Y, axis=1, keepdims=True)
    # subtract mean
    Xm = X - centroid_X
    Ym = Y - centroid_Y
    if W is None:
        H = Xm @ Ym.T
    else:
        W = np.diag(W) if len(W.shape) == 1 else W
        H = Xm @ W @ Ym.T
    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        # special reflection case
        if verbose:
            print("det(R) < R, reflection detected!, correcting for it ...\n");
        # the global minimizer with a orthogonal transformation is not possible
        # the next best transformation is chosen
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    t = -R @ centroid_X + centroid_Y
    return R, t

def procrustes_transform(X, Y):
    """
    Compute a rigid transformation trans() from X to Y and return trans(X)
    """
    R, t = compute_rigid_transform(X, Y)
    return R @ X + t

def pnp_refine(prediction, observation, intrinsics, dist_coeffs):
    """
    Refine 3D prediction with observed image projection based on  the PnP algorithm.
    """
    (success, R, T) = cv2.solvePnP(prediction,
                                   observation,
                                   intrinsics,
                                   dist_coeffs,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        print('PnP failed.')
        return prediction
    else:
        refined_prediction = cv2.Rodrigues(R)[0] @ prediction.T + T    
        return refined_prediction