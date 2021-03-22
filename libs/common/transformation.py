# -*- coding: utf-8 -*-
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
    Args
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
    """Inverse of world_to_camera_frame
    Args
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

def spherical_to_tangential(rThetaPhi):
    # based on spherical coordinates, calculate tangential vectors and 
    # normal vectors on a sphere in Cartesian coordinate
    # rThetaPhi is of shape [N, 3] recording the sphereical coordinates
    tangential = np.zeros((len(rThetaPhi), 3, 2))
    normal = np.zeros((len(rThetaPhi), 3, 1))
    theta = rThetaPhi[:, 1]
    phi = rThetaPhi[:, 2]
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    # the first tangential vector goes along the longitude
    tangential[:, 0, 0] = -sintheta
    tangential[:, 1, 0] = costheta
    # the second tangential vector goes along the lattitude
    tangential[:, 0, 1] = sinphi * costheta
    tangential[:, 1, 1] = sinphi * sintheta
    tangential[:, 2, 1] = -cosphi
    # normal vector
    normal[:, 0, 0] = -cosphi * costheta
    normal[:, 1, 0] = -cosphi * sintheta
    normal[:, 2, 0] = -sinphi
    return normal, tangential

def to_spherical(xyz):
    """
    convert from Cartisian coordinate to spherical coordinate
    theta: [-pi, pi]
    phi: [-pi/2, pi/2]
    note that xyz should be float number
    """
    # return in r, phi, and theta (elevation angle from z axis down)
    return_value = np.zeros(xyz.shape, dtype=xyz.dtype)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    return_value[:,0] = np.sqrt(xy + xyz[:,2]**2) # r      
    return_value[:,1] = np.arctan2(xyz[:,1], xyz[:,0]) # theta
    return_value[:,2] = np.arctan2(xyz[:,2], np.sqrt(xy)) # phi
    return return_value

def to_xyz(rthetaphi):
    """
    convert from spherical coordinate to Cartisian coordinate
    theta: [0, 2*pi] or [-pi, pi]
    phi: [-pi/2, pi/2]
    """
    return_value = np.zeros(rthetaphi.shape, dtype=rthetaphi.dtype)
    sintheta = np.sin(rthetaphi[:,1])
    costheta = np.cos(rthetaphi[:,1])
    sinphi = np.sin(rthetaphi[:,2])
    cosphi = np.cos(rthetaphi[:,2])
    return_value[:,0] = rthetaphi[:,0]*costheta*cosphi # x
    return_value[:,1] = rthetaphi[:,0]*sintheta*cosphi # y
    return_value[:,2] = rthetaphi[:,0]*sinphi #z
    return return_value

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

def compute_rigid_transform(X, Y, verbose=False):
    assert len(X) == len(Y) and X.shape[0] == 3
    # find mean column wise
    centroid_X = np.mean(X, axis=1, keepdims=True)
    centroid_Y = np.mean(Y, axis=1, keepdims=True)
    # subtract mean
    Xm = X - centroid_X
    Ym = Y - centroid_Y
    # dot is matrix multiplication for array
    H = Xm @ Ym.T
    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        if verbose:
            print("det(R) < R, reflection detected!, correcting for it ...\n");
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    t = -R @ centroid_X + centroid_Y
    return R, t

def procrustes_transform(X, Y):
    # compute a rigid transformation trans() from X to Y and return trans(X)
    R, t = compute_rigid_transform(X, Y)
    return R @ X + t

def pnp_refine(prediction, observation, intrinsics, dist_coeffs):
    # refine 3D prediction with observed key-points based on PnP algorithm
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
    
    