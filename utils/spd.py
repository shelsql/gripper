"""
    Evaluation-related codes are modified from
    https://github.com/hughw19/NOCS_CVPR2019
"""
import logging
import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle
from tqdm import tqdm
import time
import torch
from torch.masked import masked_tensor, as_masked_tensor
import warnings

# Disable prototype warnings and such
warnings.filterwarnings(action='ignore', category=UserWarning)

def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    return logger


def load_obj(path_to_file):
    """ Load obj file.

    Args:
        path_to_file: path

    Returns:
        vertices: ndarray
        faces: ndarray, index of triangle vertices

    """
    vertices = []
    faces = []
    with open(path_to_file, 'r') as f:
        for line in f:
            if line[:2] == 'v ':
                vertex = line[2:].strip().split(' ')
                vertex = [float(xyz) for xyz in vertex]
                vertices.append(vertex)
            elif line[0] == 'f':
                face = line[1:].replace('//', '/').strip().split(' ')
                face = [int(idx.split('/')[0])-1 for idx in face]
                faces.append(face)
            else:
                continue
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    return vertices, faces

def load_ply(path_to_file):
    """ Load ply file.

    Args:
        path_to_file: path

    Returns:
        vertices: ndarray
        faces: ndarray, index of triangle vertices

    """
    vertices = []
    faces = []
    with open(path_to_file, 'r') as f:
        lines = f.readlines()
        vertex_section = False
        face_section = False
        for line in lines:
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[2])
                vertex_section = True
            elif line.startswith("element face"):
                num_faces = int(line.split()[2])
                face_section = True
            elif line.startswith("end_header"):
                vertex_section = False
                face_section = False
            elif vertex_section:
                vertex = [float(x) for x in line.split()]
                vertices.append(vertex)
            elif face_section:
                face = [int(x) for x in line.split()[1:]]
                faces.append(face)
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    return vertices, faces


def create_sphere():
    # 642 verts, 1280 faces,
    verts, faces = load_obj('assets/sphere_mesh_template.obj')
    return verts, faces


def random_point(face_vertices):
    """ Sampling point using Barycentric coordiante.

    """
    r1, r2 = np.random.random(2)
    sqrt_r1 = np.sqrt(r1)
    point = (1 - sqrt_r1) * face_vertices[0, :] + \
        sqrt_r1 * (1 - r2) * face_vertices[1, :] + \
        sqrt_r1 * r2 * face_vertices[2, :]

    return point


def pairwise_distance(A, B):
    """ Compute pairwise distance of two point clouds.point

    Args:
        A: n x 3 numpy array
        B: m x 3 numpy array

    Return:
        C: n x m numpy array

    """
    diff = A[:, :, None] - B[:, :, None].T
    C = np.sqrt(np.sum(diff**2, axis=1))

    return C

def pairwise_distances_torch(A, B):
    diff = A.unsqueeze(1) - B.unsqueeze(0)
    C = torch.sqrt(torch.sum(diff**2, dim=2))

    return C


def uniform_sample(vertices, faces, n_samples, with_normal=False):
    """ Sampling points according to the area of mesh surface.

    """
    sampled_points = np.zeros((n_samples, 3), dtype=float)
    normals = np.zeros((n_samples, 3), dtype=float)
    faces = vertices[faces]
    vec_cross = np.cross(faces[:, 1, :] - faces[:, 0, :],
                         faces[:, 2, :] - faces[:, 0, :])
    face_area = 0.5 * np.linalg.norm(vec_cross, axis=1)
    cum_area = np.cumsum(face_area)
    for i in range(n_samples):
        face_id = np.searchsorted(cum_area, np.random.random() * cum_area[-1])
        sampled_points[i] = random_point(faces[face_id, :, :])
        normals[i] = vec_cross[face_id]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    if with_normal:
        sampled_points = np.concatenate((sampled_points, normals), axis=1)
    return sampled_points


def farthest_point_sampling(points, n_samples):
    """ Farthest point sampling.

    """
    selected_pts = np.zeros((n_samples,), dtype=int)
    dist_mat = pairwise_distance(points, points)
    # start from first point
    pt_idx = 0
    dist_to_set = dist_mat[:, pt_idx]
    for i in range(n_samples):
        selected_pts[i] = pt_idx
        dist_to_set = np.minimum(dist_to_set, dist_mat[:, pt_idx])
        pt_idx = np.argmax(dist_to_set)
    return selected_pts


def sample_points_from_mesh(path, n_pts, with_normal=False, fps=False, ratio=2):
    """ Uniformly sampling points from mesh model.

    Args:
        path: path to OBJ file.
        n_pts: int, number of points being sampled.
        with_normal: return points with normal, approximated by mesh triangle normal
        fps: whether to use fps for post-processing, default False.
        ratio: int, if use fps, sample ratio*n_pts first, then use fps to sample final output.

    Returns:
        points: n_pts x 3, n_pts x 6 if with_normal = True

    """
    if path.endswith("obj"):
        vertices, faces = load_obj(path)
    elif path.endswith("ply"):
        vertices, faces = load_ply(path)
    if fps:
        points = uniform_sample(vertices, faces, ratio*n_pts, with_normal)
        pts_idx = farthest_point_sampling(points[:, :3], n_pts)
        points = points[pts_idx]
    else:
        points = uniform_sample(vertices, faces, n_pts, with_normal)
    return points


def load_depth(img_path):
    """ Load depth image from img_path. """
    depth_path = img_path + '_depth.png'
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16


def get_bbox(bbox):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    img_width = 480
    img_length = 640
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def compute_sRT_errors(sRT1, sRT2):
    """
    Args:
        sRT1: [4, 4]. homogeneous affine transformation
        sRT2: [4, 4]. homogeneous affine transformation

    Returns:
        R_error: angle difference in degree,
        T_error: Euclidean distance
        IoU: relative scale error

    """
    try:
        assert np.array_equal(sRT1[3, :], sRT2[3, :])
        assert np.array_equal(sRT1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(sRT1[3, :], sRT2[3, :])

    s1 = np.cbrt(np.linalg.det(sRT1[:3, :3]))
    R1 = sRT1[:3, :3] / s1
    T1 = sRT1[:3, 3]
    s2 = np.cbrt(np.linalg.det(sRT2[:3, :3]))
    R2 = sRT2[:3, :3] / s2
    T2 = sRT2[:3, 3]
    R12 = R1 @ R2.transpose()
    R_error = np.arccos(np.clip((np.trace(R12)-1)/2, -1.0, 1.0)) * 180 / np.pi
    T_error = np.linalg.norm(T1 - T2)
    IoU = np.abs(s1 - s2) / s2

    return R_error, T_error, IoU


############################################################
#  Evaluation
############################################################

def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]

    """
    bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, -size[2] / 2]]) + shift
    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def transform_coordinates_3d(coordinates, sRT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]

    Returns:
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates


def compute_3d_IoU(sRT_1, sRT_2, size_1, size_2, class_name_1, class_name_2, handle_visibility):
    """ Computes IoU overlaps between two 3D bboxes. """
    def asymmetric_3d_iou(sRT_1, sRT_2, size_1, size_2):
        noc_cube_1 = get_3d_bbox(size_1, 0)
        bbox_3d_1 = transform_coordinates_3d(noc_cube_1, sRT_1)
        noc_cube_2 = get_3d_bbox(size_2, 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, sRT_2)

        bbox_1_max = np.amax(bbox_3d_1, axis=0)
        bbox_1_min = np.amin(bbox_3d_1, axis=0)
        bbox_2_max = np.amax(bbox_3d_2, axis=0)
        bbox_2_min = np.amin(bbox_3d_2, axis=0)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min)
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)

        # intersections and union
        if np.amin(overlap_max - overlap_min) < 0:
            intersections = 0
        else:
            intersections = np.prod(overlap_max - overlap_min)
        union = np.prod(bbox_1_max - bbox_1_min) + np.prod(bbox_2_max - bbox_2_min) - intersections
        overlaps = intersections / union
        return overlaps

    if sRT_1 is None or sRT_2 is None:
        return -1

    if (class_name_1 in ['bottle', 'bowl', 'can'] and class_name_1 == class_name_2) or \
        (class_name_1 == 'mug' and class_name_1 == class_name_2 and handle_visibility==0):
        def y_rotation_matrix(theta):
            return np.array([[ np.cos(theta), 0, np.sin(theta), 0],
                             [ 0,             1, 0,             0],
                             [-np.sin(theta), 0, np.cos(theta), 0],
                             [ 0,             0, 0,             1]])
        n = 20
        max_iou = 0
        for i in range(n):
            rotated_RT_1 = sRT_1 @ y_rotation_matrix(2 * math.pi * i / float(n))
            max_iou = max(max_iou, asymmetric_3d_iou(rotated_RT_1, sRT_2, size_1, size_2))
    else:
        max_iou = asymmetric_3d_iou(sRT_1, sRT_2, size_1, size_2)

    return max_iou


def compute_IoU_matches(gt_class_ids, gt_sRT, gt_size, gt_handle_visibility,
                        pred_class_ids, pred_sRT, pred_size, pred_scores,
                        synset_names, iou_3d_thresholds, score_threshold=0):
    """ Find matches between NOCS prediction and ground truth instances.

    Args:
        size: 3D bounding box size
        bboxes: 2D bounding boxes

    Returns:
        gt_matches: 2-D array. For each GT box it has the index of the matched predicted box.
        pred_matches: 2-D array. For each predicted box, it has the index of the matched ground truth box.
        overlaps: IoU overlaps.
        indices:

    """
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)
    if num_pred:
        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]
        pred_class_ids = pred_class_ids[indices].copy()
        pred_size = pred_size[indices].copy()
        pred_sRT = pred_sRT[indices].copy()
    # compute IoU overlaps [pred_bboxs gt_bboxs]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j] = compute_3d_IoU(pred_sRT[i], gt_sRT[j], pred_size[i, :], gt_size[j],
                synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]], gt_handle_visibility[j])
    # loop through predictions and find matching ground truth boxes
    num_iou_3d_thres = len(iou_3d_thresholds)
    pred_matches = -1 * np.ones([num_iou_3d_thres, num_pred])
    gt_matches = -1 * np.ones([num_iou_3d_thres, num_gt])
    for s, iou_thres in enumerate(iou_3d_thresholds):
        for i in range(indices.shape[0]):
            # Find best matching ground truth box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low scores
            low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]
            # 3. Find the match
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                if gt_matches[s, j] > -1:
                    continue
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                if iou < iou_thres:
                    break
                # Do we have a match?
                if not pred_class_ids[i] == gt_class_ids[j]:
                    continue
                if iou > iou_thres:
                    gt_matches[s, j] = i
                    pred_matches[s, i] = j
                    break
    return gt_matches, pred_matches, overlaps, indices


def compute_RT_errors(sRT_1, sRT_2, ):
    """
    Args:
        sRT_1: [4, 4]. homogeneous affine transformation
        sRT_2: [4, 4]. homogeneous affine transformation

    Returns:
        theta: angle difference of R in degree
        shift: l2 difference of T in centimeter
    """
    # make sure the last row is [0, 0, 0, 1]
    if sRT_1 is None or sRT_2 is None:
        return -1
    try:
        assert np.array_equal(sRT_1[3, :], sRT_2[3, :])
        assert np.array_equal(sRT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(sRT_1[3, :], sRT_2[3, :])
        exit()

    R1 = sRT_1[:3, :3] / np.cbrt(np.linalg.det(sRT_1[:3, :3]))
    T1 = sRT_1[:3, 3]
    R2 = sRT_2[:3, :3] / np.cbrt(np.linalg.det(sRT_2[:3, :3]))
    T2 = sRT_2[:3, 3]
    # symmetric when rotating around y-axis
    # if synset_names[class_id] in ['bottle', 'can', 'bowl'] or \
    #     (synset_names[class_id] == 'mug' and handle_visibility == 0):
    #     y = np.array([0, 1, 0])
    #     y1 = R1 @ y
    #     y2 = R2 @ y
    #     cos_theta = y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))
    # else:
    R = R1 @ R2.transpose()
    cos_theta = (np.trace(R) - 1) / 2

    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result

def compute_R_errors_batch(sRT_1, sRT_2):
    """
    Args:
        sRT_1: [B, 4, 4]. batch of homogeneous affine transformations
        sRT_2: [B, 4, 4]. batch of homogeneous affine transformations

    Returns:
        theta: [B]. angle difference of R in degree
        shift: [B]. l2 difference of T in centimeter
    """
    # make sure the last row is [0, 0, 0, 1]
    # if sRT_1 is None or sRT_2 is None:
    #     return -1
    # try:
    #     assert np.all(np.array_equal(sRT_1[:, 3, :], sRT_2[:, 3, :]))
    #     assert np.all(np.array_equal(sRT_1[:, 3, :], np.tile(np.array([0, 0, 0, 1]),(sRT_1.shape[0],1))))
    # except AssertionError:
    #     print(sRT_1[:, 3, :], sRT_2[:, 3, :])
    #     exit()

    R1 = sRT_1[:, :3, :3] / np.cbrt(np.linalg.det(sRT_1[:, :3, :3])).reshape(-1,1,1)
    # T1 = sRT_1[:, :3, 3]
    R2 = sRT_2[:, :3, :3] / np.cbrt(np.linalg.det(sRT_2[:, :3, :3])).reshape(-1,1,1)
    # T2 = sRT_2[:, :3, 3]
    # symmetric when rotating around y-axis
    # if synset_names[class_id] in ['bottle', 'can', 'bowl'] or \
    #     (synset_names[class_id] == 'mug' and handle_visibility == 0):
    #     y = np.array([0, 1, 0])
    #     y1 = R1 @ y
    #     y2 = R2 @ y
    #     cos_theta = y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))
    # else:
    R1 = np.repeat(R1[:,np.newaxis,...],R2.shape[0],axis=1)
    R2 = np.repeat(R2[np.newaxis, ...], R1.shape[0], axis=0)
    R = np.matmul(R1, R2.transpose(0,1,3,2))
    cos_theta = (np.trace(R, axis1=2, axis2=3) - 1) / 2

    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
    # shift = np.linalg.norm(T1 - T2, axis=1) * 100
    # result = (theta, shift)

    return theta

def compute_outplane_rotation_errors(sRT_1, sRT_2):
    '''
        sRT_1: [B, 4, 4]. batch of homogeneous affine transformations
        sRT_2: [B, 4, 4]. batch of homogeneous affine transformations
        # 仅计算非in-plane rotation之间的误差；变换矩阵->旋转向量，之后再求解)(好像不对！)
        应该把R分解为in plane 和 out plane rotation，然后仅计算out plane rotation之间的距离
        为什么不直接沿用原来的方法，最后再
    '''
    # # make sure the last row is [0, 0, 0, 1]
    # if sRT_1 is None or sRT_2 is None:
    #     return -1
    # try:
    #     assert np.all(np.array_equal(sRT_1[:, 3, :], sRT_2[:, 3, :]))
    #     assert np.all(np.array_equal(sRT_1[:, 3, :], np.tile(np.array([0, 0, 0, 1]),(sRT_1.shape[0],1))))
    # except AssertionError:
    #     print(sRT_1[:, 3, :], sRT_2[:, 3, :])
    #     exit()

    R1 = sRT_1[:, :3, :3] / np.cbrt(np.linalg.det(sRT_1[:, :3, :3])).reshape(-1,1,1)    # b,3,3
    R2 = sRT_2[:, :3, :3] / np.cbrt(np.linalg.det(sRT_2[:, :3, :3])).reshape(-1,1,1)    # b,3,3

    R1_out_plane,_ = decompose_rotation_matrices(R1)
    R2_out_plane,_ = decompose_rotation_matrices(R2)


    R1_out_plane = np.repeat(R1_out_plane[:,np.newaxis,...],R2_out_plane.shape[0],axis=1)
    R2_out_plane = np.repeat(R2_out_plane[np.newaxis, ...], R1_out_plane.shape[0], axis=0)
    R = np.matmul(R1_out_plane, R2_out_plane.transpose(0,1,3,2))
    cos_theta = (np.trace(R, axis1=2, axis2=3) - 1) / 2

    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi

    return theta


# def batch_rotation_matrix_to_vector(R_batch):
#     # 如果几乎没有旋转的话，就return(0,0,1)
#
#     # Calculate the trace of each rotation matrix
#     trace = np.trace(R_batch, axis1=1, axis2=2)
#
#     # Calculate the rotation angle theta for each matrix
#     theta = np.arccos((trace - 1) / 2.0)
#
#     # Handle the case when theta is close to 0 (identity rotation)
#     sin_theta = np.sin(theta)
#     sin_theta[sin_theta == 0] = 1  # Avoid division by zero
#
#     # Calculate the rotation axis
#     ux = (R_batch[:, 2, 1] - R_batch[:, 1, 2]) / (2 * sin_theta)
#     uy = (R_batch[:, 0, 2] - R_batch[:, 2, 0]) / (2 * sin_theta)
#     uz = (R_batch[:, 1, 0] - R_batch[:, 0, 1]) / (2 * sin_theta)
#
#     # Stack the rotation axes to form the vector u
#     u = np.stack((ux, uy, uz), axis=1)
#
#     # Handle the case when theta is close to pi
#     mask = np.isclose(theta, np.pi)
#     if np.any(mask):
#         eigvals, eigvecs = np.linalg.eig(R_batch[mask])
#         axis = eigvecs[:, :, np.isclose(eigvals, 1)].real
#         u[mask] = axis[:, :, 0]
#
#     # Compute the rotation vector
#     rot_vecs = theta[:, np.newaxis] * u
#
#     # Handle the case when theta is close to 0 (identity rotation)
#     rot_vecs[np.isclose(theta, 0)] = 0
#
#     return rot_vecs


def decompose_rotation_matrices(R_batch):
    # R_batch: A batch of rotation matrices with shape (N, 3, 3), where N is the number of matrices

    # Calculate the angles theta for in-plane rotation (around z-axis)
    theta_batch = np.arctan2(R_batch[:, 1, 0], R_batch[:, 0, 0])

    # Construct the in-plane rotation matrices R_in (around z-axis)
    cos_theta = np.cos(theta_batch)
    sin_theta = np.sin(theta_batch)

    R_in_batch = np.zeros((R_batch.shape[0], 3, 3))
    R_in_batch[:, 0, 0] = cos_theta
    R_in_batch[:, 0, 1] = -sin_theta
    R_in_batch[:, 1, 0] = sin_theta
    R_in_batch[:, 1, 1] = cos_theta
    R_in_batch[:, 2, 2] = 1

    # Calculate the out-of-plane rotation matrices R_out
    R_in_inv_batch = np.transpose(R_in_batch, axes=(0, 2, 1))  # Since R_in is orthogonal, its inverse is its transpose
    R_out_batch = np.einsum('nij,njk->nik', R_batch, R_in_inv_batch)

    return R_out_batch, R_in_batch

def compute_RT_overlaps(gt_class_ids, gt_sRT, gt_handle_visibility, pred_class_ids, pred_sRT, synset_names):
    """ Finds overlaps between prediction and ground truth instances.

    Returns:
        overlaps:

    """
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    overlaps = np.zeros((num_pred, num_gt, 2))

    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j, :] = compute_RT_errors(pred_sRT[i], gt_sRT[j], gt_class_ids[j],
                                                  gt_handle_visibility[j], synset_names)
    return overlaps


def compute_RT_matches(overlaps, pred_class_ids, gt_class_ids, degree_thres_list, shift_thres_list):
    num_degree_thres = len(degree_thres_list)
    num_shift_thres = len(shift_thres_list)
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    pred_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_pred))
    gt_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_gt))

    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches

    assert num_pred == overlaps.shape[0]
    assert num_gt == overlaps.shape[1]
    assert overlaps.shape[2] == 2

    for d, degree_thres in enumerate(degree_thres_list):
        for s, shift_thres in enumerate(shift_thres_list):
            for i in range(num_pred):
                # Find best matching ground truth box
                # 1. Sort matches by scores from low to high
                sum_degree_shift = np.sum(overlaps[i, :, :], axis=-1)
                sorted_ixs = np.argsort(sum_degree_shift)
                # 2. Find the match
                for j in sorted_ixs:
                    # If ground truth box is already matched, go to next one
                    if gt_matches[d, s, j] > -1 or pred_class_ids[i] != gt_class_ids[j]:
                        continue
                    # If we reach IoU smaller than the threshold, end the loop
                    if overlaps[i, j, 0] > degree_thres or overlaps[i, j, 1] > shift_thres:
                        continue
                    gt_matches[d, s, j] = i
                    pred_matches[d, s, i] = j
                    break

    return gt_matches, pred_matches


def compute_ap_and_acc(pred_matches, pred_scores, gt_matches):
    # sort the scores from high to low
    assert pred_matches.shape[0] == pred_scores.shape[0]
    score_indices = np.argsort(pred_scores)[::-1]
    # pred_scores = pred_scores[score_indices]
    pred_matches = pred_matches[score_indices]
    precisions = np.cumsum(pred_matches > -1) / (np.arange(len(pred_matches)) + 1)
    recalls = np.cumsum(pred_matches > -1).astype(np.float32) / len(gt_matches)
    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])
    # compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    # accuracy
    acc = np.sum(pred_matches > -1) / len(pred_matches)

    return ap, acc


def compute_mAP(pred_results, out_dir, degree_thresholds=[180], shift_thresholds=[100],
                iou_3d_thresholds=[0.1], iou_pose_thres=0.1, use_matches_for_pose=False):
    """ Compute mean Average Precision.

    Returns:
        iou_aps:
        pose_aps:
        iou_acc:
        pose_acc:

    """
    synset_names = ['BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    num_classes = len(synset_names)
    degree_thres_list = list(degree_thresholds) + [360]
    num_degree_thres = len(degree_thres_list)
    shift_thres_list = list(shift_thresholds) + [100]
    num_shift_thres = len(shift_thres_list)
    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    if use_matches_for_pose:
        assert iou_pose_thres in iou_thres_list

    # pre-allocate more than enough memory
    iou_aps = np.zeros((num_classes + 1, num_iou_thres))
    iou_acc = np.zeros((num_classes + 1, num_iou_thres))
    iou_pred_matches_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_pred_scores_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_gt_matches_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_pred_count = [0 for _ in range(num_classes)]
    iou_gt_count = [0 for _ in range(num_classes)]

    pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_acc = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_pred_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_pred_scores_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_gt_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_pred_count = [0 for _ in range(num_classes)]
    pose_gt_count = [0 for _ in range(num_classes)]

    # loop over results to gather pred matches and gt matches for iou and pose metrics
    progress = 0
    for progress, result in enumerate(tqdm(pred_results)):
        gt_class_ids = result['gt_class_ids'].astype(np.int32)
        gt_sRT = np.array(result['gt_RTs'])
        gt_size = np.array(result['gt_scales'])
        gt_handle_visibility = result['gt_handle_visibility']

        pred_class_ids = result['pred_class_ids']
        pred_sRT = np.array(result['pred_RTs'])
        pred_size = result['pred_scales']
        pred_scores = result['pred_scores']

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue

        for cls_id in range(1, num_classes):
            # get gt and predictions in this class
            cls_gt_class_ids = gt_class_ids[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros(0)
            cls_gt_sRT = gt_sRT[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 4, 4))
            cls_gt_size = gt_size[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 3))
            if synset_names[cls_id] != 'mug':
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = gt_handle_visibility[gt_class_ids==cls_id] if len(gt_class_ids) else np.ones(0)

            cls_pred_class_ids = pred_class_ids[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
            cls_pred_sRT = pred_sRT[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 4, 4))
            cls_pred_size = pred_size[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 3))
            cls_pred_scores = pred_scores[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)

            # calculate the overlap between each gt instance and pred instance
            iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices = \
                compute_IoU_matches(cls_gt_class_ids, cls_gt_sRT, cls_gt_size, cls_gt_handle_visibility,
                                    cls_pred_class_ids, cls_pred_sRT, cls_pred_size, cls_pred_scores,
                                    synset_names, iou_thres_list)
            if len(iou_pred_indices):
                cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
                cls_pred_sRT = cls_pred_sRT[iou_pred_indices]
                cls_pred_scores = cls_pred_scores[iou_pred_indices]

            num_pred = iou_cls_pred_match.shape[1]
            pred_start = iou_pred_count[cls_id]
            pred_end = pred_start + num_pred
            iou_pred_count[cls_id] = pred_end
            iou_pred_matches_all[cls_id][:, pred_start:pred_end] = iou_cls_pred_match
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
            assert cls_pred_scores_tile.shape[1] == num_pred
            iou_pred_scores_all[cls_id][:, pred_start:pred_end] = cls_pred_scores_tile
            num_gt = iou_cls_gt_match.shape[1]
            gt_start = iou_gt_count[cls_id]
            gt_end = gt_start + num_gt
            iou_gt_count[cls_id] = gt_end
            iou_gt_matches_all[cls_id][:, gt_start:gt_end] = iou_cls_gt_match

            if use_matches_for_pose:
                thres_ind = list(iou_thres_list).index(iou_pose_thres)
                iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]
                cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_sRT = cls_pred_sRT[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
                cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
                cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)
                cls_gt_sRT = cls_gt_sRT[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
                cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)

            RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_sRT, cls_gt_handle_visibility,
                                              cls_pred_class_ids, cls_pred_sRT, synset_names)
            pose_cls_gt_match, pose_cls_pred_match = compute_RT_matches(RT_overlaps, cls_pred_class_ids, cls_gt_class_ids,
                                                                        degree_thres_list, shift_thres_list)
            num_pred = pose_cls_pred_match.shape[2]
            pred_start = pose_pred_count[cls_id]
            pred_end = pred_start + num_pred
            pose_pred_count[cls_id] = pred_end
            pose_pred_matches_all[cls_id][:, :, pred_start:pred_end] = pose_cls_pred_match
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
            assert cls_pred_scores_tile.shape[2] == num_pred
            pose_pred_scores_all[cls_id][:, :, pred_start:pred_end] = cls_pred_scores_tile
            num_gt = pose_cls_gt_match.shape[2]
            gt_start = pose_gt_count[cls_id]
            gt_end = gt_start + num_gt
            pose_gt_count[cls_id] = gt_end
            pose_gt_matches_all[cls_id][:, :, gt_start:gt_end] = pose_cls_gt_match

    # trim zeros
    for cls_id in range(num_classes):
        # IoU
        iou_pred_matches_all[cls_id] = iou_pred_matches_all[cls_id][:, :iou_pred_count[cls_id]]
        iou_pred_scores_all[cls_id] = iou_pred_scores_all[cls_id][:, :iou_pred_count[cls_id]]
        iou_gt_matches_all[cls_id] = iou_gt_matches_all[cls_id][:, :iou_gt_count[cls_id]]
        # pose
        pose_pred_matches_all[cls_id] = pose_pred_matches_all[cls_id][:, :, :pose_pred_count[cls_id]]
        pose_pred_scores_all[cls_id] = pose_pred_scores_all[cls_id][:, :, :pose_pred_count[cls_id]]
        pose_gt_matches_all[cls_id] = pose_gt_matches_all[cls_id][:, :, :pose_gt_count[cls_id]]

    # compute 3D IoU mAP
    for cls_id in range(1, num_classes):
        for s, iou_thres in enumerate(iou_thres_list):
            iou_aps[cls_id, s], iou_acc[cls_id, s] = compute_ap_and_acc(iou_pred_matches_all[cls_id][s, :],
                                                                        iou_pred_scores_all[cls_id][s, :],
                                                                        iou_gt_matches_all[cls_id][s, :])
    iou_aps[-1, :] = np.mean(iou_aps[1:-1, :], axis=0)
    iou_acc[-1, :] = np.mean(iou_acc[1:-1, :], axis=0)
    # compute pose mAP
    for i, degree_thres in enumerate(degree_thres_list):
        for j, shift_thres in enumerate(shift_thres_list):
            for cls_id in range(1, num_classes):
                cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
                cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
                cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]
                pose_aps[cls_id, i, j], pose_acc[cls_id, i, j] = compute_ap_and_acc(cls_pose_pred_matches_all,
                                                                                    cls_pose_pred_scores_all,
                                                                                    cls_pose_gt_matches_all)
            pose_aps[-1, i, j] = np.mean(pose_aps[1:-1, i, j])
            pose_acc[-1, i, j] = np.mean(pose_acc[1:-1, i, j])

    # save results to pkl
    result_dict = {}
    result_dict['iou_thres_list'] = iou_thres_list
    result_dict['degree_thres_list'] = degree_thres_list
    result_dict['shift_thres_list'] = shift_thres_list
    result_dict['iou_aps'] = iou_aps
    result_dict['pose_aps'] = pose_aps
    result_dict['iou_acc'] = iou_acc
    result_dict['pose_acc'] = pose_acc
    pkl_path = os.path.join(out_dir, 'mAP_Acc.pkl')
    with open(pkl_path, 'wb') as f:
        cPickle.dump(result_dict, f)
    return iou_aps, pose_aps, iou_acc, pose_acc


def plot_mAP(iou_aps, pose_aps, out_dir, iou_thres_list, degree_thres_list, shift_thres_list):
    """ Draw iou 3d AP vs. iou thresholds.
    """

    labels = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug', 'mean', 'nocs']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:olive', 'tab:purple', 'tab:red', 'tab:gray']
    styles = ['-', '-', '-', '-', '-', '-', '--', ':']

    fig, (ax_iou, ax_degree, ax_shift) = plt.subplots(1, 3, figsize=(8, 3.5))
    # IoU subplot
    ax_iou.set_title('3D IoU', fontsize=10)
    ax_iou.set_ylabel('Average Precision')
    ax_iou.set_ylim(0, 100)
    ax_iou.set_xlabel('Percent')
    ax_iou.set_xlim(0, 100)
    ax_iou.xaxis.set_ticks([0, 25, 50, 75, 100])
    ax_iou.grid()
    for i in range(1, iou_aps.shape[0]):
        ax_iou.plot(100*np.array(iou_thres_list), 100*iou_aps[i, :],
                    color=colors[i-1], linestyle=styles[i-1], label=labels[i-1])
    # rotation subplot
    ax_degree.set_title('Rotation', fontsize=10)
    ax_degree.set_ylim(0, 100)
    ax_degree.yaxis.set_ticklabels([])
    ax_degree.set_xlabel('Degree')
    ax_degree.set_xlim(0, 60)
    ax_degree.xaxis.set_ticks([0, 20, 40, 60])
    ax_degree.grid()
    for i in range(1, pose_aps.shape[0]):
        ax_degree.plot(np.array(degree_thres_list), 100*pose_aps[i, :len(degree_thres_list), -1],
                       color=colors[i-1], linestyle=styles[i-1], label=labels[i-1])
    # translation subplot
    ax_shift.set_title('Translation', fontsize=10)
    ax_shift.set_ylim(0, 100)
    ax_shift.yaxis.set_ticklabels([])
    ax_shift.set_xlabel('Centimeter')
    ax_shift.set_xlim(0, 10)
    ax_shift.xaxis.set_ticks([0, 5, 10])
    ax_shift.grid()
    for i in range(1, pose_aps.shape[0]):
        ax_shift.plot(np.array(shift_thres_list), 100*pose_aps[i, -1, :len(shift_thres_list)],
                      color=colors[i-1], linestyle=styles[i-1], label=labels[i-1])
    ax_shift.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(out_dir, 'mAP.png'))
    plt.close(fig)
    return


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]

    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def align_rotation(sRT):
    """ Align rotations for symmetric objects.
    Args:
        sRT: 4 x 4
    """
    s = np.cbrt(np.linalg.det(sRT[:3, :3]))
    R = sRT[:3, :3] / s
    T = sRT[:3, 3]

    theta_x = R[0, 0] + R[2, 2]
    theta_y = R[0, 2] - R[2, 0]
    r_norm = math.sqrt(theta_x**2 + theta_y**2)
    s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                      [0.0,            1.0,  0.0           ],
                      [theta_y/r_norm, 0.0,  theta_x/r_norm]])
    rotation = R @ s_map
    aligned_sRT = np.identity(4, dtype=np.float32)
    aligned_sRT[:3, :3] = s * rotation
    aligned_sRT[:3, 3] = T
    return aligned_sRT


def draw_bboxes(img, img_pts, color):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    # draw ground layer in darker color
    color_ground = (int(color[0]*0.3), int(color[1]*0.3), int(color[2]*0.3))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, 2)
    # draw pillars in minor darker color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, 2)
    # draw top layer in original color
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, 2)

    return img


def draw_detections(img, out_dir, data_name, img_id, intrinsics, pred_sRT, pred_size, pred_class_ids,
                    gt_sRT, gt_size, gt_class_ids, nocs_sRT, nocs_size, nocs_class_ids, draw_gt=True, draw_nocs=True):
    """ Visualize pose predictions.
    """
    out_path = os.path.join(out_dir, '{}_{}_pred.png'.format(data_name, img_id))

    # draw nocs results - BLUE color
    if draw_nocs:
        for i in range(nocs_sRT.shape[0]):
            if nocs_class_ids[i] in [1, 2, 4]:
                sRT = align_rotation(nocs_sRT[i, :, :])
            else:
                sRT = nocs_sRT[i, :, :]
            bbox_3d = get_3d_bbox(nocs_size[i, :], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            img = draw_bboxes(img, projected_bbox, (255, 0, 0))
    # darw ground truth - GREEN color
    if draw_gt:
        for i in range(gt_sRT.shape[0]):
            if gt_class_ids[i] in [1, 2, 4]:
                sRT = align_rotation(gt_sRT[i, :, :])
            else:
                sRT = gt_sRT[i, :, :]
            bbox_3d = get_3d_bbox(gt_size[i, :], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            img = draw_bboxes(img, projected_bbox, (0, 255, 0))
    # darw prediction - RED color
    for i in range(pred_sRT.shape[0]):
        if pred_class_ids[i] in [1, 2, 4]:
            sRT = align_rotation(pred_sRT[i, :, :])
        else:
            sRT = pred_sRT[i, :, :]
        bbox_3d = get_3d_bbox(pred_size[i, :], 0)
        transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
        img = draw_bboxes(img, projected_bbox, (0, 0, 255))

    cv2.imwrite(out_path, img)
    # cv2.imshow('vis', img)
    # cv2.waitKey(0)
    
##############################################

def depth_map_to_pointcloud(depth_map, mask, intrinsics):
    # Get dimensions
    H, W = depth_map.shape
    
    if mask is not None:
        depth_map[mask == 0] = -1
    
    # Unpack intrinsic matrix
    fx = torch.tensor(intrinsics['fx']).item()
    fy = torch.tensor(intrinsics['fy']).item()
    cx = torch.tensor(intrinsics['cx']).item()
    cy = torch.tensor(intrinsics['cy']).item()
    
    # Create grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # Convert pixel coordinates to camera coordinates
    x = (u - cx) * depth_map.cpu().numpy() / fx
    y = (v - cy) * depth_map.cpu().numpy() / fy
    z = depth_map.cpu().numpy()
    
    # Reshape to (B*S, H*W)
    x = np.reshape(x, (-1))
    y = np.reshape(y, (-1))
    z = np.reshape(z, (-1))
    
    # Stack into point cloud
    pointcloud = np.stack((x, y, z), axis=-1)
    pointcloud = pointcloud[pointcloud[:,2] > 0]
    return pointcloud


def depth_map_to_pointcloud_tensor(depth_map, mask, intrinsics):
    # Get dimensions
    H, W = depth_map.shape

    if mask is not None:
        depth_map[mask == 0] = -1


    # Create grid of pixel coordinates
    u, v = torch.meshgrid(torch.arange(W,device=depth_map.device), torch.arange(H,device=depth_map.device),indexing='xy')
    # Convert pixel coordinates to camera coordinates
    x = (u - intrinsics[0,2]) * depth_map / intrinsics[0,0]
    y = (v - intrinsics[1,2]) * depth_map / intrinsics[1,1]
    z = depth_map

    # Reshape to (B*S, H*W)
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)

    # Stack into point cloud
    pointcloud = torch.stack((x, y, z), dim=-1)
    pointcloud = pointcloud[z > 0]
    return pointcloud

def image_coords_to_camera_space(depth_map, coords_2d, intrinsics):
    
    # Unpack intrinsic matrix
    fx = intrinsics['fx'].item()
    fy = intrinsics['fy'].item()
    cx = intrinsics['cx'].item()
    cy = intrinsics['cy'].item()
    
    #print(coords_2d.shape, coords_2d, np.max(coords_2d[:,0]), np.max(coords_2d[:,1]))
    depths = depth_map[coords_2d[:,0], coords_2d[:,1]]
    x = (coords_2d[:,1] - cx) * depths / fx
    y = (coords_2d[:,0] - cy) * depths / fy
    z = depths
    
    pointcloud = np.stack((x, y, z), axis=-1)
    return pointcloud

def save_pointcloud(pointcloud, filename):
    """
    Save a point cloud to a text file.

    Args:
        pointcloud (numpy array): Point cloud array of shape (N, 3).
        filename (str): Name of the file to save.
    """
    if torch.is_tensor(pointcloud):
        pointcloud = pointcloud.cpu().numpy()
    if pointcloud.shape[1] == 3:
        with open(filename, 'w') as f:
            for point in pointcloud:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
    if pointcloud.shape[1] == 6:
        with open(filename, 'w') as f:
            for point in pointcloud:
                f.write(f"{point[0]} {point[1]} {point[2]} {point[3]} {point[4]} {point[5]}\n")
                
def read_pointcloud(filename):
    f = open(filename)
    lines = f.readlines()
    pointcloud = []
    for line in lines:
        x, y, z = float(line.split(' ')[0]), float(line.split(' ')[1]), float(line.split(' ')[2])
        pointcloud.append([x, y, z])
    pointcloud = np.array(pointcloud)
    f.close()
    return pointcloud
            
def transform_pointcloud(pointcloud, transformation_matrix):
    
    # Append a column of ones to make homogeneous coordinates
    homogeneous_points = np.hstack((pointcloud, np.ones((pointcloud.shape[0], 1))))
    
    # Perform transformation
    transformed_points = np.dot(transformation_matrix, homogeneous_points.T).T
    
    # Divide by the last coordinate (homogeneous division)
    transformed_points = transformed_points[:, :3] / transformed_points[:, 3][:, np.newaxis]
    
    return transformed_points

def transform_batch_pointcloud_torch(pointcloud, transformation_matrix):
    
    N, D = pointcloud.shape
    assert(N == transformation_matrix.shape[0])
    # Append a column of ones to make homogeneous coordinates
    ones = torch.ones((pointcloud.shape[0], 1), device = pointcloud.device)
    homogeneous_points = torch.concat([pointcloud, ones],axis=1).unsqueeze(2) # N, 4, 1
    
    # Perform transformation
    transformed_points = torch.bmm(transformation_matrix, homogeneous_points).squeeze() # N, 4
    
    # Divide by the last coordinate (homogeneous division)
    transformed_points = transformed_points[:, :3] / transformed_points[:, 3:4]
    
    return transformed_points

def transform_pointcloud_torch(pointcloud, transformation_matrix):
    N, D = pointcloud.shape
    N_trans, D1, D2 = transformation_matrix.shape
    assert(D1==4)
    assert(D2==4)
    #pointcloud = pointcloud.unsqueeze(0).repeat(N_trans,1,1)
    
    # Append a column of ones to make homogeneous coordinates
    ones = torch.ones((pointcloud.shape[0], 1), device = pointcloud.device)
    homogeneous_points = torch.concat([pointcloud, ones],axis=1).permute(1,0).float() # 4, N
    homogeneous_points = homogeneous_points.unsqueeze(0).repeat(N_trans, 1, 1) # N_trans, 4, N
    transformed_points = torch.bmm(transformation_matrix, homogeneous_points).permute(0,2,1) # N_trans, N, 4
    transformed_points = transformed_points[:,:,:3] / transformed_points[:,:,3:4]
    return transformed_points
    
def project_points(cam_space_coords, intrinsics):
    
    # Unpack intrinsic matrix
    fx = intrinsics['fx'].item()
    fy = intrinsics['fy'].item()
    cx = intrinsics['cx'].item()
    cy = intrinsics['cy'].item()
    
    K = np.zeros((3,4))
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx
    K[1,2] = cy
    K[2,2] = 1
    
    B, N, D = cam_space_coords.shape
    cam_space_coords = cam_space_coords.reshape(B*N, D).cpu().numpy() # N, 3
    ones = np.ones((cam_space_coords.shape[0], 1))
    homo_coords = np.concatenate([cam_space_coords, ones], axis=1) # N, 4
    img_coords = np.dot(K, homo_coords.T).T # N, 3
    img_coords = img_coords[:,:2] / img_coords[:,2:3]
    img_coords = img_coords.reshape(B, N, 2)
    return img_coords


def project_points_float(cam_space_coords, intrinsics):
    # Unpack intrinsic matrix
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    K = np.zeros((3, 4))
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1

    N, D = cam_space_coords.shape
    ones = np.ones((cam_space_coords.shape[0], 1))
    homo_coords = np.concatenate([cam_space_coords, ones], axis=1)  # N, 4
    img_coords = np.dot(K, homo_coords.T).T  # N, 3
    img_coords = img_coords[:, :2] / img_coords[:, 2:3]
    return img_coords

def get_2dbboxes(masks):
    
    if len(masks.shape) == 4:
        B, S, H, W =  masks.shape
        # Initialize tensor to store bounding boxes
        bboxes = torch.zeros((B, S, 4))
        
        # Iterate over batch dimension
        for b in range(B):
            # Iterate over sequence dimension
            for s in range(S):
                # Find coordinates of non-zero elements in the mask
                non_zero_coords = torch.nonzero(masks[b, s].float())
                #print(non_zero_coords.shape)
                #print(non_zero_coords)
                
                # Extract bounding box coordinates
                ymin = non_zero_coords[:, 0].min()
                ymax = non_zero_coords[:, 0].max()
                xmin = non_zero_coords[:, 1].min()
                xmax = non_zero_coords[:, 1].max()
                
                # Store bounding box coordinates
                bboxes[b, s] = torch.tensor([ymin, xmin, ymax, xmax])
    elif len(masks.shape) == 3:     # make bbox square
        B, H, W = masks.shape
        # Initialize tensor to store bounding boxes
        bboxes = torch.zeros((B, 4))
        # Iterate over batch dimension
        for b in range(B):
            # Find coordinates of non-zero elements in the mask
            non_zero_coords = torch.nonzero(masks[b].float())
            #print(non_zero_coords.shape)
            #print(non_zero_coords)
            
            # Extract bounding box coordinates
            ymin = non_zero_coords[:, 0].min()
            ymax = non_zero_coords[:, 0].max() + 1
            xmin = non_zero_coords[:, 1].min()
            xmax = non_zero_coords[:, 1].max() + 1

            w = xmax-xmin
            h = ymax-ymin
            if h > w:
                dif = h-w
                xmin = xmin-dif//2
                xmax = xmax+dif//2
                if xmin < 0:
                    xmax = xmax - xmin
                    xmin = 0
                if xmax > W:
                    xmin = xmin-xmax+W
                    xmax = W
            elif w>h :
                dif = w-h
                ymin = ymin-dif/2
                ymax = ymax+dif/2
                if ymin < 0:
                    ymax = ymax - ymin
                    ymin = 0
                if ymax > H:
                    ymin = ymin-ymax+H
                    ymax = H
            ymin = max(ymin,0)
            xmin = max(xmin,0)
            ymax = min(ymax, H)
            xmax = min(xmax, W)
            # Store bounding box coordinates
            bboxes[b] = torch.tensor([ymin, xmin, ymax, xmax])
            bboxes = torch.clamp(bboxes,0)
    
    return bboxes.int().numpy()

def create_3dmeshgrid(x, y, z, device):
    x_ = torch.arange(x, device = device)
    y_ = torch.arange(y, device = device)
    z_ = torch.arange(z, device = device)
    mesh_x, mesh_y, mesh_z = torch.meshgrid(x_, y_, z_, indexing="ij")
    meshgrid = torch.stack([mesh_x, mesh_y, mesh_z], axis = 3)
    return meshgrid

def calc_masked_batch_var(cosine_sims, ref_masks):
    # cosine sims shape N_pts, N_ref, feat_H, feat_W
    N_pts, N_ref, feat_H, feat_W = cosine_sims.shape
    cosine_sims = cosine_sims.reshape(N_pts, N_ref, feat_H*feat_W)
    ref_masks = ref_masks.reshape(N_ref, feat_H*feat_W).unsqueeze(0)
    #print((ref_masks>0).device, cosine_sims.device)
    #print(torch.nan.device)
    cosine_sims = torch.where(ref_masks>0, cosine_sims, torch.nan)
    mean = torch.nanmean(cosine_sims, dim = 2).unsqueeze(2)
    var = torch.sqrt(
        torch.nanmean(
            torch.pow( torch.abs(cosine_sims - mean), 2)
        ))
    
    return var

def calc_coords_3d_var(sims_and_coords, threshold):
    # sims and coords N_2d_pts, N_3d_pts, 8
    '''
    mask = (sims_and_coords[:,:,0:1] > threshold).repeat(1, 1, 3)
    #print(torch.sum(mask, dim=1)[:10])
    target_coords = masked_tensor(sims_and_coords[:,:,5:], mask)
    #print(target_coords.shape)
    #print(torch.sum(target_coords, dim = 1)[:10])
    vars = torch.var(target_coords, dim = 1)
    #print(vars.shape)
    #print(vars[:10])
    vars = torch.sum(vars, dim = 1)
    print(vars.shape)
    return vars
    '''
    
    mask = sims_and_coords[:,:,0:1] > threshold
    target_coords = torch.where(mask, sims_and_coords, torch.nan)
    mean = torch.nanmean(target_coords, dim = 1).unsqueeze(1)
    vars = torch.nanmean(
        torch.pow(target_coords - mean, 2), axis = 1
    )
    vars = torch.nansum(vars, axis = 1)
    
    return vars


if __name__ == '__main__':
    points = np.random.random((8,3))
    result = farthest_point_sampling(points,4)