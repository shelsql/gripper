# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import cv2
import numpy as np
import torch
from pyrr import Quaternion
import copy

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def rtvec_to_matrix(rvec, tvec):
	"""
	Convert rotation vector and translation vector to 4x4 matrix
	"""
	rvec = np.asarray(rvec)
	tvec = np.asarray(tvec)

	T = np.eye(4)
	R, jac = cv2.Rodrigues(rvec)
	T[:3, :3] = R
	T[:3, 3] = tvec.squeeze() # this is the fix
	return T

def convert_rvec_to_quaternion(rvec):
    """Convert rvec (which is log quaternion) to quaternion"""
    theta = np.sqrt(
        rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]
    )  # in radians
    raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

    # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
    quaternion = Quaternion.from_axis_rotation(raxis, theta)
    quaternion.normalize()
    return quaternion


def hnormalized(vector):
    hnormalized_vector = (vector / vector[-1])[:-1]
    return hnormalized_vector


def point_projection_from_3d(camera_K, points):
    projections = []
    for p in points:
        p_unflattened = np.matmul(camera_K, p)
        projection = hnormalized(p_unflattened)
        projections.append(projection)
    projections = np.array(projections)
    return projections


def solve_pnp(
    canonical_points,
    projections,
    camera_K,
    method=cv2.SOLVEPNP_EPNP,
    refinement=True,
    dist_coeffs=np.array([]),
):

    n_canonial_points = len(canonical_points)
    n_projections = len(projections)
    assert (
        n_canonial_points == n_projections
    ), "Expected canonical_points and projections to have the same length, but they are length {} and {}.".format(
        n_canonial_points, n_projections
    )

    # Process points to remove any NaNs
    canonical_points_proc = []
    projections_proc = []
    for canon_pt, proj in zip(canonical_points, projections):

        if (
            canon_pt is None
            or len(canon_pt) == 0
            or canon_pt[0] is None
            or canon_pt[1] is None
            or proj is None
            or len(proj) == 0
            or proj[0] is None
            or proj[1] is None
        ):
            continue

        canonical_points_proc.append(canon_pt)
        projections_proc.append(proj)

    # Return if no valid points
    if len(canonical_points_proc) == 0:
        return False, None, None

    canonical_points_proc = np.array(canonical_points_proc)
    projections_proc = np.array(projections_proc)

    # Use cv2's PNP solver
    try:
        pnp_retval, rvec, tvec = cv2.solvePnP(
            canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
            projections_proc.reshape(projections_proc.shape[0], 1, 2),
            camera_K,
            dist_coeffs,
            flags=method,
        )

        if refinement:
            pnp_retval, rvec, tvec = cv2.solvePnP(
                canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
                projections_proc.reshape(projections_proc.shape[0], 1, 2),
                camera_K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=True,
                rvec=rvec,
                tvec=tvec,
            )
        translation = tvec[:, 0]
        quaternion = (convert_rvec_to_quaternion(rvec[:, 0]).xyzw)
        # x,y,z,w = quaternion
        # quaternion = np.array([w,x,y,z])

    except:
        pnp_retval = False
        translation = None
        quaternion = None

    return pnp_retval, translation, quaternion


def solve_pnp_ransac(
    canonical_points,
    projections,
    camera_K,
    method=cv2.SOLVEPNP_EPNP,
    inlier_thresh_px=5.0,  # this is the threshold for each point to be considered an inlier
    dist_coeffs=np.array([]),
):

    n_canonial_points = canonical_points.shape[0]
    n_projections = projections.shape[0]
    assert (
        n_canonial_points == n_projections
    ), "Expected canonical_points and projections to have the same length, but they are length {} and {}.".format(
        n_canonial_points, n_projections
    )

    # Process points to remove any NaNs
    canonical_points_proc = []
    projections_proc = []
    #for canon_pt, proj in zip(canonical_points, projections):
    for i in range(n_canonial_points):
        canon_pt = canonical_points[i]
        proj = projections[i]
        if (
            canon_pt is None
            or len(canon_pt) == 0
            or canon_pt[0] is None
            or canon_pt[1] is None
            or proj is None
            or len(proj) == 0
            or proj[0] is None
            or proj[1] is None
        ):
            continue

        canonical_points_proc.append(canon_pt)
        projections_proc.append(proj)

    # Return if no valid points
    if len(canonical_points_proc) == 0:
        return False, None, None, None

    canonical_points_proc = np.array(canonical_points_proc)
    projections_proc = np.array(projections_proc)

    # Use cv2's PNP solver
    try:
        pnp_retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
            projections_proc.reshape(projections_proc.shape[0], 1, 2),
            camera_K,
            dist_coeffs,
            reprojectionError=inlier_thresh_px,
            flags=method,
        )
        #print("cv2 solve results:")
        #print("pnp_retval", pnp_retval)
        #print("rvec", rvec)
        #print("tvec", tvec)
        print("inliers", inliers.shape)

        translation = tvec[:, 0]
        quaternion = (convert_rvec_to_quaternion(rvec[:, 0]).xyzw)
        
        rt_matrix = rtvec_to_matrix(rvec, tvec)
        # x,y,z,w = quaternion
        # quaternion = np.array([w,x,y,z])

    except:
        pnp_retval = False
        translation = None
        quaternion = None
        rt_matrix = None
        inliers = None

    return pnp_retval, translation, rt_matrix, inliers


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def add_from_pose(translation, quaternion, keypoint_positions_wrt_cam_gt, camera_K):
    #print(translation,quaternion)
    transform = np.eye(4)
    transform[:3, :3] = quaternion_to_matrix(quaternion).numpy()
    transform[:3, -1] = np.array(translation).squeeze()
    kp_pos_gt_homog = np.hstack(
        (
            keypoint_positions_wrt_cam_gt,
            np.ones((keypoint_positions_wrt_cam_gt.shape[0], 1)),
        )
    )
    kp_pos_aligned = np.transpose(np.matmul(transform, np.transpose(kp_pos_gt_homog)))[
        :, :3
    ]
    #print(kp_pos_aligned,keypoint_positions_wrt_cam_gt)
    # The below lines were useful when debugging pnp ransac, so left here for now
    # projs = point_projection_from_3d(camera_K, kp_pos_aligned)
    # temp = np.linalg.norm(kp_projs_est_pnp - projs, axis=1) # all of these should be below the inlier threshold above!
    kp_3d_errors = kp_pos_aligned - keypoint_positions_wrt_cam_gt
    kp_3d_l2_errors = np.linalg.norm(kp_3d_errors, axis=1)
    add = np.mean(kp_3d_l2_errors)
    return add


def nocs2pose(camera_K, nocs_image, mask_image, scale):
    # nocs_image = cv2.resize(nocs_image, (126, 224), interpolation=cv2.INTER_NEAREST)
    # mask_image = cv2.resize(mask_image, (126, 224), interpolation=cv2.INTER_NEAREST)
    h, w, c = nocs_image.shape
    camera_scale_x = w / camera_K['xres']
    camera_scale_y = h / camera_K['yres']
    camera_K = np.array([[camera_K['fx'] * camera_scale_x, 0, camera_K['cx'] * camera_scale_x],
                        [0, camera_K['fy'] * camera_scale_y , camera_K['cy'] * camera_scale_y],
                        [0, 0, 1]])
    h, w, c = nocs_image.shape
    pc_gripper = np.zeros([h, w, 5])
    pc_gripper[:, :, :3] = copy.deepcopy(nocs_image)
    pc_gripper[:, :, 0] = (pc_gripper[:, :, 0] - 0.5) * scale[0]
    pc_gripper[:, :, 1] = (pc_gripper[:, :, 1] - 0.5) * scale[1]
    pc_gripper[:, :, 2] = pc_gripper[:, :, 2] * scale[2]
    np.savetxt('test.txt', pc_gripper[:, :, :3].reshape(-1, 3))
    pc_gripper[:, :, 3:] = np.indices((h, w), dtype=np.int32).transpose(1,2,0)
    pc_gripper = pc_gripper[mask_image == 1]
    x_3d = pc_gripper[:, :3]
    x_2d = pc_gripper[:, [4, 3]]
    dist_coeffs = np.array([0., 0., 0., 0.])

    # pnp_retval, trans, quat, inliers = solve_pnp_ransac(
    #     canonical_points = x_3d,# .reshape(x_3d.shape[0], 1, 3),
    #     projections = x_2d,# .reshape(x_2d.shape[0], 1, 2),
    #     camera_K = camera_K,
    #     dist_coeffs = dist_coeffs,
    #     # reprojectionError=inlier_thresh_px,
    #     # flags=method,
    # )

    pnp_retval, trans, quat = solve_pnp_ransac(
        canonical_points = x_3d,# .reshape(x_3d.shape[0], 1, 3),
        projections = x_2d,# .reshape(x_2d.shape[0], 1, 2),
        camera_K = camera_K,
        method = cv2.SOLVEPNP_EPNP,
        dist_coeffs = dist_coeffs,
        # reprojectionError=inlier_thresh_px,
        # flags=method,
    )
    return pnp_retval, quat, trans
