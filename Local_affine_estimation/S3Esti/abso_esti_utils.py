import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
from abso_esti_model.abso_esti_image_set import get_affine_patch
from scipy.ndimage.filters import maximum_filter


def get_wrap_patches(image_tensor, model_scale, model_angle, point_vec,
                     patch_size, esti_scale_ratio_list, scale_angle_result_list=None,
                     scale_adapt_ratio=None):
    esti_patch_size = 32

    centers = point_vec[:, :2].astype('float32')
    patch_num_now = centers.shape[0]
    scale_init = np.ones((patch_num_now,))
    angle_init = np.zeros((patch_num_now,))

    patch = get_affine_patch(
        image_tensor, centers, esti_patch_size,
        scale_init, angle_init, esti_scale_ratio_list)
    with torch.no_grad():
        scale_resp = model_scale(patch)
        angle_resp = model_angle(patch)

    scale_ind_pred = torch.argmax(scale_resp, dim=1)
    angle_ind_pred = torch.argmax(angle_resp, dim=1)
    scale = model_scale.map_id_to_scale(scale_ind_pred).cpu().numpy()
    angle = model_angle.map_id_to_angle(angle_ind_pred).cpu().numpy()
    s_to_base = 1 / (scale + 1e-16)
    a_to_base = -angle

    base_s = 1
    if scale_adapt_ratio is not None:
        base_s = scale_adapt_ratio
    s_to_base_crop = s_to_base / base_s
    patch_new = get_affine_patch(image_tensor,
                                 centers, patch_size, s_to_base_crop, a_to_base, [1])
    patch_new = torch.clamp((patch_new + 1) * 127, min=0, max=255)
    if scale_angle_result_list is not None:
        scale_angle_result_list.append(s_to_base)
        scale_angle_result_list.append(a_to_base)
    return patch_new


def cross_main_pair(scale_mat, angle_mat):
    point_num, scale_num = scale_mat.shape
    assert angle_mat.shape[0] == point_num
    angle_num = angle_mat.shape[1]
    new_col = scale_num + angle_num - 1
    scale_pair_mat = np.zeros((point_num, new_col), dtype=scale_mat.dtype)
    angle_pair_mat = np.zeros((point_num, new_col), dtype=scale_mat.dtype)
    scale_pair_mat[:, :angle_num - 1] = np.tile(scale_mat[:, 0:1], (1, angle_num - 1))
    scale_pair_mat[:, angle_num - 1:] = scale_mat
    angle_pair_mat[:, :angle_num] = angle_mat
    if scale_num-1>0:
        angle_pair_mat[:, angle_num:] = np.tile(angle_mat[:, 0:1], (1, scale_num - 1))
    return scale_pair_mat, angle_pair_mat


def get_wrap_patches_multi(image_tensor, model_scale, model_angle, point_vec,
                           patch_size, esti_scale_ratio_list, ignore_str=None,
                           scale_angle_result_list=None,
                           point_id_list=None, scale_adapt_ratio=None,
                           scale_num=1,angle_num=1):
    esti_patch_size = 32
    point_num = point_vec.shape[0]
    device = image_tensor.device

    centers = point_vec[:, :2].astype('float32')
    patch_num_now = centers.shape[0]
    scale_init = np.ones((patch_num_now,))
    angle_init = np.zeros((patch_num_now,))

    patch = get_affine_patch(
        image_tensor, centers, esti_patch_size,
        scale_init, angle_init, esti_scale_ratio_list)
    with torch.no_grad():
        scale_resp = model_scale(patch)
        angle_resp = model_angle(patch)
        scale_score = F.softmax(scale_resp, dim=1).cpu().numpy()
        angle_score = F.softmax(angle_resp, dim=1).cpu().numpy()
    scale_max_len = 30
    angle_max_len = 45
    scale_win = np.ones((1, scale_max_len), dtype='float32')
    angle_win = np.ones((1, angle_max_len), dtype='float32')
    scale_max = maximum_filter(scale_score, footprint=scale_win)
    angle_max = maximum_filter(angle_score, footprint=angle_win)
    scale_local_max = scale_score * (scale_score == scale_max).astype('float32')
    angle_local_max = angle_score * (angle_score == angle_max).astype('float32')
    each_max_num = 3
    if scale_num is None:
        scale_num=each_max_num
    if angle_num is None:
        angle_num=each_max_num
    scale_sort = np.sort(scale_local_max, axis=1)[:, ::-1]
    angle_sort = np.sort(angle_local_max, axis=1)[:, ::-1]
    scale_sort_ind = np.argsort(scale_local_max, axis=1)[:, ::-1]
    angle_sort_ind = np.argsort(angle_local_max, axis=1)[:, ::-1]
    scale_pair_ind, angle_pair_ind = cross_main_pair(
        scale_sort_ind[:, :scale_num], angle_sort_ind[:, :angle_num])
    scale_pair_score, angle_pair_score = cross_main_pair(
        scale_sort[:, :scale_num], angle_sort[:, :angle_num])
    point_id = np.tile(np.arange(point_num)[:, np.newaxis], (1, scale_num + angle_num - 1))
    min_thre = 0.001
    remain_pos = ((scale_pair_score > min_thre) & (angle_pair_score > min_thre))
    scale_ind_pred = scale_pair_ind[remain_pos]
    angle_ind_pred = angle_pair_ind[remain_pos]
    point_id_pred = point_id[remain_pos]

    scale_ind_pred = torch.from_numpy(scale_ind_pred).to(device)
    angle_ind_pred = torch.from_numpy(angle_ind_pred).to(device)
    scale = model_scale.map_id_to_scale(scale_ind_pred).cpu().numpy()
    angle = model_angle.map_id_to_angle(angle_ind_pred).cpu().numpy()
    s_to_base = 1 / (scale + 1e-16)
    a_to_base = -angle

    if ignore_str == 'scale':
        s_to_base[:] = 1
    if ignore_str == 'angle':
        a_to_base[:] = 0

    base_s = 1
    if scale_adapt_ratio is not None:
        base_s = scale_adapt_ratio
    s_to_base_crop = s_to_base / base_s
    centers_new = centers[point_id_pred]
    patch_new = get_affine_patch(image_tensor,
                                 centers_new, patch_size, s_to_base_crop, a_to_base, [1])
    patch_new = torch.clamp((patch_new + 1) * 127, min=0, max=255)
    point_vec_new = point_vec[point_id_pred]

    if scale_angle_result_list is not None:
        scale_angle_result_list.append(s_to_base)
        scale_angle_result_list.append(a_to_base)
    if point_id_list is not None:
        point_id_list.append(point_id_pred)
    return patch_new, point_vec_new


def get_wrap_patches_no_alg(image_tensor, model_scale, model_angle, point_vec,
                            patch_size, esti_scale_ratio_list):
    esti_patch_size = 32
    point_num = point_vec.shape[0]
    device = image_tensor.device

    centers = point_vec[:, :2].astype('float32')
    patch_num_now = centers.shape[0]

    scale_ind_base = np.array([[1, 0.5, 2]])
    scale = np.tile(scale_ind_base, (patch_num_now, 1))
    angle_ind_base = np.array([[math.pi, 0, 0]])
    angle = np.tile(angle_ind_base, (patch_num_now, 1))
    point_id_pred = np.tile(np.arange(patch_num_now)[:, np.newaxis], (1, 3))
    scale = scale.reshape(-1)
    angle = angle.reshape(-1)
    point_id_pred = point_id_pred.reshape(-1)

    s_to_base = 1 / (scale + 1e-16)
    a_to_base = -angle

    base_s = 1
    s_to_base_crop = s_to_base / base_s
    centers_new = centers[point_id_pred]
    patch_new = get_affine_patch(image_tensor,
                                 centers_new, patch_size, s_to_base_crop, a_to_base, [1])
    patch_new = torch.clamp((patch_new + 1) * 127, min=0, max=255)
    point_vec_new = point_vec[point_id_pred]
    return patch_new, point_vec_new
