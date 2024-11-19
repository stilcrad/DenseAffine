import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import math
from torch.utils.data import Dataset


def get_rand_scale_angle(scale_limit, angle_limit, patch_num):
    rand_scale = (scale_limit[0] + (scale_limit[1] - scale_limit[0]) *
                  np.random.random((patch_num,)))
    rand_angle = (angle_limit[0] + (angle_limit[1] - angle_limit[0]) *
                  np.random.random((patch_num,)))
    return rand_scale, rand_angle


def get_affine_patch(tensor, centers, patch_size, scale_vec, angle_vec,
                     scale_ratio_list):
    torch_version = float('.'.join((torch.__version__).split('.')[:2]))
    torch_version_thre = 1.2

    channel_num, row, col = tensor.shape[1:4]
    patch_num = centers.shape[0]
    patch_size_half = round(patch_size / 2)
    corner = np.array([[-patch_size_half, -patch_size_half],
                       [-patch_size_half, patch_size_half],
                       [patch_size_half, -patch_size_half],
                       [patch_size_half, patch_size_half]]).astype('float32')
    scale_vec = scale_vec.cpu().numpy() if torch.is_tensor(scale_vec) else scale_vec
    angle_vec = angle_vec.cpu().numpy() if torch.is_tensor(angle_vec) else angle_vec
    centers = centers.cpu().numpy() if torch.is_tensor(centers) else centers
    sin_angle = np.sin(angle_vec)
    cos_angle = np.cos(angle_vec)
    patch_list = []
    for k, scale_ratio_now in enumerate(scale_ratio_list):
        mat_list = [get_trans_mat(sin_angle[pos], cos_angle[pos], scale_vec[pos] * scale_ratio_now)
                    for pos in range(patch_num)]
        trans_corner = [cv2.perspectiveTransform(corner[np.newaxis, :], H_mat)
                        for H_mat in mat_list]
        trans_corner = [trans_corner[pos].squeeze(0) + centers[pos:pos + 1] for
                        pos in range(patch_num)]
        corner_norm = corner / patch_size_half
        trans_corner_norm = [get_norm_xy(item, row, col) for item in trans_corner]
        theta = [cv2.getPerspectiveTransform(corner_norm, trans_corner_norm[pos])[np.newaxis, :]
                 for pos in range(patch_num)]
        theta = np.concatenate(theta, axis=0)
        theta = torch.from_numpy(theta[:, :2, :].astype('float32'))
        grid_size = torch.Size((theta.shape[0], channel_num, patch_size, patch_size))
        if torch_version > torch_version_thre:
            grid = F.affine_grid(theta, grid_size, align_corners=True)
        else:
            grid = F.affine_grid(theta, grid_size)
        grid = grid.view(1, grid.shape[0], patch_size * patch_size, 2)
        grid = grid.to(tensor.device)
        if torch_version > torch_version_thre:
            patch_now = F.grid_sample(tensor, grid, padding_mode='zeros', align_corners=True)
        else:
            patch_now = F.grid_sample(tensor, grid, padding_mode='zeros')
        patch_now = patch_now.view(tensor.shape[1], patch_num, patch_size, patch_size)
        patch_now = patch_now.transpose(0, 1)
        patch_list.append(patch_now)

    patch = torch.cat(patch_list, dim=1)
    return patch


def get_norm_xy(xy, row, col):
    xy_new = np.c_[(xy[:, 0] / (col / 2.)) - 1., (xy[:, 1] / (row / 2.)) - 1.]
    return xy_new


def get_trans_mat(sin_v, cos_v, scale):
    mat = np.array([[scale * cos_v, -scale * sin_v, 0],
                    [scale * sin_v, scale * cos_v, 0],
                    [0, 0, 1]], dtype='float32')
    return mat


def get_affine_tensor_batch(scale, angle):
    num = scale.shape[0]
    sin_v = torch.sin(angle)
    cos_v = torch.cos(angle)
    tensor = torch.zeros((num, 2, 3), device=scale.device, dtype=scale.dtype)
    tensor[:, 0, 0] = scale * cos_v
    tensor[:, 0, 1] = -scale * sin_v
    tensor[:, 1, 0] = scale * sin_v
    tensor[:, 1, 1] = scale * cos_v
    return tensor


def remove_out_point(point, xy_range):
    # xy_range:x1,x2,y1,y2
    if point.size < 1:
        return point
    inner_pos = ((point[:, 0] > xy_range[0]) & (point[:, 0] < xy_range[1]) &
                 (point[:, 1] > xy_range[2]) & (point[:, 1] < xy_range[3]))
    point_result = point[inner_pos, :]
    return point_result, inner_pos


def to_tensor(image_np):
    tensor = torch.from_numpy(image_np.transpose((0, 3, 1, 2))).float()
    new_limit = [-1, 1]
    tensor = new_limit[0] + (tensor / 255) * (new_limit[1] - new_limit[0])
    return tensor


def load_file_current_dir(dir_path, ext):
    all_list = os.listdir(dir_path)
    all_list = sorted(all_list)
    all_list = [os.path.join(dir_path, name_now) for name_now in all_list]
    file_list = []
    for name_now in all_list:
        if os.path.isfile(name_now):
            if os.path.splitext(name_now)[-1] == ext:
                file_list.append(name_now)
            else:
                continue
        else:
            sub_file_list = load_file_current_dir(name_now, ext)
            file_list.extend(sub_file_list)

    return file_list
