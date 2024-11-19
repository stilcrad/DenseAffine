import torch
import numpy as np
import torch.nn.functional as F
import math


def get_center_desc(coarse_desc):
    patch_num, desc_len = coarse_desc.shape[:2]
    center_point = torch.zeros(patch_num, 1, 1, 2)
    center_point = center_point.to(coarse_desc.device)
    desc = torch.nn.functional.grid_sample(coarse_desc, center_point)
    desc = desc.squeeze(-1).squeeze(-1)
    desc = desc / (torch.norm(desc, dim=1, keepdim=True) + 1e-12)

    return desc


def superpoint_patch_desc(superpoint_net, patch_gray):
    input_gray = patch_gray / 255
    image_num = input_gray.shape[0]

    patch_num = patch_gray.shape[0]
    each_num = min(patch_num, 100)
    split_num = math.ceil(patch_num / each_num)
    desc_list = []
    for split_id in range(split_num):
        patch_pos = slice(split_id * each_num, min((split_id + 1) * each_num, patch_num), 1)
        input_gray_now = input_gray[patch_pos]
        with torch.no_grad():
            _, coarse_desc = superpoint_net(input_gray_now)
        desc_now = get_center_desc(coarse_desc)
        desc_list.append(desc_now.cpu().numpy())
    desc = np.concatenate(desc_list, axis=0)

    return desc


def POP_patch_desc(POP_net, patch):
    patch_half_size = math.floor(patch.shape[3] / 2)
    patch_num = patch.shape[0]
    # each_num = min(patch_num, 2000)
    each_num = 200
    split_num = math.ceil(patch_num / each_num)
    desc_list = []
    for split_id in range(split_num):
        patch_pos = slice(split_id * each_num, min((split_id + 1) * each_num, patch_num), 1)
        patch_now = patch[patch_pos]
        with torch.no_grad():
            _, desc_map = POP_net(patch_now)
        desc_now = desc_map[:, :, patch_half_size, patch_half_size]
        desc_list.append(desc_now.cpu().numpy())
    desc = np.concatenate(desc_list, axis=0)

    return desc
