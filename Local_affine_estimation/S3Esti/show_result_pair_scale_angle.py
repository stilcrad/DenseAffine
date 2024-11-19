import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt
from HPatches_homo_dataset import HPatchesHomoDataset
from torch.utils.data import DataLoader


def draw_center_line(name, image_list, H_pair, centers, write_root, draw_mark):
    color_list = [(255, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

    scale_max = 5
    scale_min = 1 / scale_max
    line_color=(0, 255, 255)
    if centers is None:
        point_radius = 8
        line_thick = 3
        line_len = 100
    else:
        point_radius = 1
        line_thick = 1
        line_len = 5

    max_len = line_len * scale_max
    min_len = line_len * scale_min

    write_dir = os.path.join(write_root, name)
    if draw_mark:
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
    rela_angle_dict = {}
    rela_scale_dict = {}
    each_num = len(image_list)
    draw_margin = 10
    for id_1 in range(each_num):
        image1 = image_list[id_1].copy()
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        row, col = image1.shape[:2]
        if centers is None:
            centers = np.array([[col / 2, row / 2]])

        center_num = centers.shape[0]
        for id_2 in range(id_1 + 1, each_num):
            H_now = H_pair[(id_1, id_2)]
            image2 = image_list[id_2].copy()
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

            rela_scale_dict[(id_1, id_2)] = np.zeros((center_num,))
            rela_angle_dict[(id_1, id_2)] = np.zeros((center_num,))
            angle_rela_each = np.zeros((center_num, 2))
            scale_rela_each = np.zeros((center_num, 2))
            for c_id in range(center_num):
                center_r = centers[c_id, 1]
                center_c = centers[c_id, 0]

                point_center = np.array([[center_c, center_r, 1],
                                         [center_c, center_r - line_len, 1],
                                         [center_c, center_r + line_len, 1],
                                         [center_c - line_len, center_r, 1],
                                         [center_c + line_len, center_r, 1]])
                len1 = np.sqrt(np.sum(np.square(point_center[(1, 3), :2] -
                                                point_center[(2, 4), :2]), axis=1))
                angle1 = np.arctan2(point_center[(2, 4), 1] - point_center[(1, 3), 1],
                                    point_center[(2, 4), 0] - point_center[(1, 3), 0]) * 180 / math.pi
                angle1[angle1 < 0] = angle1[angle1 < 0] + 360
                if draw_mark:
                    point_center_int = np.round(point_center).astype('int')
                    p_base = (point_center_int[0, 0], point_center_int[0, 1])
                    for p_id in range(1, point_center_int.shape[0]):
                        p2 = (point_center_int[p_id, 0], point_center_int[p_id, 1])
                        image1 = cv2.line(image1, p_base, p2, color=line_color, thickness=line_thick)
                        image1 = cv2.circle(image1, p2, radius=point_radius,
                                            color=color_list[p_id - 1], thickness=-1)

                warped_center = np.dot(point_center, np.transpose(H_now))
                warped_center = warped_center[:, :2] / warped_center[:, 2:]

                end_point = warped_center[1:]
                diff = end_point - warped_center[0:1]
                if np.max(diff[0] * diff[1]) > 0.1 or np.max(diff[2] * diff[3]) > 0.1:
                    c_wrap = warped_center[0, 0]
                    r_wrap = warped_center[0, 1]
                    warped_center = np.array([[c_wrap, r_wrap, 1],
                                              [c_wrap, r_wrap - line_len, 1],
                                              [c_wrap, r_wrap + line_len, 1],
                                              [c_wrap - line_len, r_wrap, 1],
                                              [c_wrap + line_len, r_wrap, 1]])
                    end_point = warped_center[1:]
                    diff = end_point - warped_center[0:1]

                each_len = np.sqrt(np.sum(np.square(diff), axis=1))
                large_pos = (each_len > max_len)
                small_pos = (each_len < min_len)
                if np.sum(large_pos) > 0:
                    end_point[large_pos] = ((max_len / each_len[large_pos][:, np.newaxis]) *
                                            diff[large_pos, :] + warped_center[0:1])
                if np.sum(small_pos) > 0:
                    end_point[small_pos] = ((min_len / each_len[small_pos][:, np.newaxis]) *
                                            diff[small_pos] + warped_center[0:1])
                warped_center[1:] = end_point

                len2 = np.sqrt(np.sum(np.square(warped_center[(1, 3), :2] -
                                                warped_center[(2, 4), :2]), axis=1))
                angle2 = np.arctan2(warped_center[(2, 4), 1] - warped_center[(1, 3), 1],
                                    warped_center[(2, 4), 0] - warped_center[(1, 3), 0]) * 180 / math.pi
                angle2[angle2 < 0] = angle2[angle2 < 0] + 360
                scale_rela_each_now = len2 / len1
                scale_rela = np.sqrt(np.prod(scale_rela_each_now))

                angle_rela_prob = np.stack((angle2 - angle1, angle2 + 360 - angle1,
                                            angle2 - 360 - angle1), axis=1)
                angle_rela_pos = np.argmin(np.abs(angle_rela_prob), axis=1)
                angle_rela_each_now = angle_rela_prob[np.arange(2), angle_rela_pos]
                angle_rela_each_now[angle_rela_each_now < -math.pi] += (math.pi * 2)
                angle_rela_each_now[angle_rela_each_now > math.pi] -= (math.pi * 2)
                angle_prob_num = 2
                angle_mean_prob = np.zeros((angle_prob_num,))
                angle_std_prob = np.zeros((angle_prob_num,))
                angle_mean_prob[0] = np.mean(angle_rela_each_now)
                angle_std_prob[0] = np.std(angle_rela_each_now)
                angle_rela_each_local = (angle_rela_each_now + 360) % 360
                angle_std_prob[1] = np.std(angle_rela_each_local)
                angle_rela_local = np.mean(angle_rela_each_local)
                if angle_rela_local > 180:
                    angle_rela_local = angle_rela_local - 360
                angle_mean_prob[1] = angle_rela_local
                angle_rela_pos = np.argmin(angle_std_prob)
                angle_rela = angle_mean_prob[angle_rela_pos]

                rela_scale_dict[(id_1, id_2)][c_id] = scale_rela
                rela_angle_dict[(id_1, id_2)][c_id] = angle_rela
                scale_rela_each[c_id] = scale_rela_each_now
                angle_rela_each[c_id] = angle_rela_each_now
                if draw_mark:
                    row2, col2 = image2.shape[0], image2.shape[1]
                    warped_center_draw = np.round(warped_center).astype('int')
                    warped_center_draw[:, 0] = np.minimum(np.maximum(0, warped_center_draw[:, 0]), col2 - 1)
                    warped_center_draw[:, 1] = np.minimum(np.maximum(0, warped_center_draw[:, 1]), row2 - 1)
                    p_base = (warped_center_draw[0, 0], warped_center_draw[0, 1])
                    for p_id in range(1, warped_center_draw.shape[0]):
                        p2 = (warped_center_draw[p_id, 0], warped_center_draw[p_id, 1])
                        image2 = cv2.line(image2, p_base, p2, color=line_color, thickness=line_thick)
                        image2 = cv2.circle(image2, p2, radius=point_radius,
                                            color=color_list[p_id - 1], thickness=-1)
            if draw_mark:
                row1 = image1.shape[0]
                row2 = image2.shape[0]
                if row1 < row2:
                    white_image = image1[0].copy()
                    white_image[:] = 255
                    white_image = np.tile(white_image, (row2 - row1, 1, 1))
                    image1 = np.concatenate((image1, white_image), axis=0)
                elif row1 > row2:
                    white_image = image2[0].copy()
                    white_image[:] = 255
                    white_image = np.tile(white_image, (row1 - row2, 1, 1))
                    image2 = np.concatenate((image2, white_image), axis=0)
                white_image = image1[:, :draw_margin, :].copy()
                white_image[:] = 255
                image_write = np.concatenate((image1, white_image, image2), axis=1)
                col_half = round(image_write.shape[1] / 2)
                id_show_xy = (col_half - 300, 50)
                scale_rela_first = rela_scale_dict[(id_1, id_2)][0]
                angle_rela_first = rela_angle_dict[(id_1, id_2)][0]
                show_txt = ('a:%.1f,%.1f, %.1f, s:%.2f,%.2f, %.2f' % (
                    angle_rela_each[0, 0], angle_rela_each[0, 1], angle_rela_first,
                    scale_rela_each[0, 0], scale_rela_each[0, 1], scale_rela_first))
                cv2.putText(image_write, show_txt, id_show_xy,
                            cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)

                write_name = os.path.join(write_dir, '%d_%d.jpg' % (id_1, id_2))
                cv2.imwrite(write_name, image_write)

            temp = 1

    return rela_scale_dict, rela_angle_dict


def load_pair_result(pair_result_path, key_name):
    with open(pair_result_path, 'r') as f:
        lines = f.readlines()
    lines = [line.split(',') for line in lines]
    key_pos = lines[0].index(key_name)
    result_dict = {}
    for k in range(len(lines)):
        line_now = lines[k]
        result_dict[line_now[0]] = float(line_now[key_pos + 3])

    return result_dict

def get_dataset_pair_s_o(dataset_root):
    pair_show_root = 'debug_dir'
    image_ext = '.ppm'
    image_row, image_col = 480, 640
    data_HPatches = HPatchesHomoDataset(dataset_root, image_ext, image_row, image_col)
    draw_mark = False
    rela_scale_dict = {}
    rela_angle_dict = {}
    for idx in range(len(data_HPatches)):
        name = data_HPatches.name_list[idx]
        H_pair = data_HPatches.H_pair_dict[name]
        image_list = data_HPatches[idx]
        each_num = len(image_list)
        rela_scale_now, rela_angle_now = draw_center_line(
            name, image_list, H_pair, None, pair_show_root, draw_mark)
        for key_now in rela_scale_now.keys():
            key_str = '%s-%d_%d' % (name, key_now[0], key_now[1])
            assert rela_scale_now[key_now].shape[0] == 1
            assert rela_angle_now[key_now].shape[0] == 1
            rela_scale_dict[key_str] = rela_scale_now[key_now][0]
            rela_angle_dict[key_str] = rela_angle_now[key_now][0]
    pair_num = len(rela_scale_dict)
    scale_vec = np.zeros((pair_num,))
    angle_vec = np.zeros((pair_num,))
    pair_key = list(rela_scale_dict.keys())
    for idx, key_now in enumerate(pair_key):
        scale_vec[idx] = rela_scale_dict[key_now]
        angle_vec[idx] = rela_angle_dict[key_now]

    return scale_vec, angle_vec, pair_key
