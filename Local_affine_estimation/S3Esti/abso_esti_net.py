import torch.nn as nn
import torch
from Local_affine_estimation.S3Esti import vgg, knn
import numpy as np
import math


class EstiNet(nn.Module):
    def __init__(self, need_bn, device, out_channels, scale_ratio, in_channels=3,
                 z_channels=256, patch_size=32):
        super(EstiNet, self).__init__()

        self.need_bn = need_bn
        model_type = 'A'
        self.z_channels = z_channels
        self.patch_size = patch_size
        # scale_ratio: the ratios used to generate the multiple-scale input patches
        max_base_ratio = min(scale_ratio)
        self.base_ratio_list = [item / max_base_ratio for item in scale_ratio]
        self.base_ratio_num = len(self.base_ratio_list)
        # L2 loss
        self.MSELoss = torch.nn.MSELoss(reduction='mean')
        self.theta_con_loss_fun = torch.nn.MSELoss(reduction='mean')
        # smooth L1 loss
        self.SmoothL1Loss = torch.nn.SmoothL1Loss(reduction='mean')
        scale_limit = [1 / 9, 9]
        self.scale_limit = scale_limit
        scale_limit = [math.log(item) for item in self.scale_limit]
        self.scale_num = 600
        self.scale_step = (scale_limit[1] - scale_limit[0]) / self.scale_num
        self.scale_vec = (scale_limit[0] + self.scale_step * np.arange(self.scale_num))
        self.scale_vec = torch.from_numpy(self.scale_vec).to(device)
        angle_limit = [-math.pi, math.pi]
        self.angle_limit = angle_limit
        self.angle_num = 720
        self.angle_step = (angle_limit[1] - angle_limit[0]) / self.angle_num
        self.angle_vec = (angle_limit[0] + self.angle_step * np.arange(self.angle_num))
        self.angle_vec = torch.from_numpy(self.angle_vec).to(device)

        # get the relative scales, which are used to compute
        # the accuracy of the relative transformation
        scale_rela_limit = [scale_limit[0] - scale_limit[1],
                            scale_limit[1] - scale_limit[0]]
        self.scale_rela_num = math.ceil((scale_rela_limit[1] - scale_rela_limit[0]) / self.scale_step)
        self.scale_rela_vec = (scale_rela_limit[0] + self.scale_step * np.arange(self.scale_rela_num))
        self.scale_rela_vec = torch.from_numpy(self.scale_rela_vec).to(device)
        # LogSoftmax is used to compute the loss values of different scale and orientations
        self.log_softmax = nn.LogSoftmax(dim=1)

        # encoder
        all_in_channel = self.base_ratio_num * in_channels
        nonlinearity, kwargs = nn.LeakyReLU, {"inplace": True}
        encoder_core = vgg.make_layers(vgg.vgg_cfg[model_type], batch_norm=need_bn,
                                       nonlinearity=nonlinearity, nonlinearity_kwargs=kwargs)
        self.encoder = knn.Unit(all_in_channel, self.z_channels, encoder_core,
                                relu=False, batch_norm=need_bn)
        # obtain the transform parameter
        self.regressor = nn.Sequential(
            nn.Linear(self.z_channels, round(self.z_channels / 2)),
            nn.BatchNorm1d(round(self.z_channels / 2)),
            nn.ReLU(True),
            nn.Linear(round(self.z_channels / 2), out_channels),
        )

    def forward(self, patch):
        batch_size = patch.shape[0]
        z = self.encoder(patch)
        assert z.shape[2] == 1 and z.shape[3] == 1
        z = z.view(batch_size, self.z_channels)
        resp = self.regressor(z)

        return resp

    def get_max_label(self, scale_resp1, angle_resp1, scale_resp2, angle_resp2,
                      scale12, angle12):
        batch_size = scale_resp1.shape[0]
        # get the scores of different scales and orientations
        scale1_ls = self.log_softmax(scale_resp1)
        angle1_ls = self.log_softmax(angle_resp1)
        scale2_ls = self.log_softmax(scale_resp2)
        angle2_ls = self.log_softmax(angle_resp2)
        # get the index differences between two patches
        log_scale12 = torch.log(scale12.cpu())
        scale_diff = torch.round(log_scale12 / self.scale_step).long()
        angle_diff = torch.round(angle12.cpu() / self.angle_step).long()
        # scale_diff = scale_diff.repeat(9, 1)
        # angle_diff = angle_diff.repeat(9, 1)

        # get the corresponding indexes of scale and orientation
        scale1_col = torch.arange(self.scale_num, dtype=torch.int64)
        scale1_col = scale1_col.unsqueeze(0).repeat(batch_size, 1)
        angle1_col = torch.arange(self.angle_num, dtype=torch.int64)
        angle1_col = angle1_col.unsqueeze(0).repeat(batch_size, 1)
        scale2_col = scale1_col + scale_diff
        angle2_col = angle1_col + angle_diff
        angle2_col = (angle2_col + self.angle_num) % self.angle_num
        # the corresponding scale may be out of the range
        # so only keep the corresponding scales in the available range
        scale_inner_pos = ((scale2_col >= 0) & (scale2_col < self.scale_num))
        # generate the row index
        scale_row_ind = torch.arange(batch_size, dtype=torch.int64).cpu()
        scale_row = scale_row_ind.unsqueeze(1).repeat(1, self.scale_num)
        angle_row_ind = torch.arange(batch_size, dtype=torch.int64).cpu()
        angle_row = angle_row_ind.unsqueeze(1).repeat(1, self.angle_num)
        # sum the scores of the corrsponding scales and orientations
        scale_ls_sum = torch.full_like(scale1_ls, fill_value=-np.inf)
        scale_ls_sum[scale_row[scale_inner_pos], scale1_col[scale_inner_pos]] = (
            scale1_ls[scale_row[scale_inner_pos], scale1_col[scale_inner_pos]] +
            scale2_ls[scale_row[scale_inner_pos], scale2_col[scale_inner_pos]])
        angle_ls_sum = (angle1_ls[angle_row, angle1_col] +
                        angle2_ls[angle_row, angle2_col])
        # obtain the scale and orientation indexes corresponding to the maximal score
        scale_max_pos = torch.argmax(scale_ls_sum, dim=1).cpu()
        scale1_label = scale1_col[scale_row_ind, scale_max_pos]
        scale2_label = scale2_col[scale_row_ind, scale_max_pos]
        angle_max_pos = torch.argmax(angle_ls_sum, dim=1).cpu()
        angle1_label = angle1_col[angle_row_ind, angle_max_pos]
        angle2_label = angle2_col[angle_row_ind, angle_max_pos]

        if torch.sum(scale1_label < 0).item() > 0:
            print('error: scale1_label has negative value')
        if torch.sum(scale2_label < 0).item() > 0:
            print('error: scale2_label has negative value')
        if torch.sum(angle1_label < 0).item() > 0:
            print('error: angle1_label has negative value')
        if torch.sum(angle2_label < 0).item() > 0:
            print('error: angle2_label has negative value')

        return scale1_label, angle1_label, scale2_label, angle2_label

    def get_pred_error(self, scale1_ind, angle1_ind, scale2_ind, angle2_ind,
                       scale12, angle12):
        scale1 = torch.exp(self.scale_vec[scale1_ind])
        angle1 = self.angle_vec[angle1_ind]
        scale2 = torch.exp(self.scale_vec[scale2_ind])
        angle2 = self.angle_vec[angle2_ind]

        scale12_pred = scale2 / (scale1 + 1e-16)
        angle12_pred = (angle2 - angle1) % (math.pi * 2)
        angle12_pred[angle12_pred < -math.pi] += (math.pi * 2)
        angle12_pred[angle12_pred > math.pi] -= (math.pi * 2)
        # scale12 = scale12.repeat(9, 1)
        # angle12 = angle12.repeat(9, 1)

        scale_error_vec = torch.abs(1 - scale12_pred / scale12.squeeze(1))
        angle12_pred = angle12_pred.unsqueeze(1)
        angle_error_prob = torch.cat((angle12_pred - angle12, angle12_pred + math.pi * 2 - angle12,
                                      angle12_pred - math.pi * 2 - angle12), dim=1)
        angle_error_vec = torch.min(torch.abs(angle_error_prob), dim=1)[0]
        scale_error = torch.mean(scale_error_vec).item()
        angle_error = torch.mean(angle_error_vec).item()

        return scale_error, angle_error, scale1, angle1, scale2, angle2

    def get_rela_pred_error(self, scale12_ind, angle12_ind, scale12, angle12):
        scale12_pred = torch.exp(self.scale_rela_vec[scale12_ind])
        angle12_pred = self.angle_vec[angle12_ind]

        scale_error_vec = torch.abs(1 - scale12_pred / scale12)
        angle_error_prob = torch.cat((angle12_pred - angle12, angle12_pred + math.pi * 2 - angle12,
                                      angle12_pred - math.pi * 2 - angle12), dim=1)
        angle_error_vec = torch.min(torch.abs(angle_error_prob), dim=1)[0]
        scale_error = torch.mean(scale_error_vec).item()
        angle_error = torch.mean(angle_error_vec).item()
        return scale_error, angle_error, scale12_pred.squeeze(1), angle12_pred.squeeze(1)

    def map_scale_to_id(self, scale):
        log_scale = torch.log(scale)
        scale_diff = abs(log_scale - self.scale_vec)
        id = torch.argmin(scale_diff, dim=1, keepdim=True)
        return id

    def map_scale_rela_to_id(self, scale):
        log_scale = torch.log(scale)
        scale_diff = abs(log_scale - self.scale_rela_vec)
        id = torch.argmin(scale_diff, dim=1, keepdim=True)
        return id

    def map_angle_to_id(self, angle):
        angle_diff = abs(angle - self.angle_vec)
        id = torch.argmin(angle_diff, dim=1, keepdim=True)
        return id

    def map_id_to_scale(self, scale_ind):
        scale = torch.exp(self.scale_vec[scale_ind])
        return scale

    def map_id_to_angle(self, angle_ind):
        angle = self.angle_vec[angle_ind]
        return angle

    def split_forward(self, patches_s, patches_t):
        patch_all_num = patches_s.shape[0]
        each_num = 1000
        split_num = math.ceil(patch_all_num / each_num)
        z = torch.zeros(patch_all_num, self.z_channels, 1, 1, device=patches_s.device)
        for split_id in range(split_num):
            pos_now = slice(split_id * each_num, min((split_id + 1) * each_num, patch_all_num))
            patches_s_now = patches_s[pos_now]
            patches_t_now = patches_t[pos_now]
            z_s = self.encoder(patches_s_now)
            z_t = self.encoder(patches_t_now)
            assert z_s.shape[2] == 2 and z_s.shape[3] == 2 and z_t.shape[2] == 1 and z_t.shape[3] == 1
            # obtain the theta
            z_t_rep = z_t.repeat(1, 1, 2, 2)
            z_concat = torch.cat((z_s, z_t_rep), dim=1)
            z_now = self.fuse(z_concat)
            z[pos_now] = z_now
        z = z.view(patch_all_num, self.z_channels)
        para_base = self.regressor(z)
        theta_base = para_base[:, :4]
        ratio_base = para_base[:, 4]
        bias_base = para_base[:, 5]
        # construct the theta
        theta = theta_base.view(patch_all_num, 2, 2)
        zero_dxdy = torch.zeros(patch_all_num, 2, 1, dtype=theta.dtype, device=theta.device)
        theta = torch.cat((theta, zero_dxdy), dim=2)
        # construct the gray para
        ratio = torch.tanh(ratio_base) * 0.2 + 1
        bias = torch.tanh(bias_base)
        return theta, ratio, bias
