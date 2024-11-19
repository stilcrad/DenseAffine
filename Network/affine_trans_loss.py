import random
from affine_model import *
from Local_affine_estimation.S3Esti.abso_esti_net import EstiNet

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device( 'cpu')

def get_rand_vec(limit, patch_num):
    # generate the value randomly
    rand_vec = (limit[0] + (limit[1] - limit[0]) * np.random.random((patch_num,)))
    return rand_vec


def get_random_border_point(border_x1_ratio, point_num):
    border_x2_ratio = 1 - border_x1_ratio
    x_ratio = np.random.random((point_num,))
    x_inner_pos = ((x_ratio > border_x1_ratio) & (x_ratio < border_x2_ratio))
    x_out_pos = (~x_inner_pos)
    x_out_num = np.sum(x_out_pos)
    # re-sample the points if the original points are close to the image border
    min_border_ratio = 0.2
    if x_out_num > 0:
        # randomly sample the points in the left or right border
        side2_pos = (np.random.random((x_out_num,)) > 0.5)
        among_border_ratio = ((1 - min_border_ratio) * np.random.random((x_out_num,)) +
                              min_border_ratio)
        new_x_ratio = among_border_ratio * border_x1_ratio[x_out_pos]
        new_x_ratio[side2_pos] = 1 - new_x_ratio[side2_pos]
        x_ratio[x_out_pos] = new_x_ratio
    return x_ratio


def get_affine_patch(tensor, centers, patch_size, scale_vec, angle_vec,
                     scale_ratio_list, is_train):
    # use different "align_corners" parameters according to the version of pytorch
    torch_version = float('.'.join((torch.__version__).split('.')[:2]))
    torch_version_thre = 1.2

    patch_num = centers.shape[0]
    patch_size_half = round(patch_size / 2)
    corner = np.array([[-patch_size_half, -patch_size_half],
                       [-patch_size_half, patch_size_half],
                       [patch_size_half, -patch_size_half],
                       [patch_size_half, patch_size_half]]).astype('float32')
    # generate the scales and orientations randomly
    scale_vec = scale_vec.cpu().numpy() if torch.is_tensor(scale_vec) else scale_vec
    angle_vec = angle_vec.cpu().numpy() if torch.is_tensor(angle_vec) else angle_vec
    centers = centers.cpu().numpy() if torch.is_tensor(centers) else centers
    sin_angle = np.sin(angle_vec)
    cos_angle = np.cos(angle_vec)
    # scale_ratio_list: the ratios used to generate the multiple-scale input patches
    patch_list = []
    for k, scale_ratio_now in enumerate(scale_ratio_list):
        # during the training, randomly resize the image and then crop the patches
        if is_train:
            pre_ratio_l = [0.8, 1.5]
        else:
            # during the testing, directly crop the patches on the original image
            pre_ratio_l = [1, 1]
        rand_pre_ratio = pre_ratio_l[0] + (pre_ratio_l[1] - pre_ratio_l[0]) * random.random()
        tensor_now = F.interpolate(tensor, scale_factor=rand_pre_ratio,
                                   mode='bilinear', align_corners=True)
        scale_vec_now = scale_vec * rand_pre_ratio
        centers_now = centers * rand_pre_ratio
        channel_num, row, col = tensor_now.shape[1:4]

        mat_list = [get_trans_mat(sin_angle[pos], cos_angle[pos], scale_vec_now[pos] * scale_ratio_now)
                    for pos in range(patch_num)]
        trans_corner = [cv2.perspectiveTransform(corner[np.newaxis, :], H_mat)
                        for H_mat in mat_list]
        trans_corner = [trans_corner[pos].squeeze(0) + centers_now[pos:pos + 1] for
                        pos in range(patch_num)]
        # get the transformation parameters
        # the coordinates should be mapped to [-1,1] to satisfy the affine_grid process
        corner_norm = corner / patch_size_half
        trans_corner_norm = [get_norm_xy(item, row, col) for item in trans_corner]
        theta = [cv2.getPerspectiveTransform(corner_norm, trans_corner_norm[pos])[np.newaxis, :]
                 for pos in range(patch_num)]
        theta = np.concatenate(theta, axis=0)
        theta = torch.from_numpy(theta[:, :2, :].astype('float32'))
        # get the transformed patches
        grid_size = torch.Size((theta.shape[0], channel_num, patch_size, patch_size))
        if torch_version > torch_version_thre:
            grid = F.affine_grid(theta, grid_size, align_corners=True)
        else:
            grid = F.affine_grid(theta, grid_size)
        grid = grid.view(1, grid.shape[0], patch_size * patch_size, 2)
        grid = grid.to(tensor_now.device)
        if torch_version > torch_version_thre:
            patch_now = F.grid_sample(tensor_now, grid, padding_mode='zeros',
                                      mode='nearest', align_corners=True)
        else:
            patch_now = F.grid_sample(tensor_now, grid, padding_mode='zeros', mode='nearest')
        patch_now = patch_now.view(tensor_now.shape[1], patch_num, patch_size, patch_size)
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
    tensor = torch.from_numpy(image_np).float()
    new_limit = [-1, 1]
    tensor = new_limit[0] + (tensor / 255) * (new_limit[1] - new_limit[0])
    return tensor


def load_file_current_dir(dir_path):
    all_list = os.listdir(dir_path)
    all_list = sorted(all_list)
    all_list = [os.path.join(dir_path, name_now) for name_now in all_list]
    file_list = []
    for name_now in all_list:
        if os.path.isfile(name_now):
            if os.path.splitext(name_now)[-1] in avail_image_ext:
                file_list.append(name_now)
            else:
                continue
        else:
            sub_file_list = load_file_current_dir(name_now)
            file_list.extend(sub_file_list)

    return file_list




class affine_trans_loss(nn.Module):
    def __init__(self,data):
        super(affine_trans_loss, self).__init__()

        self.scale_ratio_list = [0.5, 1, 2]
        self.scale_limit = [1.0, 3.0]
        self.patch_size = 32

        self.angle_limit = [-math.pi, math.pi]
        self.is_train = True
        self.forward(data)

    def homogenize(self, coord):
        coord = torch.cat((coord, torch.ones_like(coord[:, :, [0]])), -1)
        return coord

    def epipolar_cost(self, coord1, coord2, fmatrix):
        coord1_h = self.homogenize(coord1).transpose(1, 2)
        coord2_h = self.homogenize(coord2).transpose(1, 2)
        epipolar_line = fmatrix.bmm(coord1_h)  # Bx3xn
        epipolar_line_ = epipolar_line / torch.clamp(torch.norm(epipolar_line[:, :2, :], dim=1, keepdim=True), min=1e-8)
        essential_cost = torch.abs(torch.sum(coord2_h * epipolar_line_, dim=1))  # Bxn
        return essential_cost

    def epipolar_loss(self, coord1, coord2, fmatrix, weight):
        essential_cost = self.epipolar_cost(coord1, coord2, fmatrix)
        loss = torch.mean(weight * essential_cost)
        return loss

    def affine_loss(self,coord1,coord2,fmatrix,pred_A,index ):

        fmatrix_ = fmatrix.unsqueeze(0).cuda()
        if type(pred_A) != torch.Tensor:
            pred_A = torch.tensor(pred_A).cuda()

        if pred_A.shape[-1] != 4:
            pred_A = pred_A.reshape(pred_A.shape[0],4)

        a1 = pred_A[:,0]
        a2 = pred_A[:,1]
        a3 = pred_A[:,2]
        a4 = pred_A[:,3]
        u1 = coord1[:,0]
        v1 = coord1[:,1]
        u2 = coord2[:,0]
        v2 = coord2[:,1]
        f1 =fmatrix_[:,0,0]
        f2 =fmatrix_[:,0,1]
        f3 =fmatrix_[:,0,2]
        f4 =fmatrix_[:,1,0]
        f5 =fmatrix_[:,1,1]
        f6 =fmatrix_[:,1,2]
        f7 =fmatrix_[:,2,0]
        f8 =fmatrix_[:,2,1]
        f9 =fmatrix_[:,2,2]

        loss_a =torch.mul((u2+torch.mul(a1,u1)),f1) +torch.mul(torch.mul(a1,v1),f2) +torch.mul(a1,f3) +torch.mul((v2 +torch.mul(a3,u1)),f4) + torch.mul(torch.mul(a3,v1),f5) +torch.mul(a3,f6) +f7

        loss_b = torch.mul(torch.mul(a2,u1),f1) + torch.mul((u2+torch.mul(a2,v1)),f2) + torch.mul(a2,f3) +torch.mul(torch.mul(a4,u1),f4) + torch.mul((v2+torch.mul(a4,v1)),f5) +torch.mul(a4,f6)+f8

        A_loss = torch.median(torch.abs(loss_a ))+torch.median(torch.abs(loss_b))

        return torch.clamp(A_loss,min = 0,max=0.100)

    def affine_loss_withE(self,K1, K2, pt1, pt2, Essential, pred_A):

        pt1_homogeneous = torch.cat((pt1.cpu(), torch.ones((len(pt1), 1))), dim=1)
        pt2_homogeneous = torch.cat((pt2, torch.ones((len(pt2), 1)).cuda()), dim=1)

        # Calculate inverse of camera calibration matrices
        K1_inv = torch.inverse(K1).cuda()
        K2_inv = torch.inverse(K2).cuda()

        # Normalize points to obtain normalized image coordinates
        pt1_normalized = torch.matmul(pt1_homogeneous, K1_inv.t())
        pt2_normalized = torch.matmul(pt2_homogeneous, K2_inv.t())

        Essential_ = Essential

        pred_A= pred_A.reshape(pred_A.shape[0],4)

        a1 = pred_A[:, 0]
        a2 = pred_A[:, 1]
        a3 = pred_A[:, 2]
        a4 = pred_A[:, 3]
        u1 = pt1_normalized[:, 0]
        v1 = pt1_normalized[:, 1]
        u2 = pt2_normalized[:, 0]
        v2 = pt2_normalized[:, 1]
        e1 = Essential_[0, 0]
        e2 = Essential_[0, 1]
        e3 = Essential_[0, 2]
        e4 = Essential_[1, 0]
        e5 = Essential_[1, 1]
        e6 = Essential_[1, 2]
        e7 = Essential_[2, 0]
        e8 = Essential_[2, 1]
        e9 = Essential_[2, 2]

        loss_a = torch.mul((u2 + torch.mul(a1, u1)), e1) + torch.mul(torch.mul(a1, v1), e2) + torch.mul(a1,
                                                                                                        e3) + torch.mul(
            (v2 + torch.mul(a3, u1)), e4) + torch.mul(torch.mul(a3, v1), e5) + torch.mul(a3, e6) + e7

        loss_b = torch.mul(torch.mul(a2, u1), e1) + torch.mul((u2 + torch.mul(a2, v1)), e2) + torch.mul(a2,
                                                                                                        e3) + torch.mul(
            torch.mul(a4, u1), e4) + torch.mul((v2 + torch.mul(a4, v1)), e5) + torch.mul(a4, e6) + e8

        A_loss = torch.abs(loss_a) + torch.abs(loss_b)

        return torch.median(A_loss)

    def compute_epipolar_line_error_withE(self,K1, K2, E, pt1, pt2):
        # Convert points to homogeneous coordinates
        if type(E) == np.ndarray:
            E = torch.tensor(E).cuda().float()
        pt1_homogeneous = torch.cat((pt1, torch.ones((len(pt1), 1)).cuda()), dim=1)
        pt2_homogeneous = torch.cat((pt2, torch.ones((len(pt2), 1)).cuda()), dim=1)

        # Calculate inverse of camera calibration matrices
        K1_inv = torch.inverse(K1).cuda()
        K2_inv = torch.inverse(K2).cuda()

        # Normalize points to obtain normalized image coordinates
        pt1_normalized = torch.matmul(pt1_homogeneous, K1_inv.t())
        pt2_normalized = torch.matmul(pt2_homogeneous, K2_inv.t())

        # Compute epipolar lines in the second image
        epipolar_lines = torch.matmul(pt1_normalized, E.t())

        # Compute distances between points and their corresponding epipolar lines
        distances = torch.abs(torch.sum(epipolar_lines * pt2_normalized, dim=1)) / torch.sqrt(
            epipolar_lines[:, 0] ** 2 + epipolar_lines[:, 1] ** 2)

        return torch.median(distances)

    def expand_homo_ones(self,arr2d, axis=1):
        """Raise 2D array to homogenous coordinates
        Args:
            - arr2d: (N, 2) or (2, N)
            - axis: the axis to append the ones
        """
        if axis == 0:
            ones = np.ones((1, arr2d.shape[1]))
        else:
            ones = np.ones((arr2d.shape[0], 1))
        return np.concatenate([arr2d, ones], axis=axis)

    def affine_sampson_distance(self,pts1, pts2, affine, homos=False):
        if homos:
            pts1 = self.expand_homo_ones(pts1, axis=1)  # if pts1.shape[1] == 2 else pts1
            pts2 = self.expand_homo_ones(pts2, axis=1)  # if pts2.shape[1] == 2 else pts2
        F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.USAC_ACCURATE, ransacReprojThreshold=0.999,
                                         confidence=0.999)
        F = torch.tensor(F,dtype=torch.DoubleTensor())
        f11, f12, f13, f21, f22, f23, f31, f32, f33 = F.reshape(9)
        dist_list = []
        for i in range(pts1.shape[0]):
            pt1 = pts1[i]
            pt2 = pts2[i]
            x1, y1 = pt1
            x2, y2 = pt2
            a11, a12, a21, a22 = affine[i].reshape(4)
            f11, f12, f13, f21, f22, f23, f31, f32, f33 = F.reshape(9)
            M0 = x1 * (a11 * f11 + a21 * f21) + y1 * (
                        a11 * f12 + a21 * f22) + a11 * f13 + a21 * f23 + f11 * x2 + f21 * y2 + f31
            M1 = f13 + f11 * x2 + f12 * y1
            M2 = a11 * f12 + a21 * f22
            M3 = f11
            M4 = f23 + f21 * x1 + f22 * y1
            M5 = a11 * f11 + a21 * f21
            M6 = f21

            N0 = x1 * (a12 * f11 + a22 * f21) + y1(
                a12 * f12 + a22 * f22) + a12 * f13 + a22 * f23 + f12 * x2 + f22 * y2 + f32
            N1 = f13 + f11 * x1 + f12 * y1
            N2 = a12 * f11 + a22 * f21
            N3 = f12
            N4 = f23 + f21 * x1 + f22 * y1
            N5 = a12 * f12 + a22 * f22
            N6 = f22

            dist1 = M0 ** 2 / (M1 ** 2 + M2 ** 2 + M3 ** 2 + M4 * 2 + M5 ** 2 + M6 ** 2)
            dist2 = N0 ** 2 / (N1 ** 2 + N2 ** 2 + N3 ** 2 + N4 * 2 + N5 ** 2 + N6 ** 2)

            dist = torch.mean(dist1 + dist2)
            dist_list.append(dist)

        f_dist = np.mean(dist_list)

        return f_dist

    def affine_sampson_distance_Only_affine(self,pts1, pts2, affine, homos=False):
        if homos:
            pts1 = self.expand_homo_ones(pts1, axis=1)  # if pts1.shape[1] == 2 else pts1
            pts2 = self.expand_homo_ones(pts2, axis=1)  # if pts2.shape[1] == 2 else pts2
        F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.USAC_ACCURATE, ransacReprojThreshold=0.999,
                                         confidence=0.999)
        f11, f12, f13, f21, f22, f23, f31, f32, f33 = F.reshape(9)
        dist_list = []
        for i in range(pts1.shape[0]):
            pt1 = pts1[i]
            pt2 = pts2[i]
            x1, y1 = pt1
            x2, y2 = pt2
            a11, a12, a21, a22 = affine[i].reshape(4)
            f11, f12, f13, f21, f22, f23, f31, f32, f33 = F.reshape(9)
            M0 = x1 * (a11 * f11 + a21 * f21) + y1 * (
                    a11 * f12 + a21 * f22) + a11 * f13 + a21 * f23 + f11 * x2 + f21 * y2 + f31
            M1 = f13 + f11 * x2 + f12 * y1
            M2 = a11 * f12 + a21 * f22
            M3 = f11
            M4 = f23 + f21 * x1 + f22 * y1
            M5 = a11 * f11 + a21 * f21
            M6 = f21

            N0 = x1 * (a12 * f11 + a22 * f21) + y1(
                a12 * f12 + a22 * f22) + a12 * f13 + a22 * f23 + f12 * x2 + f22 * y2 + f32
            N1 = f13 + f11 * x1 + f12 * y1
            N2 = a12 * f11 + a22 * f21
            N3 = f12
            N4 = f23 + f21 * x1 + f22 * y1
            N5 = a12 * f12 + a22 * f22
            N6 = f22

            dist1 = M0 ** 2 / (M1 ** 2 + M4 * 2)
            dist2 = N0 ** 2 / (N1 ** 2 + N4 * 2)

            dist = torch.mean(dist1 + dist2)
            dist_list.append(dist)

        f_dist = np.mean(dist_list)

        return f_dist


    def self_supervised(self, data):

        image= data["out"]["im1"]
        centers = data["coor1"][:32]
        row, col = image.shape[0], image.shape[1]
        # image = self.flip_transpose_image(image_ori)
        images = np.tile(image[np.newaxis, :], [2, 3, 1, 1])
        # during the training, argument the training samples with the
        # illumination changes to improve the generalization ability
        # if self.is_train:
        #     # different images will be transformed with different illumination changes
        #     images = self.random_ill_change(images)

        # randomly generate the scale and angle parameters
        tensors = to_tensor(images)
        tensor0 = tensors[0:1]
        tensor1 = tensors[1:2]
        self.patch_num = 32
        angle0 = get_rand_vec(self.angle_limit, self.patch_num)
        angle1 = get_rand_vec(self.angle_limit, self.patch_num)
        # randomly sample the scale of the first image
        scale0 = get_rand_vec([1, self.scale_limit[1]], self.patch_num)
        # randomly sample the scale change, and then obtain the scale of the second image
        scale1_rela_ratio = get_rand_vec([1, self.scale_limit[1]], self.patch_num)
        scale1 = scale0 * scale1_rela_ratio
        # randomly swap the items in scale0 and scale1 to improve the generalization
        # exchange_pos = (np.random.random((self.patch_num,)) < 0.5)
        # scale0_rem = scale0[exchange_pos]
        # scale0[exchange_pos] = scale1[exchange_pos]
        # scale1[exchange_pos] = scale0_rem

        # sample the centroid point of the patches
        # the sample border is determined by the scale range
        # obtain the x coordinate
        dist_vec = scale0 * (self.patch_size / 2)
        border_x1_ratio = dist_vec / col
        x_ratio = get_random_border_point(border_x1_ratio, self.patch_num)
        x_vec = x_ratio * col
        # obtain the y coordinate
        border_y1_ratio = dist_vec / row
        y_ratio = get_random_border_point(border_y1_ratio, self.patch_num)
        y_vec = y_ratio * row
        # centers = np.c_[x_vec, y_vec].astype('float32')

        # extract the patches according to the scale and angle parameters
        patch0 = get_affine_patch(
            tensor0, centers, self.patch_size, scale0, angle0,
            self.scale_ratio_list, self.is_train)
        patch1 = get_affine_patch(
            tensor1, centers, self.patch_size, scale1, angle1,
            self.scale_ratio_list, self.is_train)

        # get the relative scales and angles
        scale0_to1 = torch.from_numpy(1 / (scale1 / scale0))
        angle0_to1 = torch.from_numpy(angle1 - angle0)
        angle0_to1 = angle0_to1 % (math.pi * 2)
        # guarantee that the angle difference are in [-pi,pi]
        angle0_to1[angle0_to1 < -math.pi] += (math.pi * 2)
        angle0_to1[angle0_to1 > math.pi] -= (math.pi * 2)

        if self.is_train:
            sample = {'patch1': patch0, 'patch2': patch1,
                      'scale': scale0_to1, 'angle': angle0_to1,
                      'center': centers, }
        else:
            sample = {'patch1': patch0, 'patch2': patch1,
                      'scale': scale0_to1, 'angle': angle0_to1,
                      'full_image': tensors, 'center': centers,
                      }
        batch =  sample

        patch1, patch2, scale, angle = \
            batch['patch1'], batch['patch2'], batch['scale'], batch['angle']
        batch_size = patch1.shape[0]
        patch1 = batch['patch1'].view(tuple(np.r_[batch_size * patch1.shape[1],
                                                  patch1.shape[2:]]))
        patch2 = batch['patch2'].view(tuple(np.r_[batch_size * patch2.shape[1],
                                                  patch2.shape[2:]]))
        scale12 = batch['scale'].view((batch_size , 1))
        # scale12 = batch['scale'].view((batch_size * scale.shape[0], 1))
        angle12 = batch['angle'].view((batch_size , 1))
        # angle12 = batch['angle'].view((batch_size * angle.shape[0], 1))

        patch1 = patch1.to(device)
        patch2 = patch2.to(device)
        scale12 = scale12.to(device)
        angle12 = angle12.to(device)

        patch_num_now = patch1.shape[0]
        scale_num = 300
        angle_num = 360
        patch_size = 32
        scale_ratio_list = [0.5, 1, 2]
        esti_scale_ratio_list = [0.5, 1, 2]

        model_scale = EstiNet(need_bn=True, device=device, out_channels=scale_num,
                              patch_size=patch_size, scale_ratio=esti_scale_ratio_list)
        model_angle = EstiNet(need_bn=True, device=device, out_channels=angle_num,
                              patch_size=patch_size, scale_ratio=esti_scale_ratio_list)

        model_scale.cuda()
        model_angle.cuda()

        patch1_ = patch1.unsqueeze(1).reshape(32, 9, 32, -1).cuda()
        patch2_ = patch2.unsqueeze(1).reshape(32, 9, 32, -1).cuda()

        scale1_resp = model_scale(patch1_)
        angle1_resp = model_angle(patch1_)
        scale2_resp = model_scale(patch2_)
        angle2_resp = model_angle(patch2_)
        # obtain the labels of the relative scale and orientation
        scale12_label = model_scale.map_scale_rela_to_id(scale12)
        angle12_label = model_scale.map_angle_to_id(angle12)
        # obtain the absolute labels by minimize the current loss
        with torch.no_grad():
            scale1_label, angle1_label, scale2_label, angle2_label = \
                model_scale.get_max_label(scale1_resp, angle1_resp, scale2_resp,
                                          angle2_resp, scale12, angle12)
        # compute the current estimation error
        with torch.no_grad():
            # validate the correctness of the relative labels (helpful to debug)
            scale_rela_label_error, angle_rela_label_error, _, _ = model_scale.get_rela_pred_error(
                scale12_label, angle12_label, scale12, angle12)
            # compute the current estimation error between the predictions and labels
            scale1_ind_pred = torch.argmax(scale1_resp, dim=1)
            angle1_ind_pred = torch.argmax(angle1_resp, dim=1)
            scale2_ind_pred = torch.argmax(scale2_resp, dim=1)
            angle2_ind_pred = torch.argmax(angle2_resp, dim=1)
            scale_error, angle_error, _, _, _, _ = model_scale.get_pred_error(
                scale1_ind_pred, angle1_ind_pred, scale2_ind_pred,
                angle2_ind_pred, scale12, angle12)
            # validate the correctness of the absolute labels (helpful to debug)
            scale_label_error, angle_label_error, _, _, _, _ = model_scale.get_pred_error(
                scale1_label, angle1_label, scale2_label, angle2_label,
                scale12, angle12)

        # compute the classification loss
        scale1_label, angle1_label, scale2_label, angle2_label = \
            (scale1_label.to(device), angle1_label.to(device),
             scale2_label.to(device), angle2_label.to(device))

        CE_loss = torch.nn.CrossEntropyLoss()
        loss_scale1 = CE_loss(scale1_resp, scale1_label) + CE_loss(scale2_resp, scale2_label)
        loss_angle1 = CE_loss(angle1_resp, angle1_label) + CE_loss(angle2_resp, angle2_label)

        loss1 = loss_scale1 + loss_angle1


        image = data["out"]["im2"]
        centers = data["coor2"][:32]
        row, col = image.shape[0], image.shape[1]
        # image = self.flip_transpose_image(image_ori)
        images = np.tile(image[np.newaxis, :], [2, 3, 1, 1])
        # during the training, argument the training samples with the
        # illumination changes to improve the generalization ability
        # if self.is_train:
        #     # different images will be transformed with different illumination changes
        #     images = self.random_ill_change(images)

        # randomly generate the scale and angle parameters
        tensors = to_tensor(images)
        tensor0 = tensors[0:1]
        tensor1 = tensors[1:2]
        self.patch_num = 32
        angle0 = get_rand_vec(self.angle_limit, self.patch_num)
        angle1 = get_rand_vec(self.angle_limit, self.patch_num)
        # randomly sample the scale of the first image
        scale0 = get_rand_vec([1, self.scale_limit[1]], self.patch_num)
        # randomly sample the scale change, and then obtain the scale of the second image
        scale1_rela_ratio = get_rand_vec([1, self.scale_limit[1]], self.patch_num)
        scale1 = scale0 * scale1_rela_ratio
        # randomly swap the items in scale0 and scale1 to improve the generalization
        exchange_pos = (np.random.random((self.patch_num,)) < 0.5)
        scale0_rem = scale0[exchange_pos]
        scale0[exchange_pos] = scale1[exchange_pos]
        scale1[exchange_pos] = scale0_rem

        # sample the centroid point of the patches
        # the sample border is determined by the scale range
        # obtain the x coordinate
        dist_vec = scale0 * (self.patch_size / 2)
        border_x1_ratio = dist_vec / col
        x_ratio = get_random_border_point(border_x1_ratio, self.patch_num)
        x_vec = x_ratio * col
        # obtain the y coordinate
        border_y1_ratio = dist_vec / row
        y_ratio = get_random_border_point(border_y1_ratio, self.patch_num)
        y_vec = y_ratio * row
        # centers = np.c_[x_vec, y_vec].astype('float32')

        # extract the patches according to the scale and angle parameters
        patch0 = get_affine_patch(
            tensor0, centers, self.patch_size, scale0, angle0,
            self.scale_ratio_list, self.is_train)
        patch1 = get_affine_patch(
            tensor1, centers, self.patch_size, scale1, angle1,
            self.scale_ratio_list, self.is_train)

        # get the relative scales and angles
        scale0_to1 = torch.from_numpy(1 / (scale1 / scale0))
        angle0_to1 = torch.from_numpy(angle1 - angle0)
        angle0_to1 = angle0_to1 % (math.pi * 2)
        # guarantee that the angle difference are in [-pi,pi]
        angle0_to1[angle0_to1 < -math.pi] += (math.pi * 2)
        angle0_to1[angle0_to1 > math.pi] -= (math.pi * 2)

        if self.is_train:
            sample = {'patch1': patch0, 'patch2': patch1,
                      'scale': scale0_to1, 'angle': angle0_to1,
                      'center': centers, }
        else:
            sample = {'patch1': patch0, 'patch2': patch1,
                      'scale': scale0_to1, 'angle': angle0_to1,
                      'full_image': tensors, 'center': centers,
                      }
        batch = sample

        patch1, patch2, scale, angle = \
            batch['patch1'], batch['patch2'], batch['scale'], batch['angle']
        batch_size = patch1.shape[0]
        patch1 = batch['patch1'].view(tuple(np.r_[batch_size * patch1.shape[1],
                                                  patch1.shape[2:]]))
        patch2 = batch['patch2'].view(tuple(np.r_[batch_size * patch2.shape[1],
                                                  patch2.shape[2:]]))
        scale12 = batch['scale'].view((batch_size, 1))
        # scale12 = batch['scale'].view((batch_size * scale.shape[0], 1))
        angle12 = batch['angle'].view((batch_size, 1))
        # angle12 = batch['angle'].view((batch_size * angle.shape[0], 1))

        patch1 = patch1.to(device)
        patch2 = patch2.to(device)
        scale12 = scale12.to(device)
        angle12 = angle12.to(device)

        # patch_num_now = patch1.shape[0]
        # scale_num = 300
        # angle_num = 360
        # patch_size = 32
        # scale_ratio_list = [0.5, 1, 2]
        # esti_scale_ratio_list = [0.5, 1, 2]
        #
        # model_scale = EstiNet(need_bn=True, device=device, out_channels=scale_num,
        #                       patch_size=patch_size, scale_ratio=esti_scale_ratio_list)
        # model_angle = EstiNet(need_bn=True, device=device, out_channels=angle_num,
        #                       patch_size=patch_size, scale_ratio=esti_scale_ratio_list)
        #
        # model_scale.cuda()
        # model_angle.cuda()

        patch1_ = patch1.unsqueeze(1).reshape(-1, 9, 32, 32).cuda()
        patch2_ = patch2.unsqueeze(1).reshape(-1, 9, 32, 32).cuda()

        scale1_resp = model_scale(patch1_)
        angle1_resp = model_angle(patch1_)
        scale2_resp = model_scale(patch2_)
        angle2_resp = model_angle(patch2_)
        # obtain the labels of the relative scale and orientation
        scale12_label = model_scale.map_scale_rela_to_id(scale12)
        angle12_label = model_scale.map_angle_to_id(angle12)
        # obtain the absolute labels by minimize the current loss
        with torch.no_grad():
            scale1_label, angle1_label, scale2_label, angle2_label = \
                model_scale.get_max_label(scale1_resp, angle1_resp, scale2_resp,
                                          angle2_resp, scale12, angle12)
        # compute the current estimation error
        with torch.no_grad():
            # validate the correctness of the relative labels (helpful to debug)
            scale_rela_label_error, angle_rela_label_error, _, _ = model_scale.get_rela_pred_error(
                scale12_label, angle12_label, scale12, angle12)
            # compute the current estimation error between the predictions and labels
            scale1_ind_pred = torch.argmax(scale1_resp, dim=1)
            angle1_ind_pred = torch.argmax(angle1_resp, dim=1)
            scale2_ind_pred = torch.argmax(scale2_resp, dim=1)
            angle2_ind_pred = torch.argmax(angle2_resp, dim=1)
            scale_error, angle_error, _, _, _, _ = model_scale.get_pred_error(
                scale1_ind_pred, angle1_ind_pred, scale2_ind_pred,
                angle2_ind_pred, scale12, angle12)
            # validate the correctness of the absolute labels (helpful to debug)
            scale_label_error, angle_label_error, _, _, _, _ = model_scale.get_pred_error(
                scale1_label, angle1_label, scale2_label, angle2_label,
                scale12, angle12)

        # compute the classification loss
        scale1_label, angle1_label, scale2_label, angle2_label = \
            (scale1_label.to(device), angle1_label.to(device),
             scale2_label.to(device), angle2_label.to(device))

        CE_loss = torch.nn.CrossEntropyLoss()
        loss_scale2 = CE_loss(scale1_resp, scale1_label) + CE_loss(scale2_resp, scale2_label)
        loss_angle2 = CE_loss(angle1_resp, angle1_label) + CE_loss(angle2_resp, angle2_label)

        loss2 = loss_scale2 + loss_angle2

        loss = 0.5 *(loss1 +loss2)

        return loss


    def forward(self, data):
        pred_A = data['pred_A']
        coord1_A = data["coor1"]
        coord2_A = data["coor2"]
        F_gt  = data["F_gt"]
        E_gt =  data["E_gt"]
        _index = data["_index"]
        K1 = data["K1"]
        K2 = K1

        try:
            self_sup_loss = self.self_supervised(data)

        except RuntimeError :
            self_sup_loss = 20

        # affine_loss = self.affine_loss_withE(coord1_A, coord2_A, F_gt, pred_A,_index)
        affine_loss = self.affine_loss_withE(K1, K2, coord1_A, coord2_A, E_gt, pred_A)
        epipolar_loss = self.compute_epipolar_line_error_withE(K1, K2, E_gt, coord1_A, coord2_A)

        loss_all = self_sup_loss + affine_loss

        return loss_all


