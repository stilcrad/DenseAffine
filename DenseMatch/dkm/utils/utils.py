import numpy
import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
from PIL import Image
import pygcransac
# from gcransac_parameter_types import *
from copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Code taken from https://github.com/PruneTruong/DenseMatching/blob/40c29a6b5c35e86b9509e65ab0cd12553d998e5f/validation/utils_pose_estimation.py
# --- GEOMETRY ---
def estimate_pose(kpts0, kpts1, K0, K1, norm_thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    K0inv = np.linalg.inv(K0[:2,:2])
    K1inv = np.linalg.inv(K1[:2,:2])

    kpts0 = (K0inv @ (kpts0-K0[None,:2,2]).T).T 
    kpts1 = (K1inv @ (kpts1-K1[None,:2,2]).T).T

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf, method=cv2.USAC_ACCURATE
    )

    print("threshold",(mask[mask == [1]].shape[0])/mask.shape[0])
    ret = None
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t, mask.ravel() > 0)
    return ret

def get_affine_correspondences(lafs1, lafs2,  tentatives):
    # Initialize the ACs
    ACs = np.zeros((len(tentatives), 8))
    row = 0
    # Iterate through all tentative correspondences and them to the matrix
    for m in tentatives:
        # The LAF in the source image
        LAF1 = lafs1[m]
        # The LAF in the destination image
        LAF2 = lafs2[m]
        # Calculating the local affine transformation as A = LAF2 * inv(LAF1)
        A = np.matmul(LAF2[:, :2], np.linalg.inv(LAF1[:, :2]))

        # Saving the coordinates in the ACs matrix
        ACs[row, 0] = LAF1[0, 2]
        ACs[row, 1] = LAF1[1, 2]
        ACs[row, 2] = LAF2[0, 2]
        ACs[row, 3] = LAF2[1, 2]
        ACs[row, 4] = A[0, 0]
        ACs[row, 5] = A[0, 1]
        ACs[row, 6] = A[1, 0]
        ACs[row, 7] = A[1, 1]
        row += 1

    return np.float32(ACs)

def get_probabilities(tentatives):
    probabilities = []
    # Since the correspondences are assumed to be ordered by their SNN ratio a priori,
    # we just assign a probability according to their order.
    for i in range(len(tentatives)):
        probabilities.append(1.0 - i / len(tentatives))

    probabilities = list(numpy.array(probabilities).astype(np.float64))
    return probabilities



def verify_pygcransac_ess_for_pc(kps1, kps2, tentatives, K1, K2, h1, w1, h2, w2, sampler_id):
    # Copy the coordinates in the destination image selected by the tentative correspondences
    # NG-RANSAC and AR-Sampler require an inlier probability to be provided for each point.
    # Since deep learning-based prediction is not applied here, we calculate the probabilities
    # from the SNN ratio ranks.tentatives
    tentatives = [i for i in range(len(kps1))]
    correspondences = np.float32([np.hstack((kps1[m], kps2[m])) for m in tentatives]).reshape(len(kps1), 4)
    if sampler_id == 3 or sampler_id == 4:
        inlier_probabilities = get_probabilities(tentatives)

    E, mask = pygcransac.findEssentialMatrix(
        np.ascontiguousarray(correspondences.astype(np.float64)),  # Point correspondences in the two images
        K1,  # Intrinsic camera parameters of the source image
        K2,  # Intrinsic camera parameters of the destination image
        h1, w1, h2, w2,  # The sizes of the images
        probabilities=inlier_probabilities,
        # Inlier probabilities. This is not used if the sampler is not 3 (NG-RANSAC) or 4 (AR-Sampler)
        threshold=0.75,  # Inlier-outlier threshold
        conf=0.9,  # RANSAC confidence
        min_iters=500,
        # The minimum iteration number in RANSAC. If time does not matter, I suggest setting it to, e.g., 1000
        max_iters=50000,  # The maximum iteration number in RANSAC
        sampler=sampler_id)  # Sampler index (0 - Uniform, 1 - PROSAC, 2 - P-NAPSAC, 3 - NG-RANSAC, 4 - AR-Sampler)
    print(deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    return E, mask

def verify_affine_pygcransac_essential_matrix(ACs, tentatives, h1, w1, h2, w2, K1, K2, sampler_id):
    # NG-RANSAC and AR-Sampler require an inlier probability to be provided for each point.
    # Since deep learning-based prediction is not applied here, we calculate the probabilities
    # from the SNN ratio ranks. This part can be straightforwardly replaced by a deep probability
    # predictor, e.g., NG-RANSAC, CLNet, OANet, Deep MAGSAC++.
    inlier_probabilities = []
    if sampler_id == 3 or sampler_id == 4:
        inlier_probabilities = get_probabilities(tentatives)

    # Run GC-RANSAC with the AC-based solver
    #findEssentialMatrix(correspondences: numpy.ndarray[numpy.float64], K1: numpy.ndarray[numpy.float64], K2: numpy.ndarray[numpy.float64], h1: int, w1: int, h2: int, w2: int, probabilities: numpy.ndarray[numpy.float64], threshold: float = 1.0, conf: float = 0.99, spatial_coherence_weight: float = 0.1, max_iters: int = 10000, min_iters: int = 50, use_sprt: bool = True, min_inlier_ratio_for_sprt: float = 0.01, sampler: int = 1, neighborhood: int = 1, neighborhood_size: float = 20.0, lo_number: int = 50, sampler_variance: float = 0.1, solver: int = 0) -> tuple

    E, mask = pygcransac.findEssentialMatrix(
        np.ascontiguousarray(ACs.astype(np.float64)),  # The input affine correspondences
        K1,  # Intrinsic camera parameters of the source image
        K2,  # Intrinsic camera parameters of the destination image
        int(h1), int(w1), int(h2), int(w2),
        probabilities=inlier_probabilities, # The sizes of the input images
        threshold=0.65,  # The inlier-outlier threshold
        conf = 0.999,
        spatial_coherence_weight=0.5,  # The weight for the spatial coherence term. It seems this is important for ACs.
        # The index of the sampler to be used. 0 - uniform, 1 - PROSAC, 2 - P-NAPSAC, 3 - NG-RANSAC's sampler, 4 - AR-Sampler
        max_iters=50000,  # The maximum number of iterations
        min_iters=500,  # The minimum number of iterations
         # The inlier probabilities for all points
        use_sprt= False,
        min_inlier_ratio_for_sprt= 0.01,
        sampler = 1,
        neighborhood= 1,
        neighborhood_size= 20.0,
        lo_number =1000,
        sampler_variance = 0.1,
        # The neighborhood size. For grid-based neighborhood, it is the division number along each axis, 8 works well. For FLANN, it is the radius of the hypersphere used.
        solver=Solver.AffineBased.value)  # The id of the used solver. 0 - point-based, 1 - SIFT-based, 2 - AC-based
    # print(deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    return E, mask

def estimate_pose_affine(ACs,kpts1,kpts2,tentatives,w1,h1,w2,h2,K1,K2,sampler_id):
    if len(kpts1) < 5:
        return None
    K1inv = np.linalg.inv(K1[:2, :2])
    K2inv = np.linalg.inv(K2[:2, :2])

    kpts1 = (K1inv @ (kpts1 - K1[None, :2, 2]).T).T
    kpts2 = (K2inv @ (kpts2 - K2[None, :2, 2]).T).T
    #
    E, mask = verify_affine_pygcransac_essential_matrix(ACs,
                                                        # The input affine correspondences
                                                        tentatives,
                                                        # The matches containing the indices of the matched keypoints
                                                        h1,
                                                        # The height of the source image
                                                        w1,
                                                        # The width of the source image
                                                        h2,
                                                        # The height of the destination image
                                                        w2,
                                                        # The width of the destination image
                                                        K1[:3,:3],
                                                        K2[:3,:3],
                                                        Sampler.ARSampler.value)  # The id of the used sampler. 0 - uniform, 1 - PROSAC, 3 - NG-RANSAC's sampler, 4 - AR-Sampler
    ret = None
    mask_ = np.array([[np.uint8(i)] for i in mask])
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(_E, kpts1, kpts2, np.eye(3), 1e9, mask=mask_)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t, mask_.ravel() > 0)

    return ret


def rotate_intrinsic(K, n):
    base_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rot = np.linalg.matrix_power(base_rot, n)
    return rot @ K


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array(
            [
                [np.cos(r), -np.sin(r), 0.0, 0.0],
                [np.sin(r), np.cos(r), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1.0 / scales[0], 1.0 / scales[1], 1.0])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


# From Patch2Pix https://github.com/GrumpyZhou/patch2pix
def get_depth_tuple_transform_ops(resize=None, normalize=True, unscale=False):
    ops = []
    if resize:
        ops.append(TupleResize(resize, mode=InterpolationMode.BILINEAR))
    return TupleCompose(ops)


def get_tuple_transform_ops(resize=None, normalize=True, unscale=False):
    ops = []
    if resize:
        ops.append(TupleResize(resize))
    if normalize:
        ops.append(TupleToTensorScaled())
        ops.append(
            TupleNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )  # Imagenet mean/std
    else:
        if unscale:
            ops.append(TupleToTensorUnscaled())
        else:
            ops.append(TupleToTensorScaled())
    return TupleCompose(ops)


class ToTensorScaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor, scale the range to [0, 1]"""

    def __call__(self, im):
        if not isinstance(im, torch.Tensor):
            im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
            im /= 255.0
            return torch.from_numpy(im)
        else:
            return im

    def __repr__(self):
        return "ToTensorScaled(./255)"


class TupleToTensorScaled(object):
    def __init__(self):
        self.to_tensor = ToTensorScaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorScaled(./255)"


class ToTensorUnscaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __call__(self, im):
        return torch.from_numpy(np.array(im, dtype=np.float32).transpose((2, 0, 1)))

    def __repr__(self):
        return "ToTensorUnscaled()"


class TupleToTensorUnscaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __init__(self):
        self.to_tensor = ToTensorUnscaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorUnscaled()"


class TupleResize(object):
    def __init__(self, size, mode=InterpolationMode.BICUBIC):
        self.size = size
        self.resize = transforms.Resize(size, mode)

    def __call__(self, im_tuple):
        return [self.resize(im) for im in im_tuple]

    def __repr__(self):
        return "TupleResize(size={})".format(self.size)


class TupleNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, im_tuple):
        return [self.normalize(im) for im in im_tuple]

    def __repr__(self):
        return "TupleNormalize(mean={}, std={})".format(self.mean, self.std)


class TupleCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im_tuple):
        for t in self.transforms:
            im_tuple = t(im_tuple)
        return im_tuple

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    # https://github.com/zju3dv/LoFTR/blob/94e98b695be18acb43d5d3250f52226a8e36f839/src/loftr/utils/geometry.py adapted from here
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>, should be normalized in (-1,1)
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    (
        n,
        h,
        w,
    ) = depth0.shape
    kpts0_depth = F.grid_sample(depth0[:, None], kpts0[:, :, None], mode="bilinear")[
        :, 0, :, 0
    ]
    kpts0 = torch.stack(
        (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    # Sample depth, get calculable_mask on depth != 0
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_n = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)
    kpts0_cam = kpts0_n

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    w_kpts0 = torch.stack(
        (2 * w_kpts0[..., 0] / w - 1, 2 * w_kpts0[..., 1] / h - 1), dim=-1
    )  # from [0.5,h-0.5] -> [-1+1/h, 1-1/h]
    # w_kpts0[~covisible_mask, :] = -5 # xd

    w_kpts0_depth = F.grid_sample(
        depth1[:, None], w_kpts0[:, :, None], mode="bilinear"
    )[:, 0, :, 0]
    consistent_mask = (
        (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
    ).abs() < 0.05
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    return valid_mask, w_kpts0


def numpy_to_pil(x: np.ndarray):
    """
    Args:
        x: Assumed to be of shape (h,w,c)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.max() <= 1.01:
        x *= 255
    x = x.astype(np.uint8)
    return Image.fromarray(x)


def tensor_to_pil(x, unnormalize=False):
    if unnormalize:
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        x = x * imagenet_std[:, None, None] + imagenet_mean[:, None, None]
    x = x.detach().permute(1, 2, 0).cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    return numpy_to_pil(x)


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def to_cpu(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cpu()
    return batch


def get_pose(calib):
    w, h = np.array(calib["imsize"])[0]
    return np.array(calib["K"]), np.array(calib["R"]), np.array(calib["T"]).T, h, w


def compute_relative_pose(R1, t1, R2, t2):
    rots = R2 @ (R1.T)
    trans = -rots @ t1 + t2
    return rots, trans
