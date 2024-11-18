import numpy as np
import matplotlib.pyplot as plt
import cv2
import pygcransac
from copy import deepcopy
import kornia as K
import kornia.feature as KF
import torch
from kornia_moons.feature import *
from time import time
from Network.gcransac_parameter_types import *
import torch.nn as nn
import warnings
from Network.laf import *

from Network.check import *

concatenate = torch.cat

# Function to load the image into a pytorch tensor
def load_torch_image(fname,w1,h1):
    img = K.image_to_tensor( cv2.resize(cv2.imread(fname), (int(w1), int(h1))), False).float() / 255.
    img = K.color.bgr_to_rgb(img)
    return img
#
# # Deciding about the device used. Prefer CUDA if available.
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
#

def map_location_to_cpu(storage, *args, **kwargs) :
    """Map location of device to CPU, util for loading things from HUB."""
    return storage



class LAFAffNetShapeEstimator(nn.Module):


    def __init__(self, pretrained: bool = False, preserve_orientation: bool = True, weight_path: str ="/home/xxx/project/python/DenseAffine/weights/outdoor/Aff_res_shape.pth" ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 3, kernel_size=8, stride=1, padding=0, bias=True),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.patch_size = 32
        if pretrained:
            hncheckpoint = torch.load(weight_path)
            self.load_state_dict(hncheckpoint['state_dict'])
            self.preserve_orientation = preserve_orientation
            self.eval()

    @staticmethod
    def _normalize_input(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Utility function that normalizes the input by batch."""
        sp, mp = torch.std_mean(x, dim=(-3, -2, -1), keepdim=True)
        # WARNING: we need to .detach() input, otherwise the gradients produced by
        # the patches extractor with F.grid_sample are very noisy, making the detector
        # training totally unstable.
        return (x - mp.detach()) / (sp.detach() + eps)

    def forward(self, laf: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            LAF: :math:`(B, N, 2, 3)`
            img: :math:`(B, 1, H, W)`

        Returns:
            LAF_out: :math:`(B, N, 2, 3)`
        """
        KORNIA_CHECK_LAF(laf)
        KORNIA_CHECK_SHAPE(img, ["B", "1", "H", "W"])
        B, N = laf.shape[:2]
        PS: int = self.patch_size
        patches: torch.Tensor = extract_patches_from_pyramid(img, make_upright(laf), PS, True).view(-1, 1, PS, PS)
        xy = self.features(self._normalize_input(patches)).view(-1, 3)
        a1 = torch.cat([1.0 + xy[:, 0].reshape(-1, 1, 1), 0 * xy[:, 0].reshape(-1, 1, 1)], dim=2)
        a2 = torch.cat([xy[:, 1].reshape(-1, 1, 1), 1.0 + xy[:, 2].reshape(-1, 1, 1)], dim=2)
        new_laf_no_center = torch.cat([a1, a2], dim=1).reshape(B, N, 2, 2)
        new_laf = torch.cat([new_laf_no_center, laf[:, :, :, 2:3]], dim=3)
        scale_orig = get_laf_scale(laf)
        if self.preserve_orientation:
            ori_orig = get_laf_orientation(laf)
        ellipse_scale = get_laf_scale(new_laf)
        laf_out = scale_laf(make_upright(new_laf), scale_orig / ellipse_scale)
        if self.preserve_orientation:
            laf_out = set_laf_orientation(laf_out, ori_orig)
        return laf_out


def is_mps_tensor_safe(x: Tensor) -> bool:
    """Return whether tensor is on MPS device."""
    return 'mps' in str(x.device)

class HardNet(nn.Module):
    r"""Module, which computes HardNet descriptors of given grayscale patches of 32x32.

    This is based on the original code from paper "Working hard to know your neighbor's
    margins: Local descriptor learning loss". See :cite:`HardNet2017` for more details.

    Args:
        pretrained: Download and set pretrained weights to the model.

    Returns:
        torch.Tensor: HardNet descriptor of the patches.

    Shape:
        - Input: :math:`(B, 1, 32, 32)`
        - Output: :math:`(B, 128)`

    Examples:
        >>> input = torch.rand(16, 1, 32, 32)
        >>> hardnet = HardNet()
        >>> descs = hardnet(input) # 16x128
    """
    patch_size = 32

    def __init__(self, pretrained ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

        # use torch.hub to load pretrained model
        if pretrained:
            hncheckpoint = torch.load("weights/HardNet++.pth")
            self.load_state_dict(hncheckpoint['state_dict'])
        self.eval()

    @staticmethod
    def _normalize_input(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Utility function that normalizes the input by batch."""
        if not is_mps_tensor_safe(x):
            sp, mp = torch.std_mean(x, dim=(-3, -2, -1), keepdim=True)
        else:
            mp = torch.mean(x, dim=(-3, -2, -1), keepdim=True)
            sp = torch.std(x, dim=(-3, -2, -1), keepdim=True)
        # WARNING: we need to .detach() input, otherwise the gradients produced by
        # the patches extractor with F.grid_sample are very noisy, making the detector
        # training totally unstable.
        return (x - mp.detach()) / (sp.detach() + eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        KORNIA_CHECK_SHAPE(input, ["B", "1", "32", "32"])
        x_norm: torch.Tensor = self._normalize_input(input)
        x_features: torch.Tensor = self.features(x_norm)
        x_out = x_features.view(x_features.size(0), -1)
        return F.normalize(x_out, dim=1)


class LAFDescriptor(nn.Module):
    r"""Module to get local descriptors, corresponding to LAFs (keypoints).

    Internally uses :func:`~kornia.feature.get_laf_descriptors`.

    Args:
        patch_descriptor_module: patch descriptor module, e.g. :class:`~kornia.feature.SIFTDescriptor`
            or :class:`~kornia.feature.HardNet`. Default: :class:`~kornia.feature.HardNet`.
        patch_size: patch size in pixels, which descriptor expects.
        grayscale_descriptor: ``True`` if patch_descriptor expects single-channel image.
    """

    def __init__(
        self, patch_descriptor_module = None, patch_size: int = 32, grayscale_descriptor = True
    ) -> None:
        super().__init__()
        if patch_descriptor_module is None:
            patch_descriptor_module = HardNet(True)
        self.descriptor = patch_descriptor_module
        self.patch_size = patch_size
        self.grayscale_descriptor = grayscale_descriptor

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(descriptor={self.descriptor.__repr__()}, "
            f"patch_size={self.patch_size}, "
            f"grayscale_descriptor='{self.grayscale_descriptor})"
        )

    def forward(self, img, lafs) :
        r"""Three stage local feature detection.

        First the location and scale of interest points are determined by
        detect function. Then affine shape and orientation.

        Args:
            img: image features with shape :math:`(B,C,H,W)`.
            lafs: local affine frames :math:`(B,N,2,3)`.

        Returns:
            Local descriptors of shape :math:`(B,N,D)` where :math:`D` is descriptor size.
        """
        return get_laf_descriptors(img, lafs, self.descriptor, self.patch_size, self.grayscale_descriptor)



def rgb_to_grayscale(image: Tensor, rgb_weights: Tensor | None = None) -> Tensor:
    r"""Convert a RGB image to grayscale version of image.

    .. image:: _static/img/rgb_to_grayscale.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB image to be converted to grayscale with shape :math:`(*,3,H,W)`.
        rgb_weights: Weights that will be applied on each channel (RGB).
            The sum of the weights should add up to one.
    Returns:
        grayscale version of the image with shape :math:`(*,1,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = rgb_to_grayscale(input) # 2x1x4x5
    """
    KORNIA_CHECK_IS_TENSOR(image)

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if rgb_weights is None:
        # 8 bit images
        if image.dtype == torch.uint8:
            rgb_weights = torch.tensor([76, 150, 29], device=image.device, dtype=torch.uint8)
        # floating point images
        elif image.dtype in (torch.float16, torch.float32, torch.float64):
            rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
        else:
            raise TypeError(f"Unknown data type: {image.dtype}")
    else:
        # is tensor that we make sure is in the same device/dtype
        rgb_weights = rgb_weights.to(image)

    # unpack the color image channels with RGB order
    r: Tensor = image[..., 0:1, :, :]
    g: Tensor = image[..., 1:2, :, :]
    b: Tensor = image[..., 2:3, :, :]

    w_r, w_g, w_b = rgb_weights.unbind()
    return w_r * r + w_g * g + w_b * b


def get_laf_descriptors(
    img , lafs, patch_descriptor, patch_size = 32, grayscale_descriptor = True
) -> Tensor:
    r"""Function to get local descriptors, corresponding to LAFs (keypoints).

    Args:
        img: image features with shape :math:`(B,C,H,W)`.
        lafs: local affine frames :math:`(B,N,2,3)`.
        patch_descriptor: patch descriptor module, e.g. :class:`~kornia.feature.SIFTDescriptor`
            or :class:`~kornia.feature.HardNet`.
        patch_size: patch size in pixels, which descriptor expects.
        grayscale_descriptor: True if ``patch_descriptor`` expects single-channel image.

    Returns:
        Local descriptors of shape :math:`(B,N,D)` where :math:`D` is descriptor size.
    """
    KORNIA_CHECK_LAF(lafs)
    patch_descriptor = patch_descriptor.to(img)
    patch_descriptor.eval()

    timg: Tensor = img
    if lafs.shape[1] == 0:
        warnings.warn(f"LAF contains no keypoints {lafs.shape}, returning empty tensor")
        return torch.empty(lafs.shape[0], lafs.shape[1], 128)
    if grayscale_descriptor and img.size(1) == 3:
        timg = rgb_to_grayscale(img)

    patches: Tensor = extract_patches_from_pyramid(timg, lafs, patch_size)
    # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
    # So we need to reshape a bit :)
    B, N, CH, H, W = patches.size()
    return patch_descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)

def get_laf_orientation(LAF) :
    """Return orientation of the LAFs, in degrees.

    Args:
        LAF: :math:`(B, N, 2, 3)`

    Returns:
        angle in degrees :math:`(B, N, 1)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = get_laf_orientation(input)  # BxNx1
    """
    KORNIA_CHECK_LAF(LAF)
    angle_rad = torch.atan2(LAF[..., 0, 1], LAF[..., 0, 0])
    return rad2deg(angle_rad).unsqueeze(-1)




def set_laf_orientation(LAF, angles_degrees) :
    """Change the orientation of the LAFs.

    Args:
        LAF: :math:`(B, N, 2, 3)`
        angles: :math:`(B, N, 1)` in degrees.

    Returns:
        LAF oriented with angles :math:`(B, N, 2, 3)`
    """
    KORNIA_CHECK_LAF(LAF)
    B, N = LAF.shape[:2]
    ori = get_laf_orientation(LAF).reshape_as(angles_degrees)
    return rotate_laf(LAF, angles_degrees - ori)
def laf_from_center_scale_ori(xy, scale = None, ori = None) :
    """Creates a LAF from keypoint center, scale and orientation.

    Useful to create kornia LAFs from OpenCV keypoints.

    Args:
        xy: :math:`(B, N, 2)`.
        scale: :math:`(B, N, 1, 1)`. If not provided, scale = 1.0 is assumed
        angle in degrees: :math:`(B, N, 1)`. If not provided orientation = 0 is assumed

    Returns:
        LAF :math:`(B, N, 2, 3)`
    """
    KORNIA_CHECK_SHAPE(xy, ["B", "N", "2"])
    device = xy.device
    dtype = xy.dtype
    B, N = xy.shape[:2]
    if scale is None:
        scale = torch.ones(B, N, 1, 1, device=device, dtype=dtype)
    if ori is None:
        ori = zeros(B, N, 1, device=device, dtype=dtype)
    KORNIA_CHECK_SHAPE(scale, ["B", "N", "1", "1"])
    KORNIA_CHECK_SHAPE(ori, ["B", "N", "1"])
    unscaled_laf = concatenate([angle_to_rotation_matrix(ori.squeeze(-1)), xy.unsqueeze(-1)], dim=-1)
    laf = scale_laf(unscaled_laf, scale)
    return laf


def laf_from_ours(kpts,scale ,angle,
                         mrSize=6.0,
                         device=torch.device('cpu'),
                         with_resp= False):
    N = len(kpts)
    xy = torch.tensor([(x, y) for x,y in kpts ], device=device, dtype=torch.float).view(1, N, 2)
    scales = torch.tensor(mrSize * scale, device=device, dtype=torch.float).view(1, N, 1, 1)
    angles = torch.tensor(angle , device=device, dtype=torch.float).view(1, N, 1)
    laf = laf_from_center_scale_ori(xy, scales, angles).reshape(1, -1, 2, 3)
    if not with_resp:
        return laf.reshape(1, -1, 2, 3)
    resp = torch.tensor([0.5]*N, device=device, dtype=torch.float).view(1, N, 1)
    return laf, resp


def scale_laf(laf, scale_coef) :
    """Multiplies region part of LAF ([:, :, :2, :2]) by a scale_coefficient.

    So the center, shape and orientation of the local feature stays the same, but the region area changes.

    Args:
        LAF :math:`(B, N, 2, 3)`
        scale_coef: broadcastable tensor or float.

    Returns:
        LAF :math:`(B, N, 2, 3)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> scale = 0.5
        >>> output = scale_laf(input, scale)  # BxNx2x3
    """
    # print(scale_coef.shape)
    mrSize = 6
    if  len(scale_coef.shape) ==1:
        N = scale_coef.shape[0]
        scale_coef = torch.tensor(mrSize * scale_coef, device=device, dtype=torch.float).view(1, N, 1, 1)

    else:
        N = scale_coef.shape[1]
        scale_coef = torch.tensor(mrSize * scale_coef, device=device, dtype=torch.float).view(1, N, 1, 1)
    if (type(scale_coef) is not float) and (type(scale_coef) is not Tensor):
        raise TypeError("scale_coef should be float or Tensor " "Got {}".format(type(scale_coef)))
    KORNIA_CHECK_LAF(laf)
    centerless_laf = laf[:, :, :2, :2]
    return concatenate([scale_coef * centerless_laf, laf[:, :, :, 2:]], dim=3)


# A function to convert the point ordering to probabilities used in NG-RANSAC's sampler or AR-Sampler.
def get_probabilities(tentatives):
    probabilities = []
    # Since the correspondences are assumed to be ordered by their SNN ratio a priori,
    # we just assign a probability according to their order.
    for i in range(len(tentatives)):
        probabilities.append(1.0 - i / len(tentatives))
    return probabilities


# A function to convert to local affine frames (LAFs) to their centroids to obtain simple keypoints
def get_coordinates(lafs1, lafs2):
    kps1 = [[lafs1[i, 0, 2], lafs1[i, 1, 2]] for i in range(lafs1.shape[0])]
    kps2 = [[lafs2[i, 0, 2], lafs2[i, 1, 2]] for i in range(lafs2.shape[0])]
    return kps1, kps2


# A function to convert pairs of LAFs to affine correspondences (ACs)
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


def verify_cv2_essential_matrix(kps1, kps2, tentatives, K1, K2):
    # Copy the coordinates in the source image selected by the tentative correspondences
    src_pts = np.float32([kps1[m.queryIdx] for m in tentatives]).reshape(-1, 2)
    # Copy the coordinates in the destination image selected by the tentative correspondences
    dst_pts = np.float32([kps2[m.trainIdx] for m in tentatives]).reshape(-1, 2)

    # Normalize the threshold
    threshold = 0.75
    avgDiagonal = (K1[0][0] + K1[1][1] + K2[0][0] + K2[1][1]) / 4;
    normalizedThreshold = threshold / avgDiagonal

    # Normalize the point coordinates
    normalizedSourcePoints = cv2.undistortPoints(np.expand_dims(src_pts, axis=1), cameraMatrix=K1, distCoeffs=None)
    normalizedDestinationPoints = cv2.undistortPoints(np.expand_dims(dst_pts, axis=1), cameraMatrix=K2, distCoeffs=None)

    # Estimate the essential matrix from the normalized coordinates
    # using the normalized threshold.
    E, mask = cv2.findEssentialMat(normalizedSourcePoints,
                                   normalizedDestinationPoints,
                                   focal=1.0,
                                   pp=(0., 0.),
                                   method=cv2.USAC_ACCURATE,
                                   prob=0.99,
                                   threshold=normalizedThreshold)

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
    E, mask = pygcransac.findEssentialMatrix(
        np.ascontiguousarray(ACs),  # The input affine correspondences
        K1,  # Intrinsic camera parameters of the source image
        K2,  # Intrinsic camera parameters of the destination image
        h1, w1, h2, w2,  # The sizes of the input images
        threshold=0.75,  # The inlier-outlier threshold
        sampler=sampler_id,
        # The index of the sampler to be used. 0 - uniform, 1 - PROSAC, 2 - P-NAPSAC, 3 - NG-RANSAC's sampler, 4 - AR-Sampler
        max_iters=5000,  # The maximum number of iterations
        min_iters=50,  # The minimum number of iterations
        probabilities=inlier_probabilities,  # The inlier probabilities for all points
        spatial_coherence_weight=0.4,  # The weight for the spatial coherence term. It seems this is important for ACs.
        neighborhood=1,
        # Neighborhood type. 0 - grid-based (faster/less accurate), 1 - FLANN-based (slower/more accurate).
        neighborhood_size=20,
        # The neighborhood size. For grid-based neighborhood, it is the division number along each axis, 8 works well. For FLANN, it is the radius of the hypersphere used.
        solver=Solver.AffineBased.value)  # The id of the used solver. 0 - point-based, 1 - SIFT-based, 2 - AC-based
    print(deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    return E, mask

