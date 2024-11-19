import cv2
import numpy as np
from einops.einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from DenseMatch.dkm.utils.utils import warp_kpts


class DepthRegressionLoss(nn.Module):
    def __init__(
        self,
        robust=True,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
    ):
        super().__init__()
        self.robust = robust  # measured in pixels
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale

    def geometric_dist(self, depth1, depth2, T_1to2, K1, K2, dense_matches, scale):
        """[summary]

        Args:
            H ([type]): [description]
            scale ([type]): [description]

        Returns:
            [type]: [description]
        """
        b, h1, w1, d = dense_matches.shape
        with torch.no_grad():
            x1_n = torch.meshgrid(
                *[
                    torch.linspace(
                        -1 + 1 / n, 1 - 1 / n, n, device=dense_matches.device
                    )
                    for n in (b, h1, w1)
                ]
            )
            x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(b, h1 * w1, 2)
            mask, x2 = warp_kpts(
                x1_n.double(),
                depth1.double(),
                depth2.double(),
                T_1to2.double(),
                K1.double(),
                K2.double(),
            )
            prob = mask.float().reshape(b, h1, w1)
        gd = (dense_matches - x2.reshape(b, h1, w1, 2)).norm(dim=-1)  # *scale?
        return gd, prob

    def dense_depth_loss(self, dense_certainty, prob, gd, scale, eps=1e-8):
        """[summary]

        Args:
            dense_certainty ([type]): [description]
            prob ([type]): [description]
            eps ([type], optional): [description]. Defaults to 1e-8.

        Returns:
            [type]: [description]
        """
        smooth_prob = prob
        ce_loss = F.binary_cross_entropy_with_logits(dense_certainty[:, 0], smooth_prob)
        depth_loss = gd[prob > 0]
        if not torch.any(prob > 0).item():
            depth_loss = (gd * 0.0).mean()  # Prevent issues where prob is 0 everywhere
        return {
            f"ce_loss_{scale}": ce_loss.mean(),
            f"depth_loss_{scale}": depth_loss.mean(),
        }

    def forward(self, dense_corresps, batch):
        """[summary]

        Args:
            out ([type]): [description]
            batch ([type]): [description]

        Returns:
            [type]: [description]
        """
        scales = list(dense_corresps.keys())
        tot_loss = 0.0
        prev_gd = 0.0
        for scale in scales:
            dense_scale_corresps = dense_corresps[scale]
            dense_scale_certainty, dense_scale_coords = (
                dense_scale_corresps["dense_certainty"],
                dense_scale_corresps["dense_flow"],
            )
            dense_scale_coords = rearrange(dense_scale_coords, "b d h w -> b h w d")
            b, h, w, d = dense_scale_coords.shape
            gd, prob = self.geometric_dist(
                batch["query_depth"],
                batch["support_depth"],
                batch["T_1to2"],
                batch["K1"],
                batch["K2"],
                dense_scale_coords,
                scale,
            )
            if (
                scale <= self.local_largest_scale and self.local_loss
            ):  # Thought here is that fine matching loss should not be punished by coarse mistakes, but should identify wrong matching
                prob = prob * (
                    F.interpolate(prev_gd[:, None], size=(h, w), mode="nearest")[:, 0]
                    < (2 / 512) * (self.local_dist * scale)
                )
            depth_losses = self.dense_depth_loss(dense_scale_certainty, prob, gd, scale)
            scale_loss = (
                self.ce_weight * depth_losses[f"ce_loss_{scale}"]
                + depth_losses[f"depth_loss_{scale}"]
            )  # scale ce loss for coarser scales
            if self.scale_normalize:
                scale_loss = scale_loss * 1 / scale
            tot_loss = tot_loss + scale_loss
            prev_gd = gd.detach()
        return tot_loss



def expand_homo_ones(arr2d, axis=1):
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

def sampson_distance(pts1, pts2, homos=True, eps=1e-8):
    """Calculate symmetric epipolar distance between 2 sets of points
    Args:
        - pts1, pts2: points correspondences in the two images,
          each has shape of (num_points, 2)
        - F: fundamental matrix that fulfills x2^T*F*x1=0,
          where x1 and x2 are the correspondence points in the 1st and 2nd image
    Return:
        A vector of (num_points,), containing root-squared epipolar distances

    """
    F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.USAC_ACCURATE, ransacReprojThreshold=0.999,
                                     confidence=0.999)
    # Homogenous coordinates
    if homos:
        pts1 = expand_homo_ones(pts1, axis=1)  # if pts1.shape[1] == 2 else pts1
        pts2 = expand_homo_ones(pts2, axis=1)  # if pts2.shape[1] == 2 else pts2

    # eg
    # l2=F*x1, l1=F^T*x2
    l2 = np.dot(F, pts1.T)  # 3,N
    l1 = np.dot(F.T, pts2.T)
    dd = np.sum(l2.T * pts2, 1)  # Distance from pts2 to l2
    epipolar_sampson_loss = dd ** 2 / (eps + l1[0, :] ** 2 + l1[1, :] ** 2 + l2[0, :] ** 2 + l2[1, :] ** 2)

    return epipolar_sampson_loss


def affine_sampson_distance(pts1, pts2, affine,  homos = False  ):
    if homos:
        pts1 = expand_homo_ones(pts1, axis=1)  # if pts1.shape[1] == 2 else pts1
        pts2 = expand_homo_ones(pts2, axis=1)  # if pts2.shape[1] == 2 else pts2
    F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.USAC_ACCURATE, ransacReprojThreshold=0.999,
                                     confidence=0.999)
    f11, f12, f13, f21, f22, f23, f31, f32, f33 = F.reshape(9)
    dist_list = []
    for i in range(pts1.shape[0]):
        pt1 = pts1[i]
        pt2 = pts2[i]
        x1,y1  = pt1
        x2,y2  = pt2
        a11 , a12 , a21 ,a22 = affine[i].reshape(4)
        f11 ,f12, f13, f21, f22 , f23 ,f31 ,f32 , f33 = F.reshape(9)
        M0= x1*(a11*f11 + a21*f21) +y1*(a11*f12+a21*f22) +a11*f13 +a21*f23 +f11*x2 +f21*y2 +f31
        M1=f13 + f11*x2  +f12 *y1
        M2=a11*f12 + a21*f22
        M3=f11
        M4=f23 +f21*x1 + f22*y1
        M5=a11*f11 +a21*f21
        M6=f21

        N0 = x1*(a12*f11 + a22*f21) + y1(a12*f12 + a22*f22) + a12*f13 + a22*f23 +f12*x2 + f22*y2 + f32
        N1 = f13 + f11*x1 + f12*y1
        N2 = a12*f11 + a22*f21
        N3 = f12
        N4 = f23 + f21*x1 + f22*y1
        N5 = a12*f12 + a22*f22
        N6 = f22

        dist1  =M0**2 /(M1**2 +M2**2 +M3**2+M4*2+M5**2+M6**2)
        dist2  =N0**2 /(N1**2 +N2**2 +N3**2+N4*2+N5**2+N6**2)

        dist = torch.mean(dist1+dist2)
        dist_list.append(dist)

    f_dist = np.mean(dist_list)

    return f_dist


def affine_sampson_distance_Only_affine(pts1, pts2, affine, homos=False):
    if homos:
        pts1 = expand_homo_ones(pts1, axis=1)  # if pts1.shape[1] == 2 else pts1
        pts2 = expand_homo_ones(pts2, axis=1)  # if pts2.shape[1] == 2 else pts2
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

        dist1 = M0 ** 2 / (M1 ** 2 +  M4 * 2 )
        dist2 = N0 ** 2 / (N1 ** 2 +  N4 * 2 )

        dist = torch.mean(dist1 + dist2)
        dist_list.append(dist)

    f_dist = np.mean(dist_list)

    return f_dist

