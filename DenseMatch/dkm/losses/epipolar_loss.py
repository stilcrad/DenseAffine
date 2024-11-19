import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DenseMatch.dkm.utils.utils import warp_kpts
import cv2


class SampsonLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

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

    def expand_homo_ones(self, arr2d, axis=1):
        """Raise 2D array to homogenous coordinates
        Args:
            - arr2d: (N, 2) or (2, N)
            - axis: the axis to append the ones
        """
        if axis == 0:
            ones = torch.ones((1, arr2d.shape[1]), dtype=arr2d.dtype, device=arr2d.device)
        else:
            ones = torch.ones((arr2d.shape[0], 1), dtype=arr2d.dtype, device=arr2d.device)
        expanded_arr = torch.cat([arr2d, ones], dim=axis)
        return expanded_arr
        # if axis == 0:
        #     ones = np.ones((1, arr2d.shape[1]))
        # else:
        #     ones = np.ones((arr2d.shape[0], 1))
        # return np.concatenate([arr2d, ones], axis=axis)

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



    def forward(self,pts1, pts2, homos=True, eps=1e-8):
        """Calculate symmetric epipolar distance between 2 sets of points
        Args:
            - pts1, pts2: points correspondences in the two images,
              each has shape of (num_points, 2)
            - F: fundamental matrix that fulfills x2^T*F*x1=0,
              where x1 and x2 are the correspondence points in the 1st and 2nd image
        Return:
            A vector of (num_points,), containing root-squared epipolar distances

        """
        pt1 = np.array(pts1.detach().cpu())
        pt2 = np.array(pts2.detach().cpu())

        F, mask = cv2.findFundamentalMat(pt1, pt2, method=cv2.USAC_ACCURATE,ransacReprojThreshold=0.25, confidence=0.999,maxIters=120000)
        # Homogenous coordinates
        F = torch.tensor(F).cuda()
        if homos:
            pts1 = self.expand_homo_ones(pts1, axis=1)  # if pts1.shape[1] == 2 else pts1
            pts2 = self.expand_homo_ones(pts2, axis=1)  # if pts2.shape[1] == 2 else pts2

        # eg
        # l2=F*x1, l1=F^T*x2
        pts1 = pts1.type(torch.DoubleTensor).cuda()
        pts2 = pts2.type(torch.DoubleTensor).cuda()

        l2 = torch.matmul(F, pts1.t())  # 3,N
        l1 = torch.matmul(F.t(), pts2.t())
        dd = torch.sum(l2.t() * pts2, dim=1)  # Distance from pts2 to l2
        l1_sq_norm = l1[0, :] ** 2 + l1[1, :] ** 2
        l2_sq_norm = l2[0, :] ** 2 + l2[1, :] ** 2
        epipolar_sampson_loss = dd ** 2 / (eps + l1_sq_norm + l2_sq_norm)

        return epipolar_sampson_loss