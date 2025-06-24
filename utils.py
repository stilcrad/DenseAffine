

from __future__ import annotations

import torch
import torch.nn.functional as F



def rad2deg(tensor: Tensor) -> Tensor:
    r"""Function that converts angles from radians to degrees.

    Args:
        tensor: Tensor of arbitrary shape.

    Returns:
        Tensor with same shape as input.

    Example:
        >>> input = tensor(3.1415926535)
        >>> rad2deg(input)
        tensor(180.)
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(tensor)}")

    return 180.0 * tensor / pi.to(tensor.device).type(tensor.dtype)


def convert_points_from_homogeneous(points: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Function that converts points from homogeneous to Euclidean space.

    Args:
        points: the points to be transformed of shape :math:`(B, N, D)`.
        eps: to avoid division by zero.

    Returns:
        the points in Euclidean space :math:`(B, N, D-1)`.

    Examples:
        >>> input = tensor([[0., 0., 1.]])
        >>> convert_points_from_homogeneous(input)
        tensor([[0., 0.]])
    """
    if not isinstance(points, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(points)}")

    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    # we check for points at max_val
    z_vec: Tensor = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask: Tensor = torch.abs(z_vec) > eps
    scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

    return scale * points[..., :-1]


def angle_to_rotation_matrix(angle: Tensor) -> Tensor:
    r"""Create a rotation matrix out of angles in degrees.

    Args:
        angle: tensor of angles in degrees, any shape :math:`(*)`.

    Returns:
        tensor of rotation matrices with shape :math:`(*, 2, 2)`.

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = angle_to_rotation_matrix(input)  # Nx3x2x2
    """
    ang_rad = deg2rad(angle)
    cos_a: Tensor = torch.cos(ang_rad)
    sin_a: Tensor = torch.sin(ang_rad)
    return torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(*angle.shape, 2, 2)
