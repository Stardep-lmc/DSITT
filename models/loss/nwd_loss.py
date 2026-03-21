"""
Normalized Wasserstein Distance (NWD) for small object detection/tracking.

Reference: A Normalized Gaussian Wasserstein Distance for Tiny Object Detection (AAAI 2022)

Key insight: IoU is extremely sensitive to small positional shifts for tiny objects.
For a 4x4 pixel target, 1-pixel shift drops IoU from 1.0 to ~0.56.
NWD models bounding boxes as 2D Gaussian distributions, providing smoother gradients.
"""

import torch
import torch.nn as nn
import math


def bbox_to_gaussian(bbox: torch.Tensor):
    """
    Model bounding box (cx, cy, w, h) as a 2D Gaussian distribution.

    The box center becomes the mean μ, and the box dimensions
    determine the standard deviation σ (half of width/height).

    Args:
        bbox: [..., 4] tensor with (cx, cy, w, h) format

    Returns:
        mu: [..., 2] mean (cx, cy)
        sigma: [..., 2] standard deviation (w/2, h/2)
    """
    mu = bbox[..., :2]                    # (cx, cy)
    sigma = bbox[..., 2:].clamp(min=1e-6) / 2.0  # (w/2, h/2)
    return mu, sigma


def wasserstein_distance_2d(mu_p, sigma_p, mu_g, sigma_g):
    """
    Compute squared 2nd-order Wasserstein distance between two diagonal Gaussians.

    For diagonal covariance matrices, the closed-form is:
        W₂² = ||μ_p - μ_g||² + ||σ_p - σ_g||²_F

    where σ is the square root of diagonal covariance (i.e., standard deviation).

    Args:
        mu_p, mu_g: [..., 2] means
        sigma_p, sigma_g: [..., 2] standard deviations

    Returns:
        w2_squared: [...] squared Wasserstein distance
    """
    # Mean distance
    mean_dist = ((mu_p - mu_g) ** 2).sum(dim=-1)

    # Standard deviation distance (Frobenius norm for diagonal matrices)
    sigma_dist = ((sigma_p - sigma_g) ** 2).sum(dim=-1)

    return mean_dist + sigma_dist


def nwd_pairwise(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor,
                 constant: float = 0.1) -> torch.Tensor:
    """
    Compute pairwise NWD between predicted and ground truth boxes.

    NWD = exp(-W₂ / C)

    where W₂ = sqrt(W₂²) is the 2nd-order Wasserstein distance,
    and C is a normalization constant.

    Note: We use W₂ (not W₂²) following the original NWD paper.
    Default C=0.1 is tuned for normalized [0,1] box coordinates.
    (Use C≈12 for pixel coordinates on ~800px images.)

    Args:
        pred_boxes: [N, 4] predicted boxes (cx, cy, w, h), normalized [0,1]
        gt_boxes: [M, 4] ground truth boxes (cx, cy, w, h), normalized [0,1]
        constant: normalization constant C

    Returns:
        nwd_matrix: [N, M] NWD scores (higher = more similar, range (0, 1])
    """
    mu_p, sigma_p = bbox_to_gaussian(pred_boxes)   # [N, 2], [N, 2]
    mu_g, sigma_g = bbox_to_gaussian(gt_boxes)     # [M, 2], [M, 2]

    # Expand for pairwise computation
    mu_p = mu_p[:, None, :]      # [N, 1, 2]
    sigma_p = sigma_p[:, None, :]
    mu_g = mu_g[None, :, :]      # [1, M, 2]
    sigma_g = sigma_g[None, :, :]

    w2_sq = wasserstein_distance_2d(mu_p, sigma_p, mu_g, sigma_g)  # [N, M]
    w2 = (w2_sq + 1e-8).sqrt()  # W₂ distance (not squared)

    nwd = torch.exp(-w2 / constant)
    return nwd


def nwd_paired(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor,
               constant: float = 0.1) -> torch.Tensor:
    """
    Compute NWD between paired (1:1) predicted and ground truth boxes.

    Args:
        pred_boxes: [N, 4] predicted boxes (cx, cy, w, h), normalized [0,1]
        gt_boxes: [N, 4] ground truth boxes (cx, cy, w, h), normalized [0,1]
        constant: normalization constant C

    Returns:
        nwd_values: [N] NWD scores
    """
    mu_p, sigma_p = bbox_to_gaussian(pred_boxes)
    mu_g, sigma_g = bbox_to_gaussian(gt_boxes)

    w2_sq = wasserstein_distance_2d(mu_p, sigma_p, mu_g, sigma_g)
    w2 = (w2_sq + 1e-8).sqrt()  # W₂ distance (not squared)

    nwd = torch.exp(-w2 / constant)
    return nwd


def nwd_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor,
             constant: float = 0.1) -> torch.Tensor:
    """
    NWD-based regression loss for paired boxes.

    L_nwd = 1 - NWD(pred, gt)

    Args:
        pred_boxes: [N, 4] predicted boxes (cx, cy, w, h)
        gt_boxes: [N, 4] ground truth boxes (cx, cy, w, h)
        constant: normalization constant C

    Returns:
        loss: scalar, mean NWD loss
    """
    nwd_values = nwd_paired(pred_boxes, gt_boxes, constant)
    return (1.0 - nwd_values).mean()


def nwd_matching_cost(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor,
                      constant: float = 0.1) -> torch.Tensor:
    """
    NWD-based matching cost for Hungarian assignment.

    Cost = 1 - NWD(pred, gt)  (lower = better match)

    This replaces the GIoU matching cost in TALA.

    Args:
        pred_boxes: [N, 4] predicted boxes (cx, cy, w, h)
        gt_boxes: [M, 4] ground truth boxes (cx, cy, w, h)
        constant: normalization constant C

    Returns:
        cost_matrix: [N, M] matching cost (lower = better)
    """
    nwd_matrix = nwd_pairwise(pred_boxes, gt_boxes, constant)
    return 1.0 - nwd_matrix