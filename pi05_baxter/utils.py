"""
utils.py - Shared preprocessing helpers.

Three responsibilities:
  1. Image preprocessing       — numpy uint8 → normalised float tensor
  2. Joint state normalisation — scale qpos/qvel to [-1, 1] using model limits
  3. Action scaling            — map policy output back to actuator control range
"""

import numpy as np
import torch


# ------------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------------

# ImageNet-style channel statistics (used by most vision backbones)
_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(rgb: np.ndarray) -> torch.Tensor:
    """
    Convert a (H, W, 3) uint8 RGB image to a (3, H, W) float32 tensor
    normalised with ImageNet statistics.

    Args:
        rgb: Raw camera frame from BaxterEnv, dtype uint8, values in [0, 255].

    Returns:
        Tensor of shape (3, H, W), float32.
    """
    if rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {rgb.dtype}")

    img = rgb.astype(np.float32) / 255.0          # [0, 1]
    img = (img - _IMG_MEAN) / _IMG_STD            # channel-wise normalisation
    img = np.transpose(img, (2, 0, 1))            # HWC → CHW
    return torch.from_numpy(img)                  # (3, H, W)


# ------------------------------------------------------------------
# Joint state normalisation
# ------------------------------------------------------------------

def normalise_joints(
    qpos: np.ndarray,
    qvel: np.ndarray,
    pos_limits: tuple[np.ndarray, np.ndarray],
    vel_limit: float = 2.0,
) -> torch.Tensor:
    """
    Concatenate and normalise qpos and qvel into a single state vector in [-1, 1].

    Args:
        qpos:       Joint positions, shape (nq,).
        qvel:       Joint velocities, shape (nv,).
        pos_limits: (lower, upper) each shape (nq,) from model.jnt_range.
        vel_limit:  Symmetric velocity limit (rad/s) used for normalisation.

    Returns:
        Float32 tensor of shape (nq + nv,).
    """
    lower, upper = pos_limits
    mid   = 0.5 * (lower + upper)
    half  = 0.5 * (upper - lower)
    # Avoid divide-by-zero for fixed joints (half == 0)
    half  = np.where(half > 1e-6, half, 1.0)

    qpos_norm = (qpos - mid) / half                        # ∈ [-1, 1]
    qvel_norm = np.clip(qvel / vel_limit, -1.0, 1.0)

    state = np.concatenate([qpos_norm, qvel_norm]).astype(np.float32)
    return torch.from_numpy(state)


# ------------------------------------------------------------------
# Action scaling
# ------------------------------------------------------------------

def scale_action(
    raw_action: np.ndarray,
    ctrl_range: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Map policy output in [-1, 1] to the actuator control range defined in the
    MuJoCo model and clip to hard limits.

    Args:
        raw_action:  Policy output, shape (nu,), assumed in [-1, 1].
        ctrl_range:  (lower, upper) from model.actuator_ctrlrange, each (nu,).

    Returns:
        Clipped float64 array of shape (nu,) ready for data.ctrl.
    """
    lower, upper = ctrl_range
    mid  = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    scaled = mid + raw_action * half
    return np.clip(scaled, lower, upper)


def extract_ctrl_range(model) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) actuator control ranges from a MjModel."""
    lower = model.actuator_ctrlrange[:, 0].copy()
    upper = model.actuator_ctrlrange[:, 1].copy()
    return lower, upper


def extract_pos_limits(model) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) joint position ranges from a MjModel."""
    lower = model.jnt_range[:, 0].copy()
    upper = model.jnt_range[:, 1].copy()
    return lower, upper
