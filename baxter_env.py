"""Minimal MuJoCo environment wrapper for the Baxter robot.

Provides a simple RL-style interface:
    env = BaxterEnv("models/baxter.xml")
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    image = env.render()
"""

import os
from typing import Any

import mujoco
import numpy as np


# Image dimensions expected by the VLA policy (SigLIP input resolution)
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

# Camera to use for policy observations (world-fixed scene camera)
CAMERA_NAME = "scene_camera"

# Baxter has 15 actuated joints: head_pan + 7 right arm + 7 left arm
NUM_JOINTS = 15


class BaxterEnv:
    """Thin MuJoCo wrapper around the Baxter MJCF model.

    Observations contain a camera image and joint position state.
    Actions are joint velocity commands written directly to data.ctrl.
    """

    def __init__(self, xml_path: str) -> None:
        """Load the Baxter MJCF model and initialise rendering.

        Args:
            xml_path: Path to baxter.xml (absolute or relative to cwd).
        """
        xml_path = os.path.abspath(xml_path)
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)

        # Renderer used to capture RGB frames for the policy
        self._renderer = mujoco.Renderer(self._model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

        self._step_count = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset simulation to the 'home' keyframe (or default pose if absent).

        Returns:
            Initial observation dict with keys "image" and "state".
        """
        mujoco.mj_resetData(self._model, self._data)

        # Try to apply the 'home' keyframe defined in baxter.xml
        keyframe_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if keyframe_id >= 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, keyframe_id)

        # Forward kinematics so rendering is consistent with the reset pose
        mujoco.mj_forward(self._model, self._data)

        self._step_count = 0
        return self.get_obs()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Apply action, advance the simulation, and return a transition tuple.

        Action is written to data.ctrl as a velocity command for all 15 actuators.
        The vector is clipped to each actuator's ctrlrange before application.

        Args:
            action: Float array of length NUM_JOINTS (15).

        Returns:
            (obs, reward, done, info) where reward and done are placeholders.
        """
        action = np.asarray(action, dtype=np.float64)

        # Clip to actuator control ranges defined in the XML
        ctrl_range = self._model.actuator_ctrlrange  # shape (nu, 2)
        action = np.clip(action, ctrl_range[:, 0], ctrl_range[:, 1])

        # Write commands and step the physics
        self._data.ctrl[:] = action
        mujoco.mj_step(self._model, self._data)

        self._step_count += 1

        obs = self.get_obs()
        reward = 0.0   # placeholder — task reward not implemented
        done = False   # placeholder — episode termination not implemented
        info: dict[str, Any] = {"step": self._step_count}

        return obs, reward, done, info

    def get_obs(self) -> dict:
        """Return the current observation.

        Returns:
            dict with:
                "image":       uint8 RGB array (C, H, W) = (3, 224, 224) — head camera
                "wrist_image": uint8 RGB array (C, H, W) = (3, 224, 224) — right hand camera
                "state":       float64 joint-position array (NUM_JOINTS,)
        """
        # Head (base) camera
        self._renderer.update_scene(self._data, camera=CAMERA_NAME)
        image_hwc = self._renderer.render()  # (H, W, 3) uint8
        image_chw = np.transpose(image_hwc, (2, 0, 1))  # (3, H, W)

        # Right wrist camera
        self._renderer.update_scene(self._data, camera="right_hand_camera")
        wrist_hwc = self._renderer.render()  # (H, W, 3) uint8
        wrist_chw = np.transpose(wrist_hwc, (2, 0, 1))  # (3, H, W)

        # Robot joint positions only (skip cube free-joint at qpos[0:7])
        # Order: head_pan(7) right_arm(8:15) left_arm(15:22)
        state = self._data.qpos[7:22].copy()  # (15,)

        return {"image": image_chw, "wrist_image": wrist_chw, "state": state}

    def render(self) -> np.ndarray:
        """Render the current frame for visualisation (HWC uint8).

        Returns:
            RGB image of shape (IMAGE_HEIGHT, IMAGE_WIDTH, 3).
        """
        self._renderer.update_scene(self._data, camera=CAMERA_NAME)
        return self._renderer.render()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def action_dim(self) -> int:
        return self._model.nu

    def close(self) -> None:
        """Release renderer resources."""
        self._renderer.close()
