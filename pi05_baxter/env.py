"""
env.py - MuJoCo Baxter simulation environment.

Loads the Baxter robot model, manages physics stepping, renders camera images,
and returns structured observations for policy consumption.
"""

import numpy as np
import mujoco
import mujoco.viewer
import cv2


class BaxterEnv:
    """
    Thin wrapper around a MuJoCo Baxter model.

    Observation dict returned by reset() and step():
        image : np.ndarray  (224, 224, 3) uint8  — RGB from the fixed head cam
        qpos  : np.ndarray  (nq,)  float64       — joint positions
        qvel  : np.ndarray  (nv,)  float64       — joint velocities
    """

    # Name of the camera defined in baxter.xml (change if your model differs)
    CAMERA_NAME = "head_camera"

    def __init__(
        self,
        model_path: str = "models/baxter.xml",
        img_size: tuple = (224, 224),
        show_viewer: bool = True,
    ):
        self.img_h, self.img_w = img_size

        # Load model and allocate data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Off-screen renderer for policy camera images
        self.renderer = mujoco.Renderer(self.model, height=self.img_h, width=self.img_w)

        # Resolve camera id (fall back to camera 0 if name not found)
        try:
            self._cam_id = self.model.camera(self.CAMERA_NAME).id
        except KeyError:
            print(f"[BaxterEnv] Camera '{self.CAMERA_NAME}' not found — using camera 0.")
            self._cam_id = 0

        # Optional passive viewer for real-time visualisation
        self._viewer = None
        if show_viewer:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Cache actuator count for action validation
        self.nu = self.model.nu

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset physics to keyframe 0 (or default pose) and return observation."""
        mujoco.mj_resetData(self.model, self.data)

        # If baxter.xml defines a keyframe named "home", use it
        try:
            keyframe_id = self.model.keyframe("home").id
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        except KeyError:
            pass

        mujoco.mj_forward(self.model, self.data)
        self._sync_viewer()
        return self._get_obs()

    def step(self, action: np.ndarray) -> dict:
        """
        Apply action to actuators, advance physics by one timestep, return observation.

        Args:
            action: (nu,) array of control signals (joint velocities from policy,
                    clipped externally before calling step).
        """
        if action.shape != (self.nu,):
            raise ValueError(f"Expected action shape ({self.nu},), got {action.shape}")

        np.copyto(self.data.ctrl, action)
        mujoco.mj_step(self.model, self.data)
        self._sync_viewer()
        return self._get_obs()

    def get_observation(self) -> dict:
        """Return current observation without stepping physics."""
        return self._get_obs()

    def close(self):
        """Release renderer and viewer resources."""
        self.renderer.close()
        if self._viewer is not None:
            self._viewer.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        """Render camera image and read joint state."""
        self.renderer.update_scene(self.data, camera=self._cam_id)
        # render() returns (H, W, 3) uint8 RGB
        rgb = self.renderer.render()

        # Resize only when model resolution differs from target (usually a no-op)
        if rgb.shape[:2] != (self.img_h, self.img_w):
            rgb = cv2.resize(rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)

        return {
            "image": rgb,                    # (224, 224, 3) uint8
            "qpos": self.data.qpos.copy(),   # (nq,) float64
            "qvel": self.data.qvel.copy(),   # (nv,) float64
        }

    def _sync_viewer(self):
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
