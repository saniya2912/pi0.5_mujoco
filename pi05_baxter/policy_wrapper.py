"""
policy_wrapper.py - Thin wrapper around the π0.5 (OpenPI) policy.

Handles:
  - Model loading from a checkpoint
  - Observation formatting (image tensor + state tensor + text instruction)
  - Inference call and raw action extraction

The openpi package is expected to be installed in the active Python environment.
See README.md for installation instructions.
"""

import numpy as np
import torch

from pi05_baxter.utils import preprocess_image, normalise_joints


# Default language instruction sent to the policy every step.
DEFAULT_INSTRUCTION = "reach the object"


class Pi05Policy:
    """
    Wraps an OpenPI π0.5 checkpoint for single-step action inference.

    Usage:
        policy = Pi05Policy(checkpoint_path="checkpoints/pi05_baxter")
        action = policy.act(obs_dict)   # obs_dict from BaxterEnv
    """

    def __init__(
        self,
        checkpoint_path: str,
        instruction: str = DEFAULT_INSTRUCTION,
        device: str | None = None,
    ):
        """
        Args:
            checkpoint_path: Path (or HuggingFace repo id) of the π0.5 checkpoint.
            instruction:     Default language command used at every inference step.
            device:          'cuda', 'cpu', or None (auto-select GPU if available).
        """
        self.instruction = instruction
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Pi05Policy] Using device: {self.device}")

        self._model = self._load_model(checkpoint_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, obs: dict) -> np.ndarray:
        """
        Run one inference step and return a joint-velocity action.

        Args:
            obs: Dict with keys:
                    "image" — (224, 224, 3) uint8 numpy array
                    "qpos"  — (nq,) float64 numpy array
                    "qvel"  — (nv,) float64 numpy array
                 Optionally "instruction" (str) to override the default.

        Returns:
            action: (nu,) float32 numpy array in [-1, 1].
                    Caller should scale via utils.scale_action before applying.
        """
        policy_input = self._format_obs(obs)
        with torch.no_grad():
            raw = self._model(policy_input)

        action = self._extract_action(raw)
        return action

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, checkpoint_path: str):
        """
        Load the π0.5 model from a checkpoint using the openpi client/loader API.

        The exact call depends on the openpi version installed.  Two common
        patterns are shown; uncomment the one that matches your install.
        """
        try:
            # Pattern A — openpi >= 0.1 with a high-level loader
            import openpi.models as pi_models          # noqa: F401
            model = pi_models.load(checkpoint_path)
            model = model.to(self.device)
            model.eval()
            return model

        except (ImportError, AttributeError):
            # Pattern B — bare checkpoint via torch.load (fallback)
            import importlib
            try:
                openpi = importlib.import_module("openpi")
                model = openpi.load_policy(checkpoint_path, device=self.device)
                return model
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load π0.5 checkpoint from '{checkpoint_path}'.\n"
                    "Make sure openpi is installed: pip install openpi\n"
                    f"Original error: {exc}"
                ) from exc

    def _format_obs(self, obs: dict) -> dict:
        """
        Convert BaxterEnv observation dict into the tensor dict expected by π0.5.

        π0.5 observation format:
            {
                "image"       : Tensor (1, 3, 224, 224)  — batch-dim prepended
                "state"       : Tensor (1, state_dim)
                "instruction" : str
            }
        """
        # --- image ---
        image_tensor = preprocess_image(obs["image"])   # (3, 224, 224)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        # --- state: flat concatenation of qpos + qvel (no normalisation here;
        #     the policy's own normalisation layers handle it internally) ---
        state_np = np.concatenate([obs["qpos"], obs["qvel"]]).astype(np.float32)
        state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device)  # (1, D)

        instruction = obs.get("instruction", self.instruction)

        return {
            "image":       image_tensor,
            "state":       state_tensor,
            "instruction": instruction,
        }

    @staticmethod
    def _extract_action(raw) -> np.ndarray:
        """
        Pull the action array out of whatever the policy returns.

        Handles:
          - dict with key "action"
          - plain Tensor
          - plain numpy array
        """
        if isinstance(raw, dict):
            action = raw["action"]
        else:
            action = raw

        if isinstance(action, torch.Tensor):
            action = action.squeeze(0).cpu().numpy()   # remove batch dim

        return action.astype(np.float32)
