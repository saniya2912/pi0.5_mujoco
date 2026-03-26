"""Run the fine-tuned π0.5 Baxter policy in MuJoCo.

Architecture
------------
  Policy server (serve_policy.py)  ←→  WebSocket  ←→  this script
          ↑                                                   ↓
  loads checkpoint 19999                         MuJoCo BaxterEnv
                                                 captures obs → sends to server
                                                 receives 10-step action chunk
                                                 executes each action for 50 physics steps
                                                 (matching 10 Hz training cadence)

Observation format sent to server (matches training exactly):
    observation/image       (224, 224, 3) uint8 HWC  — head camera
    observation/wrist_image (224, 224, 3) uint8 HWC  — right wrist camera
    observation/state       (7,)         float32     — right arm qpos[8:15]
    prompt                  str                      — language instruction

Action format received from server:
    actions  (10, 32) float32   — first 7 dims are right arm velocity commands
    Applied to d.ctrl[1:8] (RIGHT_CTRL) for 50 physics steps per action (= 0.1 s at dt=0.002).

Usage
-----
    # Terminal 1 — start the policy server
    cd openpi
    uv run scripts/serve_policy.py \\
        --policy.config pi05_baxter \\
        --policy.dir checkpoints/pi05_baxter/run_001/19999

    # Terminal 2 — run this script
    cd /home/robotlab/Desktop/saniya_ws/pi0.5_mujoco
    python run_inference.py --prompt "move the cube to the right"
"""

import collections
import csv
import dataclasses
import logging
import pathlib
import sys

import imageio
import mujoco
import mujoco.viewer
import numpy as np
import tyro

# ---------------------------------------------------------------------------
# openpi_client lives inside the openpi subpackage
# ---------------------------------------------------------------------------
_OPENPI_CLIENT = (
    pathlib.Path(__file__).parent
    / "openpi"
    / "packages"
    / "openpi-client"
    / "src"
)
if str(_OPENPI_CLIENT) not in sys.path:
    sys.path.insert(0, str(_OPENPI_CLIENT))

from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402
from baxter_env import BaxterEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Constants — must match record_demos.py exactly
# ---------------------------------------------------------------------------
DT            = 0.002   # physics timestep (seconds)
CTRL_HZ       = 100     # control frequency during data collection
DATA_HZ       = 10      # training data frequency (stride = CTRL_HZ / DATA_HZ)
SUBSTEPS      = 5       # physics steps per control step during data collection
# Each training action spans: (CTRL_HZ / DATA_HZ) * SUBSTEPS = 50 physics steps
PHYS_PER_ACTION = (CTRL_HZ // DATA_HZ) * SUBSTEPS   # = 50

# Joint slices — must match record_demos.py
RIGHT_QPOS = slice(8, 15)   # in d.qpos
RIGHT_CTRL = slice(1, 8)    # in d.ctrl


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    xml_path:     str          = "models/baxter.xml"
    host:         str          = "0.0.0.0"
    port:         int          = 8000
    prompt:       str          = "move the cube to the right"
    max_actions:  int          = 150      # max action chunks to execute (= 15 s at 10 Hz)
    replan_steps: int          = 5        # execute this many actions before replanning
    save_video:   bool         = False
    video_dir:    pathlib.Path = pathlib.Path("data/inference/videos")
    log_file:     pathlib.Path = pathlib.Path("data/inference/diagnostics.csv")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: Args) -> None:
    logging.info("Loading BaxterEnv from %s", args.xml_path)
    env = BaxterEnv(args.xml_path)

    logging.info("Connecting to policy server at %s:%d", args.host, args.port)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("Connected — server metadata: %s", client.get_server_metadata())

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    if args.save_video:
        args.video_dir.mkdir(parents=True, exist_ok=True)

    obs = env.reset()
    action_plan: collections.deque[np.ndarray] = collections.deque()
    video_frames: list[np.ndarray] = []
    total_actions = 0

    logging.info("Prompt: '%s'", args.prompt)
    logging.info("Each action = %d physics steps = %.3f s", PHYS_PER_ACTION, PHYS_PER_ACTION * DT)

    csv_file = open(args.log_file, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        ["action_idx"]
        + [f"action_{i}" for i in range(7)]
        + [f"qpos_right_{i}" for i in range(7)]
    )

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while total_actions < args.max_actions and viewer.is_running():

            # ----------------------------------------------------------------
            # Replan: build a new action chunk from the server
            # ----------------------------------------------------------------
            if not action_plan:
                # Images: baxter_env returns CHW uint8 → server expects HWC uint8
                img_hwc   = np.transpose(obs["image"],       (1, 2, 0))
                wrist_hwc = np.transpose(obs["wrist_image"], (1, 2, 0))

                # State: right arm qpos (7,) — obs["state"] = qpos[7:22], right arm = [1:8]
                state_7d = obs["state"][1:8].astype(np.float32)

                element = {
                    "observation/image":       img_hwc,
                    "observation/wrist_image": wrist_hwc,
                    "observation/state":       state_7d,
                    "prompt":                  args.prompt,
                }

                logging.info("Action %d/%d — requesting chunk (state: %s)",
                             total_actions, args.max_actions,
                             np.array2string(state_7d, precision=3, suppress_small=True))

                response = client.infer(element)
                chunk = np.array(response["actions"])  # (10, 32) or (10, 7)

                logging.info("  Received chunk shape %s  |  first action: %s",
                             chunk.shape,
                             np.array2string(chunk[0, :7], precision=4, suppress_small=True))

                # Queue only replan_steps actions from the chunk
                for i in range(min(args.replan_steps, len(chunk))):
                    action_plan.append(chunk[i, :7].astype(np.float64))

            # ----------------------------------------------------------------
            # Execute one action for PHYS_PER_ACTION physics steps
            # ----------------------------------------------------------------
            action_7d = action_plan.popleft()

            # Clip to actuator ctrlrange for right arm
            ctrl_min = env.model.actuator_ctrlrange[RIGHT_CTRL, 0]
            ctrl_max = env.model.actuator_ctrlrange[RIGHT_CTRL, 1]
            action_7d = np.clip(action_7d, ctrl_min, ctrl_max)

            for _ in range(PHYS_PER_ACTION):
                env.data.ctrl[RIGHT_CTRL] = action_7d
                mujoco.mj_step(env.model, env.data)
                viewer.sync()

            # Refresh obs after executing the action
            obs = env.get_obs()

            writer.writerow(
                [total_actions]
                + action_7d.tolist()
                + obs["state"][1:8].tolist()
            )

            if args.save_video:
                video_frames.append(env.render())

            total_actions += 1

        if not viewer.is_running():
            logging.info("Viewer closed at action %d", total_actions)

    csv_file.close()
    logging.info("Diagnostics saved to %s", args.log_file)

    if args.save_video and video_frames:
        existing = sorted(args.video_dir.glob("rollout_[0-9]*.mp4"))
        next_idx = (int(existing[-1].stem.split("_")[1]) + 1) if existing else 0
        video_path = args.video_dir / f"rollout_{next_idx:04d}.mp4"
        imageio.mimwrite(str(video_path), video_frames, fps=10)
        logging.info("Video saved to %s", video_path)

    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    run(tyro.cli(Args))
