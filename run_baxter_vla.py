"""Run a Pi0.5 VLA policy on the Baxter MuJoCo simulation.

Architecture
------------
MuJoCo Baxter simulator
        ↓
capture camera image + robot state  (BaxterEnv.get_obs)
        ↓
build observation dict with flat "/" keys expected by the server
        ↓
send to policy server over WebSocket  (WebsocketClientPolicy.infer)
        ↓
receive action chunk  (shape: [chunk_size, 15])
        ↓
execute actions one-at-a-time via a deque (action_plan)
        ↓
sync the passive MuJoCo viewer so the window updates
        ↓
repeat until max_steps or viewer is closed

Usage
-----
    python run_baxter_vla.py --prompt "move cube to the right"

    # with diagnostics logged to file and video saved:
    python run_baxter_vla.py --prompt "move cube to the right" --save-video --log-file data/baxter/run.log

The policy server must already be running at --host:--port before this
script is started.  WebsocketClientPolicy blocks until the server responds.
"""

import collections
import csv
import dataclasses
import logging
import mujoco.viewer
import pathlib
import sys

import imageio
import numpy as np
import tyro

# ---------------------------------------------------------------------------
# Locate the openpi_client package.
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
# CLI arguments
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    """Command-line arguments for the Baxter VLA evaluation loop."""

    xml_path: str = "models/baxter.xml"
    host: str = "0.0.0.0"
    port: int = 8000
    prompt: str = "pick up the cube"
    max_steps: int = 300
    replan_steps: int = 10

    # --- diagnostics ---------------------------------------------------------
    # Save a video of the rollout to data/baxter/videos/
    save_video: bool = False
    video_dir: pathlib.Path = pathlib.Path("data/baxter/videos")

    # Write per-step diagnostics (actions, joint positions) to a CSV file
    log_file: pathlib.Path = pathlib.Path("data/baxter/diagnostics.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_mismatch_summary() -> None:
    """Print the observation/action space mismatch to the logger once."""
    logging.warning("=" * 60)
    logging.warning("OBSERVATION / ACTION SPACE MISMATCH SUMMARY")
    logging.warning("=" * 60)
    logging.warning("Policy trained on : LIBERO (Franka robot)")
    logging.warning("Robot simulated   : Baxter")
    logging.warning("")
    logging.warning("  observation/state expected by policy : 8-D  "
                    "(eef_pos x3, eef_rot x3, gripper x2)")
    logging.warning("  observation/state sent from Baxter   : 15-D joint positions "
                    "→ TRUNCATED to first 8 (semantically wrong)")
    logging.warning("")
    logging.warning("  action output from policy : 7-D  "
                    "(Δeef_pos x3, Δeef_rot x3, gripper x1)")
    logging.warning("  action applied to Baxter  : padded to 15-D with zeros "
                    "(joints 8-14 receive no signal)")
    logging.warning("")
    logging.warning("  → Robot motion is NOT task-relevant. "
                    "Fine-tuning on Baxter data is required.")
    logging.warning("=" * 60)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run(args: Args) -> None:
    """Connect to the policy server and run a single Baxter episode."""

    logging.info("Loading Baxter environment from %s", args.xml_path)
    env = BaxterEnv(args.xml_path)

    logging.info("Connecting to policy server at ws://%s:%d …", args.host, args.port)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("Connected.  Server metadata: %s", client.get_server_metadata())

    # Print mismatch summary once so it appears in the log file
    _log_mismatch_summary()

    # Prepare output directories
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    if args.save_video:
        args.video_dir.mkdir(parents=True, exist_ok=True)

    obs = env.reset()
    action_plan: collections.deque[np.ndarray] = collections.deque()
    video_frames: list[np.ndarray] = []

    logging.info("Prompt : '%s'", args.prompt)
    logging.info("Starting rollout — max_steps=%d, replan_steps=%d",
                 args.max_steps, args.replan_steps)
    logging.info("Close the viewer window (or press Escape) to stop.")

    # Open CSV for per-step diagnostics
    csv_file = open(args.log_file, "w", newline="")
    writer = csv.writer(csv_file)
    # Header: step, raw policy action (7D), baxter qpos (15D), ctrl sent (15D)
    writer.writerow(
        ["step"]
        + [f"policy_action_{i}" for i in range(7)]
        + [f"qpos_{i}" for i in range(15)]
        + [f"ctrl_{i}" for i in range(15)]
    )

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        raw_action_7d = np.zeros(7)  # last raw policy action for logging

        for step in range(args.max_steps):

            if not viewer.is_running():
                logging.info("Viewer closed — stopping at step %d.", step)
                break

            # -------------------------------------------------------------------
            # Build observation dict.
            # NOTE: state is truncated to 8D (LIBERO format); Baxter has 15 joints.
            # -------------------------------------------------------------------
            element = {
                "observation/image":       obs["image"],
                "observation/wrist_image": obs["wrist_image"],
                "observation/state":       obs["state"][:8].astype(np.float32),
                "prompt":                  args.prompt,
            }

            # -------------------------------------------------------------------
            # Request new action chunk when plan is empty.
            # Policy returns 7D end-effector actions; we pad to 15D for Baxter.
            # -------------------------------------------------------------------
            if not action_plan:
                logging.info("Step %d: requesting new action chunk", step)
                response = client.infer(element)

                action_chunk: np.ndarray = response["actions"]  # (chunk_size, 7)
                raw_action_7d = action_chunk[0].copy()

                logging.info(
                    "  Raw policy action (7D): %s",
                    np.array2string(raw_action_7d, precision=4, suppress_small=True)
                )
                logging.info(
                    "  Action stats — mean: %.4f  std: %.4f  min: %.4f  max: %.4f",
                    action_chunk.mean(), action_chunk.std(),
                    action_chunk.min(), action_chunk.max()
                )

                # Pad 7D → 15D; joints 7-14 get zero velocity (no policy signal)
                baxter_chunk = np.zeros((len(action_chunk), 15), dtype=np.float32)
                baxter_chunk[:, :action_chunk.shape[1]] = action_chunk
                for i in range(min(args.replan_steps, len(baxter_chunk))):
                    action_plan.append(baxter_chunk[i])

            action = action_plan.popleft()

            obs, reward, done, info = env.step(action)

            # Log per-step data to CSV
            writer.writerow(
                [step]
                + raw_action_7d.tolist()
                + obs["state"].tolist()
                + action.tolist()
            )

            viewer.sync()

            if args.save_video:
                video_frames.append(env.render())

            if done:
                logging.info("Episode finished early at step %d.", step)
                break

    csv_file.close()
    logging.info("Diagnostics saved to %s", args.log_file)

    # Save video
    if args.save_video and video_frames:
        existing = sorted(args.video_dir.glob("rollout_[0-9]*.mp4"))
        next_idx = (int(existing[-1].stem.split("_")[1]) + 1) if existing else 0
        video_path = args.video_dir / f"rollout_{next_idx:04d}.mp4"
        logging.info("Saving video (%d frames) to %s", len(video_frames), video_path)
        imageio.mimwrite(str(video_path), video_frames, fps=20)

    logging.info("Done.")
    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Also write logs to file alongside the CSV
    log_path = pathlib.Path("data/baxter/run.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ]
    )
    args = tyro.cli(Args)
    run(args)
