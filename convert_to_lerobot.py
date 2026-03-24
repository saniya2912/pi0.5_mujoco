"""Convert Baxter MuJoCo demo HDF5 files to LeRobot dataset format.

Our HDF5 structure (per episode_NNNN.hdf5):
  observations/image        (T, 3, 224, 224) uint8   scene camera (CHW)
  observations/wrist_image  (T, 3, 224, 224) uint8   wrist camera (CHW)
  observations/state        (T, 7)           float32  right arm qpos
  actions                   (T, 7)           float32  velocity commands
  metadata/language_instruction              bytes    task description

Output LeRobot dataset key names match LIBERO convention so the existing
LiberoInputs/LiberoOutputs transforms and norm stats work out of the box.

Usage (run from the openpi directory with uv):
  cd openpi
  uv run ../convert_to_lerobot.py
  uv run ../convert_to_lerobot.py --fps 10 --push-to-hub
"""

import dataclasses
import pathlib
import shutil

import h5py
import numpy as np
import tqdm
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIM_HZ      = 100            # simulation control rate (N_SUBSTEPS=5, DT=0.002 s)
DEFAULT_FPS = 10             # store at 10 Hz — matches LIBERO convention
REPO_ID     = "local/baxter_cube_push"
RAW_DIR     = pathlib.Path(__file__).parent / "data" / "demos"
IMG_H, IMG_W = 224, 224


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    raw_dir:     pathlib.Path = RAW_DIR
    repo_id:     str          = REPO_ID
    fps:         int          = DEFAULT_FPS
    push_to_hub: bool         = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args) -> None:
    # ── resolve episode files ──────────────────────────────────────────────
    ep_files = sorted(args.raw_dir.glob("episode_*.hdf5"))
    if not ep_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {args.raw_dir}")
    print(f"Found {len(ep_files)} episodes in {args.raw_dir}")

    # ── clean previous run ────────────────────────────────────────────────
    out_path = HF_LEROBOT_HOME / args.repo_id
    if out_path.exists():
        print(f"Removing existing dataset at {out_path}")
        shutil.rmtree(out_path)

    # ── create empty LeRobot dataset ─────────────────────────────────────
    # Key names (image, wrist_image, state, actions) match LIBERO convention
    # so LiberoInputs / LiberoOutputs transforms apply directly.
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="baxter",
        fps=args.fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (IMG_H, IMG_W, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (IMG_H, IMG_W, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # ── downsample stride: sim runs at SIM_HZ, store at args.fps ─────────
    stride = max(1, round(SIM_HZ / args.fps))
    print(f"Downsampling {SIM_HZ} Hz → {args.fps} Hz  (stride={stride})")

    # ── populate ──────────────────────────────────────────────────────────
    for ep_path in tqdm.tqdm(ep_files, desc="Converting episodes"):
        with h5py.File(ep_path, "r") as f:
            images = f["observations/image"][:]        # (T, 3, H, W) uint8
            wrists = f["observations/wrist_image"][:]  # (T, 3, H, W) uint8
            states = f["observations/state"][:]        # (T, 7) float32
            actions = f["actions"][:]                  # (T, 7) float32
            lang = f["metadata/language_instruction"][()].decode()

        T = images.shape[0]
        for i in range(0, T, stride):
            dataset.add_frame({
                # Transpose CHW → HWC for LeRobot image convention
                "image":       np.transpose(images[i], (1, 2, 0)),
                "wrist_image": np.transpose(wrists[i], (1, 2, 0)),
                "state":       states[i],
                "actions":     actions[i],
                "task":        lang,
            })
        dataset.save_episode()

    print(f"\nDataset saved to: {out_path}")
    print(f"  episodes : {dataset.num_episodes}")
    print(f"  frames   : {dataset.num_frames}")
    print(f"  fps      : {args.fps}")
    print(f"  repo_id  : {args.repo_id}")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["baxter", "mujoco", "cube-push"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )
        print("Pushed to HuggingFace Hub.")


if __name__ == "__main__":
    tyro.cli(main)
