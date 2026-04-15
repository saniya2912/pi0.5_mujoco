"""Standalone multi-task training launcher for π0.5 Baxter.

Trains on 3 tasks from local/baxter_multitask:
  - "move the cube to the right"   (100 demos)
  - "wave your hand"               (100 demos)
  - "push the block to the front"  (100 demos)

Differences from single-task run_001:
  - repo_id: local/baxter_multitask  (300 demos total)
  - num_train_steps: 40,000          (3× data → similar epoch count as run_001)
  - decay_steps: 40,000
  - weights loaded from base π0.5 (not the single-task checkpoint)

Pre-requisites
--------------
  1. All three demo directories must be populated:
       python record_demos.py --no-viewer           (if not already done)
       python record_demos_wave.py --no-viewer
       python record_demos_push_front.py --no-viewer
  2. Convert to LeRobot format:
       cd openpi && uv run ../convert_to_lerobot_multitask.py
  3. Compute normalisation statistics:
       cd openpi && uv run ../compute_norm_stats_multitask.py

Usage (run from the openpi directory with uv):
  cd openpi
  WANDB_API_KEY=<key> XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \\
      uv run ../train_multitask.py --exp-name run_002
  uv run ../train_multitask.py --exp-name run_002 --resume
  uv run ../train_multitask.py --exp-name run_002 --overwrite
"""

import dataclasses
import importlib.util
import pathlib

import tyro

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi.training.config import DataConfig, LeRobotLiberoDataConfig


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _build_config(exp_name: str, overwrite: bool, resume: bool) -> _config.TrainConfig:
    return _config.TrainConfig(
        name="pi05_baxter_multitask",
        exp_name=exp_name,
        overwrite=overwrite,
        resume=resume,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/baxter_multitask",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=4,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2e-4,
            decay_steps=40_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # EMA disabled for LoRA (saves memory, no benefit)
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Load from the base π0.5 checkpoint (not the single-task fine-tuned one)
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=40_000,
        save_interval=5_000,
        keep_period=10_000,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    exp_name: str
    overwrite: bool = False
    resume: bool = False


def main() -> None:
    args = tyro.cli(Args)
    config = _build_config(args.exp_name, args.overwrite, args.resume)

    # Load scripts/train.py from CWD (expected to be openpi/)
    train_script = pathlib.Path("scripts/train.py").resolve()
    if not train_script.exists():
        raise FileNotFoundError(
            f"scripts/train.py not found at {train_script}\n"
            "Run this script from the openpi directory:\n"
            "  cd openpi && uv run ../train_multitask.py --exp-name run_002"
        )

    spec = importlib.util.spec_from_file_location("_openpi_train", train_script)
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    train_mod.main(config)


if __name__ == "__main__":
    main()
