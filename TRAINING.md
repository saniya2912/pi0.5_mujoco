# π0.5 Baxter LoRA Fine-Tuning

This document covers everything about fine-tuning the **π0.5** Vision-Language-Action model on the Baxter cube-push MuJoCo demonstrations.

---

## Overview

| Property | Value |
|---|---|
| Base model | π0.5 (`pi05_base`) |
| Fine-tuning method | LoRA (Low-Rank Adaptation) |
| Dataset | `local/baxter_cube_push` — 100 episodes, 7700 frames, 10 Hz |
| Task | `"move the cube to the right"` |
| Action space | 7-DOF right arm velocity commands (rad/s) |
| Config name | `pi05_baxter` |
| Training script | `openpi/scripts/train.py` |
| Log | `/tmp/baxter_train.log` |
| Checkpoints | `openpi/checkpoints/pi05_baxter/run_001/` |

---

## Why LoRA

Full fine-tuning of π0.5 requires ~50 GB of GPU memory (the model is ~3B parameters including PaliGemma-2B + 300M action expert). Our RTX 5090 has 32 GB VRAM — too small for full fine-tuning.

LoRA freezes the base model weights and inserts small trainable rank-decomposition matrices into the attention layers. This reduces the trainable parameter count from ~3B to ~tens of millions, cutting peak VRAM to ~17 GB.

The openpi framework has first-class LoRA support via model variants:
- `paligemma_variant="gemma_2b_lora"` — LoRA adapters on the vision-language backbone
- `action_expert_variant="gemma_300m_lora"` — LoRA adapters on the action expert

---

## Training Config

Defined in `openpi/src/openpi/training/config.py` as `pi05_baxter`:

```python
TrainConfig(
    name="pi05_baxter",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="local/baxter_cube_push",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    batch_size=16,
    lr_schedule=CosineDecaySchedule(
        warmup_steps=500,
        peak_lr=2e-4,
        decay_steps=20_000,
        decay_lr=1e-5,
    ),
    optimizer=AdamW(clip_gradient_norm=1.0),
    ema_decay=None,                      # disabled for LoRA
    freeze_filter=...,                   # freezes all non-LoRA weights
    weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=20_000,
    save_interval=2_000,
    keep_period=5_000,
)
```

### Key design choices

**`LeRobotLiberoDataConfig` reused directly.** Our dataset stores observations under the keys `image`, `wrist_image`, `state`, `actions` — the same naming convention used by the LIBERO dataset. The built-in repack transform maps these to `observation/image`, `observation/wrist_image`, `observation/state` (what `LiberoInputs` expects), so no custom transform class was needed.

**`extra_delta_transform=False`.** LIBERO actions are joint-space deltas, so the default config converts them. Our Baxter actions are velocity commands directly from `d.ctrl` — already differential — no secondary conversion needed.

**`prompt_from_task=True`.** The LeRobot dataset stores `"move the cube to the right"` in the `task` field (added via `add_frame({"task": lang, ...})`). This flag tells the data loader to expose it as the `prompt` key.

**`ema_decay=None`.** Exponential moving average of weights is beneficial for full fine-tuning but adds memory overhead and no meaningful benefit for LoRA, where the adapter is small.

**`action_horizon=10`.** At 10 Hz, this gives the model a 1-second action prediction window. Each episode is ~77 frames at 10 Hz.

---

## Data Pipeline

```
HDF5 episodes (100 × ~761 steps @ 100 Hz)
        ↓  convert_to_lerobot.py  (stride=10)
LeRobot dataset  local/baxter_cube_push
   100 episodes, 7700 frames, 10 fps
   keys: image (224×224 HWC uint8)
         wrist_image (224×224 HWC uint8)
         state (7,) float32  — right arm qpos
         actions (7,) float32 — velocity commands
         task: "move the cube to the right"
        ↓  LeRobotLiberoDataConfig repack transform
   observation/image, observation/wrist_image, observation/state
        ↓  LiberoInputs
   image dict: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb (zeros)
   state: (32,) zero-padded from (7,)
   prompt: tokenized language instruction
        ↓  Normalize (quantile norm, from norm_stats.json)
        ↓  PadStatesAndActions → model action dim 32
```

Norm stats were computed with:
```bash
cd openpi
uv run scripts/compute_norm_stats.py --config-name pi05_baxter
```
Output: `openpi/assets/pi05_baxter/local/baxter_cube_push/norm_stats.json`

Summary of computed stats:

| Key | Mean | Std | Notes |
|---|---|---|---|
| `state[0]` (s0) | 0.683 | 0.286 | shoulder sweep range |
| `state[1]` (s1) | −0.311 | 0.727 | large range: home↔approach |
| `state[3]` (e1) | 1.351 | 0.726 | elbow, also large range |
| `actions[1]` (ṡ1) | 0.140 | 0.542 | highest-magnitude velocity |
| `actions[3]` (ė1) | −0.293 | 0.369 | |

---

## Running Training

```bash
cd openpi

# Compute norm stats (only needed once)
uv run scripts/compute_norm_stats.py --config-name pi05_baxter

# Launch training (headless, no wandb)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_baxter \
    --exp-name run_001 \
    --no-wandb-enabled \
    [--overwrite | --resume]
```

To run with wandb (requires API key):
```bash
wandb login
uv run scripts/train.py pi05_baxter --exp-name run_001
```

### Current run status

| Property | Value |
|---|---|
| Run name | `run_001` |
| Started | 2026-03-24 ~20:40 |
| Rate | ~1.5 s/step |
| Estimated completion | ~2026-03-25 05:00 |
| Log | `/tmp/baxter_train.log` |
| First checkpoint | step 2000 |

Monitor:
```bash
tail -f /tmp/baxter_train.log
ls openpi/checkpoints/pi05_baxter/run_001/
```

---

## Checkpoints

Checkpoints are saved every 2000 steps. Checkpoints at steps divisible by 5000 are kept permanently; others are pruned to keep only the latest.

```
openpi/checkpoints/pi05_baxter/run_001/
  2000/
    params/        ← LoRA adapter weights (Orbax format)
    train_state/   ← optimizer state + step count
    assets/        ← copy of norm_stats.json
  5000/            ← kept permanently
  10000/           ← kept permanently
  ...
  20000/           ← final checkpoint
```

---

## OOM Troubleshooting

The first run attempt used full fine-tuning (`Pi0Config(pi05=True)` without LoRA variants) and failed at init:

```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 2.25GiB
Peak usage: ~50 GB (53654943620 bytes)
```

Fix: switch to LoRA variants. Peak usage dropped to ~17 GB (18490814840 bytes after rematerialization), well within the 32 GB RTX 5090 budget.

If OOM persists with LoRA, reduce `batch_size` (try 8) or add gradient accumulation.

---

## Next Steps

1. **Wait for training to complete** (step 20000, ~2026-03-25 05:00)
2. **Serve the fine-tuned checkpoint** and run inference in the MuJoCo viewer:
   ```bash
   # Terminal 1 — policy server
   cd openpi
   uv run scripts/serve_policy.py \
     --checkpoint-dir checkpoints/pi05_baxter/run_001/20000

   # Terminal 2 — MuJoCo inference client
   cd ..
   python run_baxter_vla.py
   ```
3. **Evaluate** — run N episodes in MuJoCo, measure cube-push success rate vs the scripted baseline (92%)
4. **Iterate** — if success rate is low, collect more diverse demos, tune LR, or increase `num_train_steps`
