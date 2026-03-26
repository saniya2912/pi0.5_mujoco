# π0.5 Baxter LoRA Fine-Tuning

---

## Table of Contents

1. [What π0.5 is](#1-what-π05-is)
2. [How the model generates actions — flow matching](#2-how-the-model-generates-actions--flow-matching)
3. [Data pipeline — from HDF5 to model input](#3-data-pipeline--from-hdf5-to-model-input)
4. [One training step](#4-one-training-step)
5. [LoRA — what is actually being trained](#5-lora--what-is-actually-being-trained)
6. [The optimizer](#6-the-optimizer)
7. [Checkpointing](#7-checkpointing)
8. [Training timeline](#8-training-timeline)
9. [What good training looks like](#9-what-good-training-looks-like)
10. [Config reference](#10-config-reference)
11. [How to run](#11-how-to-run)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What π0.5 is

π0.5 is a **Vision-Language-Action (VLA)** model. It takes camera images, a language instruction, and the current robot joint positions as input, and outputs a chunk of future robot actions. It is built on top of **PaliGemma** (Google's 3B-parameter vision-language model) with an added **action expert** (a 300M-parameter transformer). The two components communicate through cross-attention at every transformer layer.

There are two model families in openpi:

| Model | Action generation | When to use |
|---|---|---|
| **π0** | Flow matching (continuous diffusion) | Smooth, dexterous tasks |
| **π0-FAST** | Autoregressive token prediction | High-frequency or long-horizon tasks |

We use **π0.5**, which is π0 that has already been pre-trained by Physical Intelligence on a large and diverse robot dataset before we do our own fine-tuning. This pre-training gives it a strong prior over robot motion that generalises across embodiments.

---

## 2. How the model generates actions — flow matching

This is the most important concept to understand. π0.5 does **not** directly predict `"move joint 1 to angle X"`. Instead it learns to **denoise** a chunk of actions.

### The intuition

Imagine starting from a cloud of random Gaussian noise in action space and gradually pushing that noise toward a realistic robot trajectory. The model learns the **velocity field** — the direction to push at each point in space and time — so that integrating it from `t=1` (pure noise) to `t=0` (clean action) produces a plausible action sequence.

### During training

For each batch, a random diffusion timestep `t` is sampled uniformly from `[0, 1]`. A noisy version of the ground-truth action is constructed by linearly interpolating between the clean action and random noise:

```
x_t = t · noise + (1 - t) · actions        (noisy action at time t)
u_t = noise - actions                        (target velocity field)
```

The model predicts the velocity field `v_θ(x_t, t, observation)` and the loss is the mean squared error between the prediction and the target:

```
loss = MSE(v_θ(x_t, t, obs), u_t)
```

At `t=0`, the noisy action equals the clean action (no noise added). At `t=1`, it is pure Gaussian noise. The model learns to reverse this process.

### During inference

```
Start:  x₁ = Gaussian noise   shape (batch, 10, 32)
        ↓  run model 10 denoising steps
End:    x₀ = clean action chunk
```

The 10 denoising steps integrate the learned velocity field from `t=1` to `t=0` using Euler steps. The output is a **10-step action chunk** — 10 consecutive velocity commands for each of the 7 joints. At 10 Hz, this is a 1-second prediction horizon.

The **action dimension is 32** (the model's internal dimension). We use only the first 7 elements (one per Baxter right-arm joint). The remaining 25 are padded to zero during training and discarded during inference.

---

## 3. Data pipeline — from HDF5 to model input

Every training batch goes through the following chain of transforms. Understanding this is essential for debugging data issues.

```
data/demos/episode_0000.hdf5  ...  episode_0099.hdf5
  100 episodes × ~761 steps @ 100 Hz
        ↓  convert_to_lerobot.py  (stride = 10)
~/.cache/huggingface/lerobot/local/baxter_cube_push/
  100 episodes × ~77 frames @ 10 Hz
  Stored as:
    image         (224,224,3) uint8   ← scene camera, JPEG-compressed in parquet
    wrist_image   (224,224,3) uint8   ← wrist camera
    state         (7,)        float32 ← right arm qpos
    actions       (7,)        float32 ← velocity commands
    task          str                  ← "move the cube to the right"
        ↓  LeRobot DataLoader
  Batches of 4, with action sequences of length 10
  (the DataLoader auto-builds 10-step chunks from consecutive frames)
        ↓  RepackTransform
  Renames dataset keys to match what LiberoInputs expects:
    image         → observation/image
    wrist_image   → observation/wrist_image
    state         → observation/state
    actions       → actions
    task          → prompt   (via prompt_from_task=True)
        ↓  LiberoInputs
  Builds the image dict the model expects:
    image/base_0_rgb        (224,224,3) ← scene camera
    image/left_wrist_0_rgb  (224,224,3) ← wrist camera
    image/right_wrist_0_rgb (224,224,3) ← zeros (we only have 2 cameras)
  image_mask: True for real cameras, False for zero-padded camera
        ↓  Normalize  (quantile norm, from norm_stats.json)
  state   → z-scored per-joint using precomputed mean / std
  actions → z-scored per-joint
        ↓  TokenizePrompt  (PaliGemma SentencePiece tokenizer)
  "move the cube to the right" → int32[200] token ids + bool[200] mask
        ↓  PadStatesAndActions
  state   (7,)  → (32,)    zero-padded to model's internal action_dim
  actions (10,7) → (10,32) zero-padded
        ↓  ResizeImages
  uint8 [0,255] → float32 [-1,1]
        ↓  JAX device transfer + sharding

Final tensors on GPU (batch_size=4):
  images['base_0_rgb']:        (4, 224, 224, 3) float32
  images['left_wrist_0_rgb']:  (4, 224, 224, 3) float32
  images['right_wrist_0_rgb']: (4, 224, 224, 3) float32  ← all zeros
  image_masks:                 (4,) bool  ×3
  state:                       (4, 32)    float32
  tokenized_prompt:            (4, 200)   int32
  tokenized_prompt_mask:       (4, 200)   bool
  actions:                     (4, 10, 32) float32
```

The **action sequence** of length 10 is critical. The data loader takes each frame as an anchor and looks ahead 10 frames to build the target action chunk. This is what the model learns to predict in one shot.

---

## 4. One training step

```python
# Pseudocode of train_step() in scripts/train.py

# 1. Reconstruct the full model from its definition and current parameters
model = nnx.merge(state.model_def, state.params)
model.train()

# 2. Define the loss function
def loss_fn(model, rng, observation, actions):
    loss_per_timestep = model.compute_loss(rng, observation, actions, train=True)
    return jnp.mean(loss_per_timestep)

# Inside model.compute_loss() (pi0.py):
#   a. Sample t ~ Uniform[0,1]
#   b. Sample noise ~ N(0,I)  same shape as actions: (4, 10, 32)
#   c. Interpolate: x_t = t*noise + (1-t)*actions
#   d. target velocity: u_t = noise - actions
#   e. Embed observation (images → patch tokens, language → word tokens, state → MLP)
#   f. Embed noisy actions + timestep → action tokens
#   g. Run joint transformer over all tokens
#   h. Read out predicted velocity from action token positions
#   i. Return MSE(predicted_velocity, u_t)

# 3. Compute gradients — ONLY for LoRA parameters
#    diff_state tells JAX to differentiate only trainable weights
diff_state = nnx.DiffState(0, config.trainable_filter)
loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
    model, rng, observation, actions
)

# 4. AdamW optimizer step on LoRA params only
params = state.params.filter(config.trainable_filter)
updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
new_params = optax.apply_updates(params, updates)

# 5. Write updated LoRA params back into the full parameter tree
#    Frozen base model params are untouched
nnx.update(model, new_params)
new_params = nnx.state(model)

# 6. Return updated train state + metrics
info = {
    "loss":       loss,               # MSE of velocity field prediction
    "grad_norm":  global_norm(grads), # L2 norm of LoRA gradients
    "param_norm": global_norm(kernels) # L2 norm of all kernel weights
}
```

The entire `train_step` is JIT-compiled by JAX on the first call and runs as a single compiled GPU kernel on every subsequent call. This is why startup takes ~60 seconds but each step after that is only ~0.6 seconds.

---

## 5. LoRA — what is actually being trained

Full fine-tuning of π0.5 requires ~50 GB of GPU memory (gradients and optimizer state for all 3B parameters). Our RTX 5090 has 32 GB. LoRA solves this by keeping the base model completely frozen and injecting small trainable matrices at every attention layer.

### How LoRA works

Every attention projection (Q, K, V, O) in the original model is a large matrix:

```
Original:  W ∈ ℝ^(d_model × d_model)      e.g. (2048 × 2048) = 4M params — FROZEN

LoRA adds: ΔW = A × B
           A ∈ ℝ^(d_model × rank)          e.g. (2048 × 16) = 32K params — TRAINED
           B ∈ ℝ^(rank × d_model)          e.g. (16 × 2048) = 32K params — TRAINED
           rank = 16 (much smaller than d_model)

Effective weight: W + A × B
```

The output of each attention layer is computed with the full effective weight, but gradients only flow through `A` and `B`. Everything else is frozen.

### What gets trained in our run

| Component | Variant | Trainable |
|---|---|---|
| PaliGemma vision encoder | frozen | no |
| PaliGemma LLM (18 layers) | `gemma_2b_lora` | LoRA adapters only |
| Action expert (300M) | `gemma_300m_lora` | LoRA adapters only |
| Action input/output projections | — | yes (these are small, <1M params) |

Approximate trainable parameter count: **~30–50 million** out of ~3 billion total.
Peak GPU memory: **~17 GB** (vs ~50 GB for full fine-tuning).

The `freeze_filter` in the config is a JAX filter that identifies all non-LoRA parameters. It is passed to `nnx.DiffState` to exclude frozen weights from gradient computation, and to `tx.init()` to exclude them from the optimizer state.

---

## 6. The optimizer

**AdamW** with cosine decay learning rate schedule:

```
Warmup:        steps 0 → 500      LR ramps linearly from 0 to 2e-4
Peak:          steps 500+         LR = 2e-4
Cosine decay:  over 20,000 steps  LR falls from 2e-4 to 1e-5
Gradient clip: global norm clipped to 1.0

Weight decay:  applied to LoRA weights (regularises adapters, prevents overfitting)
EMA:           disabled (ema_decay=None) — EMA is mainly useful for full fine-tuning
               and adds memory overhead without benefit for LoRA
```

AdamW maintains a first moment (mean of gradients) and second moment (variance of gradients) for each LoRA parameter independently, which makes learning robust to different gradient scales across layers.

Gradient clipping at 1.0 prevents the rare large gradient update from destabilising training. With only 100 demos, a single bad batch could otherwise cause a large spike.

---

## 7. Checkpointing

Checkpoints are saved synchronously (training pauses while writing). This was necessary because async checkpointing — which writes in a background thread while training continues — caused out-of-memory crashes when the checkpoint transfer competed with GPU training for CPU RAM.

### What is saved

```
openpi/checkpoints/pi05_baxter/run_001/
  5000/
    params/         ← ALL model parameters (frozen base + LoRA adapters)
                      ~6 GB on disk, in Zarr/Orbax format
                      used directly for inference without reloading base checkpoint
    train_state/    ← optimizer state for LoRA params + step counter
                      small (~100s of MB)
    assets/
      local/baxter_cube_push/
        norm_stats.json   ← copy of normalisation stats
  10000/            ← kept permanently (keep_period=10000)
  20000/            ← final checkpoint, kept permanently
```

The `params/` item saves the complete model (frozen + LoRA merged) so that inference can load a single checkpoint without needing the original base model separately.

### Checkpoint schedule

| Step | Saved | Kept permanently |
|---|---|---|
| 5000 | yes | no (deleted when 10000 is saved) |
| 10000 | yes | yes |
| 15000 | yes | no |
| 20000 | yes | yes (final) |

---

## 8. Training timeline

Actual timeline for `run_001` (2026-03-26):

```
13:45  Training launch — first two attempts crashed at checkpoints due to OOM
         Root cause: async checkpointing competed with Firefox (~3 GB) for RAM
         Fix: enable_async_checkpointing=False + close Firefox
         ↓
14:56  Third launch (Firefox closed, 26 GB RAM free)
         JAX JIT compilation ~60s
         ↓
14:57  Steps begin at ~1.6 it/s, batch_size=4
         ↓
16:42  Step 10000 checkpoint — saved cleanly (first successful checkpoint)
         ↓
17:27  Step ~14200 — loss plateauing around 0.030
         ↓
~17:45 Step 15000 checkpoint — saved cleanly
         ↓
18:28  Step 19999 — training complete, final checkpoint saved
         Total wall-clock time: ~3.5 hours
```

W&B run:
**https://wandb.ai/patwardhan-saniya-indian-institute-of-technology-gandhinagar/openpi/runs/txq68i0g**

Log file:
```bash
/home/robotlab/Desktop/saniya_ws/pi0.5_mujoco/train_run_001.log
```

---

## 9. What good training looks like

### Reading the training metrics

Each line logged every 100 steps looks like:

```
Step 3000: grad_norm=0.2113, loss=0.0394, param_norm=1806.9763
```

#### `loss` — flow matching MSE

The model predicts a **velocity field** that denoises a noisy action back to the clean action. Concretely:

1. Take a clean action `a` (joint velocities from a demo)
2. Add noise: `x_t = t·ε + (1−t)·a` where `ε ~ N(0,I)` and `t ~ Uniform[0,1]`
3. Ground-truth velocity field: `u_t = ε − a`
4. Model predicts `û_t`; loss = `mean((û_t − u_t)²)`

Loss measures **how wrong the predicted action trajectory is**, in normalised units. There is no universal "good" threshold — it is task-relative. Some irreducible variance remains at convergence because `t` is sampled randomly each step. Actual observed values for this run:

```
Step 100:   loss ≈ 0.100   (model knows little about Baxter yet)
Step 500:   loss ≈ 0.055   (LR warming up, noisy)
Step 3000:  loss ≈ 0.039   (fast learning phase done)
Step 10000: loss ≈ 0.030   (cosine LR decay, fine-tuning)
Step 13000: loss ≈ 0.029   (converging, nearly there)
```

#### `grad_norm` — gradient clipping indicator

All gradients across every LoRA parameter are stacked into one vector. `grad_norm` is its L2 norm, computed **before** clipping.

- **High (>1.0, e.g. steps 700–800):** model is taking large uncertain steps; common early when loss is high. These were clipped to 1.0 by `clip_gradient_norm=1.0`.
- **Low (~0.13 at step 13000):** model makes small, confident adjustments — healthy convergence.
- **Always exactly 1.0:** the clip is always active, which may indicate LR is too high.

Actual observed range: **0.68 → 1.34 → 0.13** over the course of training.

#### `param_norm` — weight drift tracker

L2 norm of **all** model parameters (frozen weights + LoRA adapters). It was ~1804 at step 0 and ~1811 at step 13000 — a change of only **+7 out of 1804 (0.4%)**.

- The frozen PaliGemma backbone is entirely unchanged.
- The LoRA adapters have grown slightly from their zero initialisation.
- The model has not drifted catastrophically. A sudden large jump (e.g. +100) would indicate numerical instability.

The slow, steady rise is exactly what LoRA should look like: only the small adapter matrices `A` and `B` are moving.

---

### Loss curve

```
Step 0:      loss ≈ 0.8–1.2   (random LoRA weights, model predicts garbage)
Step 500:    loss ≈ 0.3–0.5   (LR at peak, rapid initial learning)
Step 2000:   loss ≈ 0.1–0.2   (model has learned the rough motion structure)
Step 10000:  loss ≈ 0.05–0.1  (fine-grained adaptation)
Step 20000:  loss ≈ 0.03–0.08 (converged or mildly overfitting — OK for 100 demos)
```

Note: actual run started lower than these generic estimates because the π0.5 pre-trained base already has a strong robot-motion prior.

With only 100 demonstrations of a single scripted task, some overfitting is expected and desirable — we want the model to execute this specific task reliably.

### Gradient norm

Should stay in the range `0.05–1.0`. If it consistently hits 1.0 (the clip threshold) that is fine — it means the optimizer is working hard. If it spikes and stays above 1.0 across many steps, the LR may be too high.

### Actual results — run_001 (completed)

Full statistics computed from 198 logged steps (every 100 steps, steps 100–19900):

#### Loss

| Metric | Value |
|---|---|
| First (step 100) | 0.1005 |
| **Final (step 19900)** | **0.0328** |
| Minimum | 0.0285 at step 13100 |
| Maximum (worst spike) | 0.1217 at step 2500 |
| Overall mean | 0.0386 |
| Overall std | 0.0150 |
| **Total reduction** | **67.4%** |

Phase-by-phase mean loss — shows clean monotonic descent:

| Phase | Mean loss |
|---|---|
| Steps 0–5k | 0.0559 |
| Steps 5–10k | 0.0350 |
| Steps 10–15k | 0.0321 |
| Steps 15–20k | 0.0311 |

The final phase improved by only 0.001 over 5000 steps — the model had converged by step ~13–15k.

#### Gradient norm

| Metric | Value |
|---|---|
| First (step 100) | 0.6786 |
| Final (step 19900) | 0.1421 |
| Minimum | 0.1191 at step 17100 |
| Maximum | 1.3379 at step 800 |
| Mean | 0.2232 |
| Std | 0.1852 |
| Steps clipped (>1.0) | **3 / 198** — all in first 1000 steps |
| Mean 0–5k | 0.4386 |
| Mean 15–20k | 0.1318 |

Only 3 clipping events in the entire run, all in the initial noisy phase. Training was completely stable after step 1000.

#### Param norm (LoRA drift)

| Phase | Drift |
|---|---|
| Steps 0–5k | +4.4081 (adapters actively learning) |
| Steps 5–10k | +2.4243 (slowing down) |
| Steps 10–15k | +0.4010 (nearly stopped) |
| Steps 15–20k | **+0.0520** (essentially zero — fully converged) |
| **Total drift** | **+7.29 / 1803.86 = 0.404%** |

The drift rate collapsing to +0.05 in the final 5000 steps confirms the model fully memorised the task by step 15k. The base model weights were untouched throughout.

#### Convergence verdict

The model converged cleanly at approximately **step 13000–15000**. The final checkpoint (`19999`) is the best to use for inference. The training run was healthy by all metrics — no instability, no gradient explosions, clean checkpoint writes, and a clear 67% loss reduction from a strong pre-trained starting point.

---

### Signs of a problem

| Symptom | Likely cause |
|---|---|
| Loss doesn't fall below 0.5 after step 2000 | LR too low, or data pipeline issue |
| Loss explodes (NaN or >10) | LR too high, or a bad batch |
| `grad_norm` always exactly 1.0 | Gradient clipping too tight |
| Process dies at checkpoint | OOM during synchronous checkpoint write (see Troubleshooting) |

---

## 10. Config reference

Defined in `openpi/src/openpi/training/config.py` as `TrainConfig(name="pi05_baxter", ...)`.

| Parameter | Value | Why |
|---|---|---|
| `model.pi05` | `True` | Use the π0.5 architecture (pre-trained on diverse robot data) |
| `model.action_horizon` | `10` | Predict 1 second of actions at 10 Hz |
| `model.discrete_state_input` | `False` | Use continuous state embedding (standard for π0.5) |
| `model.paligemma_variant` | `gemma_2b_lora` | LoRA adapters on the 2B vision-language backbone |
| `model.action_expert_variant` | `gemma_300m_lora` | LoRA adapters on the 300M action expert |
| `data.repo_id` | `local/baxter_cube_push` | Our 100-episode LeRobot dataset |
| `data.prompt_from_task` | `True` | Use the `task` field as the language prompt |
| `data.extra_delta_transform` | `False` | Actions are already velocity commands, no delta conversion |
| `batch_size` | `4` | Reduced from 16 to fit within 32 GB VRAM + RAM headroom for checkpoints |
| `lr_schedule.peak_lr` | `2e-4` | Standard LoRA learning rate |
| `lr_schedule.warmup_steps` | `500` | Ramp-up before peak LR |
| `num_train_steps` | `20000` | ~3.5 hours at 1.6 it/s |
| `save_interval` | `5000` | Checkpoint every ~50 min |
| `ema_decay` | `None` | Disabled for LoRA (saves memory, no benefit) |
| `weight_loader` | `pi05_base` | Load pre-trained π0.5 base weights from GCS |

---

## 11. How to run

```bash
cd openpi

# Step 1 — compute normalisation statistics (one-time, already done)
uv run scripts/compute_norm_stats.py --config-name pi05_baxter
# Output: openpi/assets/pi05_baxter/local/baxter_cube_push/norm_stats.json

# Step 2 — launch training
WANDB_API_KEY=<your_key> \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  uv run scripts/train.py pi05_baxter \
    --exp-name run_001 \
    [--overwrite | --resume]

# Resume if training was interrupted after a successful checkpoint
uv run scripts/train.py pi05_baxter --exp-name run_001 --resume

# Monitor
tail -f /home/robotlab/Desktop/saniya_ws/pi0.5_mujoco/train_run_001.log
```

---

## 12. Troubleshooting

### Process dies at every checkpoint (OOM)

**Cause:** The synchronous checkpoint transfers the full ~6 GB model from GPU to CPU RAM. If CPU RAM is tight (Firefox, VS Code, swap full), the Linux OOM killer kills the process.

**Fixes applied:**
- `enable_async_checkpointing=False` in `checkpoints.py` — prevents concurrent GPU training and CPU write competing for RAM
- `batch_size=4` — reduces GPU activation memory
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` — leaves 15% GPU memory as headroom

**If it still crashes:** Close browser tabs / VS Code before launching training to free ~5–10 GB of RAM.

### wandb login fails with "API key must be 40 characters"

The installed wandb (0.19.11) doesn't accept the new `wandb_v1_...` key format via `wandb login`. Pass the key as an environment variable instead:
```bash
WANDB_API_KEY=wandb_v1_... uv run scripts/train.py ...
```

### Training is stuck at "JIT compiling" for >5 minutes

Normal on first run — XLA compiles the entire computation graph. After the first run, compilation is cached at `~/.cache/jax/`. Subsequent runs skip this.

### Loss doesn't decrease at all

Check the wandb `camera_views` image logged at step 0. If the images look garbled or all black, there is a data pipeline issue. The most likely cause is incorrect CHW↔HWC transposition in `convert_to_lerobot.py` or `LiberoInputs`.
