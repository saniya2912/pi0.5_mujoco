# Development Log — π0.5 + Baxter MuJoCo

A complete record of everything built, every decision made, and every bug fixed in this project.

**Goal:** Fine-tune the π0.5 Vision-Language-Action model on a Baxter robot arm simulated in MuJoCo, performing the task *"move the cube to the right"*.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Simulation Model — baxter.xml](#2-simulation-model--baxterxml)
3. [Coordinate System and Key Values](#3-coordinate-system-and-key-values)
4. [First Attempt: Kinematic Control](#4-first-attempt-kinematic-control)
5. [Dynamic Control — The Full Journey](#5-dynamic-control--the-full-journey)
   - [5.1 Why kinematic isn't enough for training data](#51-why-kinematic-isnt-enough-for-training-data)
   - [5.2 Actuator authority problem — first failure](#52-actuator-authority-problem--first-failure)
   - [5.3 Self-collision at HOME — root cause](#53-self-collision-at-home--root-cause)
   - [5.4 Feedforward sign bug](#54-feedforward-sign-bug)
   - [5.5 Arm-table collision during approach](#55-arm-table-collision-during-approach)
   - [5.6 Tracking lag — gain and step tuning](#56-tracking-lag--gain-and-step-tuning)
6. [Final Controller Design](#6-final-controller-design)
7. [Episode Structure and Scripted Policy](#7-episode-structure-and-scripted-policy)
8. [Demo Collection Results](#8-demo-collection-results)
9. [LeRobot Conversion](#9-lerobot-conversion)
10. [File Reference](#10-file-reference)
11. [Next Steps](#11-next-steps)

---

## 1. Project Overview

**Pipeline:**
```
MuJoCo simulation
  └─ Baxter right arm pushes a red cube across a table
       └─ record_demos.py  →  data/demos/episode_NNNN.hdf5  (100 episodes)
            └─ convert_to_lerobot.py  →  ~/.cache/huggingface/lerobot/local/baxter_cube_push
                 └─ [next] compute_norm_stats  →  fine-tune π0.5 with LoRA
```

**Stack:**
- MuJoCo 3.x (physics + rendering)
- Python bindings: `mujoco`, `h5py`, `numpy`, `tyro`
- openpi framework (in `openpi/`) for π0.5 policy
- LeRobot 0.1.0 for dataset format
- GPU: RTX 5090 (32 GB) — LoRA fine-tuning feasible

---

## 2. Simulation Model — baxter.xml

`models/baxter.xml` is a hand-written MJCF file built from Baxter's ROS URDF (`models/urdf/baxter.urdf`). The URDF uses `package://` mesh paths and is not loadable directly in MuJoCo; the XML was written by hand.

### Degrees of freedom

| Segment | Joints | qpos indices |
|---|---|---|
| Cube (free joint) | 7 (3 pos + 4 quat) | 0–6 |
| Head pan | 1 | 7 |
| Right arm (s0–w2) | 7 | 8–14 |
| Left arm (s0–w2) | 7 | 15–21 |

**Total: nq=22, nv=21, nu=15**

### Actuators

All 15 DOF use MuJoCo velocity actuators: `force = kv × (ctrl − qvel)`.

| Joint group | kv | ctrlrange | Max force |
|---|---|---|---|
| Shoulder/elbow (s0, s1, e0, e1) | 30 | ±3.0 rad/s | 90 Nm |
| Wrist (w0, w1, w2) | 10 | ±4.0 rad/s | 40 Nm |

> **History:** kv started at 10 (from URDF), then 30 was needed to overcome gravity. ctrlrange started at ±1.5, then doubled to ±3.0 once we understood the tracking lag problem (see §5.6).

### Cameras

| Name | Mount | FoVy |
|---|---|---|
| `scene_camera` | World-fixed (front-right, elevated) | 55° |
| `right_hand_camera` | Right wrist | 90° |
| `left_hand_camera` | Left wrist (unused in recording) | 90° |

### Collision geometry

Each body has two geom classes:
- `class="visual"` — STL mesh, `contype=0 conaffinity=0` (never collides)
- `class="collision"` — primitive cylinder/box, participates in contact

#### Collision bitmask design (final)

This was one of the most significant design decisions (see §5.5 for full story).

| Geom | contype | conaffinity | Meaning |
|---|---|---|---|
| Arm/head collision | 2 | 4 | Arm only collides with objects of type 4 (cube) |
| Table top | 1 | 4 | Table only responds to objects of type 4 (cube) |
| Floor | 1 | 4 | Same as table |
| Cube | 4 | 5 (=4+1) | Cube collides with arm (type 2 → 4&4=4≠0) and table (type 1 → 4&4=4≠0) |

Collision check formula: `collides ⟺ (A.contype & B.conaffinity) | (B.contype & A.conaffinity) ≠ 0`

Verified pairs:

| Pair | Result | Why |
|---|---|---|
| arm ↔ arm | ✗ | (2&4)|(2&4) = 0 — no self-collision |
| arm ↔ table | ✗ | (2&4)|(1&4) = 0 — arm passes through table |
| arm ↔ cube | ✓ | (2&5)|(4&4) = 4 — arm hits cube |
| cube ↔ table | ✓ | (4&4)|(1&5) = 5 — cube rests on table |

> **Why arm passes through table:** The approach trajectory from HOME to APPROACH_BASE takes the forearm through the table surface. With dynamic control, this would cause a 40 Nm constraint force blocking the arm. Allowing the arm to pass through the table is physically an approximation but is correct for this simulation since we only care about cube-contact dynamics.

### Key structural values

```
Table: body pos z=0.475, top half-height=0.025 → table surface z = 0.500
Cube:  body pos z=0.525, half-size=0.025       → cube bottom z = 0.500, top z = 0.550
       Spawned at (0.65, −0.20 ± 0.04, 0.525)
```

> **Bug fixed:** Original cube spawn had z=0.55 (centre), which placed it floating 2.5 cm above the table. Corrected to z=0.525.

### Torso collision

The torso collision geom is explicitly disabled (`contype=0 conaffinity=0`). At HOME pose, Baxter's arm cylinders penetrate the torso bounding box, which generated hundreds of Nm of constraint force and made all control attempts fail. Disabling torso collision eliminates this entirely.

---

## 3. Coordinate System and Key Values

Baxter's base is at the world origin. The workspace is in front of the robot at roughly x=0.65 m.

```
  +X  →  forward (away from robot base)
  +Y  →  left (from robot's perspective)
  +Z  →  up
```

"Move the cube to the right" = move cube in **−Y direction**.

### Joint definitions (right arm)

| Joint | Symbol | HOME value | APPROACH_BASE |
|---|---|---|---|
| Shoulder rotation | s0 | 0.0 rad | 0.886 rad |
| Shoulder lift | s1 | −0.9599 rad | 1.04 rad |
| Elbow rotation | e0 | 0.0 rad | 0.5 rad |
| Elbow flex | e1 | 2.0 rad | 0.0 rad |
| Wrist rotation | w0 | 0.0 rad | 0.0 rad |
| Wrist flex | w1 | 0.7854 rad | 0.929 rad |
| Wrist rotation | w2 | 0.0 rad | 0.0 rad |

### FK-derived endpoint mapping

From a forward-kinematics grid scan with joints 1–6 fixed at APPROACH_BASE:

```
endpoint_y ≈ BASE_Y + DY_DS0 × (s0 − BASE_S0)
BASE_S0 = 0.886,  BASE_Y = −0.111,  DY_DS0 = 0.593 rad/m
```

This lets us compute the `s0` angle needed to align the arm with any cube Y position.

---

## 4. First Attempt: Kinematic Control

The original `record_demos.py` used **kinematic control**: directly writing joint positions to `d.qpos` every step, bypassing the actuator model entirely.

```python
d.qpos[RIGHT_QPOS] = target_qpos  # kinematic: qpos set directly
mujoco.mj_step(m, d)
```

This worked mechanically — the arm followed perfectly smooth trajectories with 77% episode success rate, generating 100 demos in ~130 attempts. The cube was fully dynamic (free joint) so it responded to arm contact forces correctly.

**Why this is insufficient for VLA training:**

The recorded "actions" were not real actuator commands. They were back-calculated implied velocities (`Δqpos / Δt`), which don't correspond to anything a real or simulated robot with actuators would actually execute. Training π0.5 on these actions would teach the policy to produce unrealizable commands that assume infinite actuator authority and no dynamics.

---

## 5. Dynamic Control — The Full Journey

The task: replace `d.qpos[RIGHT_QPOS] = target` with real `d.ctrl` commands and record what the actuators actually output.

### 5.1 Why kinematic isn't enough for training data

The π0.5 model's action space is actuator velocity commands. Training on kinematic "implied velocities" creates a distribution mismatch: those values assume instantaneous joint teleportation, while at inference time the model would produce commands for a real velocity-controlled arm that experiences gravity, inertia, and contact forces.

### 5.2 Actuator authority problem — first failure

**Problem:** With kv=10, ctrlrange=±1.5, maximum actuator force = 10 × 1.5 = **15 Nm**. The gravity torque on the s1 joint with the arm extended is **~31 Nm**. The arm collapsed immediately — the actuator could not even hold position against gravity.

**First fix attempt:** Add gravity feedforward:
```python
ctrl = CTRL_GAIN × (target − current) − qfrc_bias / kv
```
This failed identically. Both `+qfrc_bias/kv` and `−qfrc_bias/kv` produced bad results.

**Root cause (discovered):** At HOME pose, Baxter's arm cylinders were penetrating the torso bounding box — MuJoCo reported **12 deeply-penetrating contacts** generating constraint forces of hundreds of Nm. These overwhelmed any controller. The `qfrc_bias` values were useless because the actual joint forces were dominated by contact forces, not gravity.

### 5.3 Self-collision at HOME — root cause

**Investigation:**
```python
print(d.ncon)  # → 12 contacts at HOME
```
The arm collision cylinders overlapped with the torso box geom, causing massive constraint forces (`qacc ≈ 460 rad/s²` on s1 vs. ~10 rad/s² expected from gravity alone).

**Fix 1 — Disable torso collision:**
```xml
<geom class="collision" name="torso_c" ... contype="0" conaffinity="0"/>
```
Result: ncon dropped from 12 → 5. Still not zero.

**Fix 2 — Disable arm self-collision via bitmasks:**

Changed the default collision class for arm geoms:
```xml
<default class="collision">
  <geom contype="2" conaffinity="1" .../>
</default>
```
With `contype=2` and `conaffinity=1`: arm-vs-arm check = `(2&1)|(2&1) = 0`. Arms don't self-collide.

Result: ncon=0 at HOME. Arm now settles perfectly to HOME within 60 control steps.

**Raised kv to 30** at this point (plus ctrlrange=±1.5): max force = 45 Nm, marginally enough. Instability was observed at kv=50 (too stiff for 2 ms timestep). Settled on kv=30.

### 5.4 Feedforward sign bug

**Problem:** Even after fixing collisions, the arm settled to s1=−1.199 instead of HOME (s1=−0.9599). The arm consistently drifted ~0.24 rad in the wrong direction.

**Investigation:**

Empirical test at HOME with ctrl=0:
```
s1: −0.9599 → −0.9679  (arm moves NEGATIVE)
```

MuJoCo's `qfrc_bias[s1] = +10.21 Nm` at HOME. The arm moves negative (downward for the folded position). This means the MuJoCo dynamics equation is:

```
M × qacc = τ_actuator − qfrc_bias
```

To hold position (qacc=0): `τ_actuator = qfrc_bias` → `ctrl = +qfrc_bias / kv` **(positive sign)**.

**The bug:** The initial implementation used `−qfrc_bias / kv` (negative sign), which ADDED to the gravity force instead of cancelling it. This explained the consistent negative drift: the "compensation" was making the problem worse.

**Verification of equilibrium with wrong sign:**

At the converged state (s1=−1.199), we measured:
- `qfrc_bias[s1] = +14.37 Nm`
- `ctrl = 0.479 rad/s`
- `τ = 30 × 0.479 = 14.37 Nm`

With the correct equation `M×qacc = τ − qfrc_bias = 14.37 − 14.37 = 0`: the arm was in equilibrium with `feedforward − feedback` exactly cancelling at the wrong position.

**Fix:**
```python
feedforward = +d.qfrc_bias[7:14] / KV_RIGHT   # was: −qfrc_bias/kv
```

After fix: `err_s1 = 0.0000` after settle. Perfect.

> This was the most subtle bug in the project. The sign depends on how MuJoCo defines `qfrc_bias` in the equations of motion. The key empirical test is: at HOME with ctrl=0, does the arm accelerate in the + or − direction? At HOME, gravity pulls s1 negative, so qfrc_bias[s1]=+10.21 (positive) but the arm moves negative. This confirms `M×qacc = τ − qfrc_bias`, requiring positive feedforward.

### 5.5 Arm-table collision during approach

**Problem:** Phase 0b (arm extension from HOME to APPROACH_BASE) was stuck. The arm reached s1=0.694 instead of target s1=1.04. Increasing gain or steps made no difference — it always converged to ~0.71.

**Investigation:**
```python
print(d.qfrc_constraint[7:14])
# → [−4.06, −40.98, 5.18, 1.74, 0, 0, 0] at s1=0.71
```

`qfrc_constraint[s1] = −40.98 Nm`. A 40 Nm constraint force pulling s1 negative. Also: `ncon=1`, contact: `rle_c ↔ table_top`.

**Root cause:** During the joint-space interpolation from HOME to APPROACH_BASE, the right forearm/elbow geom descends through the table surface. The table constraint force blocked the arm.

At APPROACH_BASE itself, there was also `rle_c ↔ table_top` contact — the final configuration also partially intersects the table. This was designed for kinematic control where penetration is ignored.

**Fix — Redesign collision bitmasks:**

Make arm pass through table while keeping arm-cube and cube-table collisions:

| Geom | Before | After |
|---|---|---|
| Arm default | contype=2, conaffinity=1 | contype=2, conaffinity=**4** |
| Table top | contype=1, conaffinity=**1** | contype=1, conaffinity=**4** |
| Floor | contype=1, conaffinity=**1** | contype=1, conaffinity=**4** |
| Cube | contype=**1**, conaffinity=**1** | contype=**4**, conaffinity=**5** |

Verification:
- arm(2,4) vs table(1,4): `(2&4)|(1&4) = 0` — no collision ✓
- arm(2,4) vs cube(4,5): `(2&5)|(4&4) = 4` — collides ✓
- cube(4,5) vs table(1,4): `(4&4)|(1&5) = 5` — collides ✓

After fix: ncon=4 (cube on table only). Arm approaches without hitting table.

### 5.6 Tracking lag — gain and step tuning

**Problem:** Even after fixing collisions, phase 0b only reached s1=0.694 out of 1.04. No ctrlrange saturation was occurring.

**Analysis:** The P-controller has inherent steady-state tracking lag when following a ramp target:

```
lag ≈ ramp_rate / gain

phase 0b ramp rate: Δs1 = 2.0 rad over 300 steps × 10ms = 0.667 rad/s
lag at gain=4: 0.667 / 4 = 0.167 rad
```

But the gravity feedforward consumed part of the control authority: at s1≈0.7, `qfrc_bias/kv = −0.796 rad/s`. The net command toward target = `feedback − 0.796`, reducing effective gain.

Additionally, phase 0a (s0 sweep) was not converging fully. For cube_y=−0.162, s0_approach=0.952 — a large sweep. With 200 steps (2 seconds) and gain=4, the arm reached s0=0.777 instead of 0.925, meaning it started phase 1 already past the cube.

**Fix — tuning sweep:**

| Parameter | Before | After |
|---|---|---|
| CTRL_GAIN | 4.0 | **6.0** |
| REACH_STEPS_A | 200 | **300** |
| REACH_STEPS_B | 300 | **400** |

Result: 90% episode success rate (18/20 in testing). The 2 failing cases are extreme edge cases with cube_y≈−0.162, requiring s0_approach=0.952 — the highest s0 the arm must reach, at the boundary of what's achievable.

Also increased ctrlrange for shoulder/elbow to **±3.0 rad/s** (from ±1.5), giving max force = 90 Nm and leaving ~2 rad/s headroom after gravity compensation.

---

## 6. Final Controller Design

```
ctrl[i] = CTRL_GAIN × (target_qpos[i] − d.qpos[i])   ← P feedback
        + d.qfrc_bias[i] / kv[i]                       ← gravity feedforward
```

Applied via `_apply_ctrl(m, d, dq)` which:
1. Adds Gaussian noise `N(0, 0.01)` to each joint command
2. Clips to `m.actuator_ctrlrange[RIGHT_CTRL]`
3. Simultaneously holds the left arm at HOME using the same P + feedforward formula
4. Writes to `d.ctrl[:]`

Physics integration: **N_SUBSTEPS=5** MuJoCo steps per control step (10 ms effective period at dt=2 ms = 100 Hz).

### MuJoCo dynamics equation

```
M × qacc = kv × (ctrl − qvel) − qfrc_bias
```

At equilibrium (qacc=0, qvel=0): `ctrl = qfrc_bias / kv` (positive sign).

This is the key equation. The `qfrc_bias` is the generalized force from gravity and Coriolis that must be **cancelled** by the actuator. The positive sign is non-obvious — it was determined empirically and confirmed by equilibrium analysis.

---

## 7. Episode Structure and Scripted Policy

Each episode runs three phases. At each step, the P+feedforward controller tracks a linearly interpolated target joint configuration.

### Phase 0a — Shoulder sweep (300 steps = 3 s)

Arm stays folded at HOME[1:]. Only s0 sweeps from 0 → s0_approach.

```python
target[0] = lerp(HOME[0], s0_approach, alpha)
target[1:] = HOME[1:]
```

s0_approach is computed from the cube's Y position:
```python
s0_approach = s0_for_y(cube_y + 0.09)   # 9 cm +Y of cube face
```

### Phase 0b — Arm extension (400 steps = 4 s)

s0 stays fixed at s0_approach. Joints 1–6 interpolate from HOME to APPROACH_BASE.

```python
target[0] = s0_approach  # fixed
target[1:] = lerp(HOME[1:], approach_config[1:], alpha)
```

### Phase 1 — Push (≤600 steps = ≤6 s)

s0 sweeps at constant rate `PUSH_RATE=−0.003 rad/step` until:
- Cube moves ≥6 cm in −Y (success), or
- s0 reaches s0_push_end (all push steps exhausted)

```python
s0_push_end = s0_for_y(cube_y - 0.16)   # push 16 cm past cube
```

### Cube randomisation

Each episode spawns the cube at:
```
x = 0.65 m (fixed)
y = −0.20 + uniform(−0.04, +0.04) m
z = 0.525 m (resting on table)
```

The scripted policy adapts s0_approach and s0_push_end to the actual cube Y position.

### Success criterion

Episode saved if cube moves ≥0.06 m (6 cm) in −Y by end of phase 1.

---

## 8. Demo Collection Results

```
python record_demos.py --no-viewer --n-episodes 100
```

**Result:** 100 successful demos in 108 attempts (**92.6% yield**).

The 8 failures were all cube_y ≈ −0.162 to −0.167 (extreme +Y edge of randomisation range), where s0_approach=0.95+ requires the arm to sweep further than it converges to in 300 steps.

### HDF5 file format

Each `data/demos/episode_NNNN.hdf5` contains:

| Key | Shape | Type | Description |
|---|---|---|---|
| `observations/image` | (T, 3, 224, 224) | uint8 | Scene camera, CHW format |
| `observations/wrist_image` | (T, 3, 224, 224) | uint8 | Right wrist camera, CHW |
| `observations/state` | (T, 7) | float32 | Right arm qpos (s0–w2) |
| `actions` | (T, 7) | float32 | Actual `d.ctrl[RIGHT_CTRL]` commands |
| `metadata/success` | scalar | bool | Always True |
| `metadata/episode_length` | scalar | int | T ≈ 761 |
| `metadata/cube_start_pos` | (3,) | float32 | Cube XYZ at episode start |
| `metadata/language_instruction` | bytes | — | `b"move the cube to the right"` |

Typical T ≈ 761 steps at 100 Hz = 7.6 seconds per episode.

---

## 9. LeRobot Conversion

**Script:** `convert_to_lerobot.py` (run from `openpi/` directory via `uv run`)

```bash
cd openpi
uv run ../convert_to_lerobot.py
```

### Design decisions

**Key names match LIBERO convention** (`image`, `wrist_image`, `state`, `actions`, `task`). This means the existing `LiberoInputs` and `LiberoOutputs` transforms in openpi apply directly without modification. No custom transform class needed.

**Downsampling from 100 Hz → 10 Hz** (stride=10). π0.5 and most VLA models are trained at 10–15 Hz. Storing 761 frames at 100 Hz per episode would result in ~76,000 total frames for 100 episodes — excessive for a simple task. At 10 Hz: 77 frames per episode × 100 episodes = **7,700 frames total**.

**Image transposition:** HDF5 stores images as `(C, H, W)` uint8. LeRobot expects `(H, W, C)`:
```python
np.transpose(images[i], (1, 2, 0))   # CHW → HWC
```

LeRobot loads them back as `(3, 224, 224)` float32 in `[0, 1]` (auto-normalized).

**Storage format:** In lerobot 0.1.0, images are stored embedded in parquet files (~3 MB per episode). No separate video files. `save_episode()` handles all encoding internally.

### Output dataset

```
~/.cache/huggingface/lerobot/local/baxter_cube_push/
```

| Property | Value |
|---|---|
| Episodes | 100 |
| Frames | 7,700 |
| FPS | 10 Hz |
| State range | [−0.96, 2.0] rad (right arm qpos) |
| Action range | [−1.0, 0.8] rad/s (velocity commands) |
| Task | "move the cube to the right" |
| Storage | 100 parquet files (~3 MB each, ~300 MB total) |

---

## 10. File Reference

```
pi0.5_mujoco/
├── models/
│   ├── baxter.xml              MJCF model (hand-written)
│   ├── meshes/                 STL visual meshes from baxter_description
│   └── urdf/baxter.urdf        Original ROS URDF (reference only)
│
├── record_demos.py             Scripted demo recorder
│                               → writes data/demos/episode_NNNN.hdf5
│
├── visualize_demo.py           Viewer-only replay (same controller, no saving)
│
├── convert_to_lerobot.py       HDF5 → LeRobot dataset converter
│                               → run from openpi/ with: uv run ../convert_to_lerobot.py
│
├── openpi/                     π0.5 framework (policy, training, inference)
│   ├── scripts/
│   │   ├── compute_norm_stats.py
│   │   └── train.py
│   ├── src/openpi/policies/
│   │   └── libero_policy.py    LiberoInputs/LiberoOutputs (reused for Baxter)
│   └── examples/
│       ├── libero/             Closest reference for our data format
│       └── aloha_real/         Reference for HDF5-based conversion
│
├── data/demos/                 100 HDF5 demos (git-ignored)
│
├── README.md                   Usage guide
└── DEVLOG.md                   This file
```

---

## 11. Next Steps

### Step 3 — Define training config

Create a `BaxterDataConfig` in `openpi/src/openpi/training/config.py` that reuses `LiberoInputs`/`LiberoOutputs` with the dataset `repo_id="local/baxter_cube_push"`.

### Step 4 — Compute norm stats

```bash
cd openpi
uv run scripts/compute_norm_stats.py --config-name <baxter_config>
```

Norm stats JSON will be saved to `assets/local/baxter_cube_push/norm_stats.json`. Watch for small `std` values in rarely-used action dimensions (e.g., w2 which stays near 0 throughout) — these can cause normalisation blow-up and may need manual clamping.

### Step 5 — LoRA fine-tuning

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py baxter_config --exp-name=run_001
```

The RTX 5090 (32 GB VRAM) can handle LoRA fine-tuning but not full fine-tuning of the full π0.5 base model.

### Step 6 — Inference in simulation

Serve the fine-tuned checkpoint and close the loop: run the policy in the MuJoCo viewer, replacing the scripted controller with π0.5 outputs.

---

*Last updated: 2026-03-24*
