# π0.5 + Baxter MuJoCo — Demo Collection

Fine-tuning the **π0.5 Vision-Language-Action** model on a Baxter robot simulated in MuJoCo.
The current step is **scripted demo collection** — generating training demonstrations for the task *"move the cube to the right"*.

---

## Repository Structure

```
models/
  baxter.xml          — Hand-written MJCF: 15-DOF Baxter, table, red cube, 3 cameras
  meshes/             — STL visual meshes (from baxter_description ROS package)
  urdf/baxter.urdf    — Original ROS URDF (reference only, not loadable as-is)
record_demos.py       — Scripted demo recorder (see below)
visualize_demo.py     — Viewer-only replay (no saving) for inspection
scripts/              — Camera calibration / diagnostic scripts
openpi/               — π0.5 framework (policy server, training utilities)
data/demos/           — Recorded HDF5 demonstrations (git-ignored, regenerate locally)
```

---

## Setup

```bash
python3 -m venv pi0.5_venv
source pi0.5_venv/bin/activate
pip install -r requirements.txt
```

Requires a display or EGL for headless rendering:
```bash
export MUJOCO_GL=egl   # headless (e.g. SSH / server)
# or leave unset for an on-screen viewer
```

---

## Demo Collection

### What it does

`record_demos.py` runs fully scripted demonstrations of the right arm pushing a red cube in the −Y direction ("to the right" from Baxter's perspective) using a **real dynamic actuator controller** — velocity commands written to `d.ctrl`, not kinematic qpos overrides.

**Task:** `"move the cube to the right"`
**Robot:** right arm only (7 DOF, dynamic velocity control)
**Cube:** spawned at a randomised Y position (±4 cm), fixed X

### Controller

Each control step computes:

```
ctrl = gain × (target_qpos − current_qpos)   [P feedback]
     + qfrc_bias / kv                         [gravity/Coriolis feedforward]
```

The feedforward term uses MuJoCo's `qfrc_bias` (gravity + Coriolis forces) divided by the actuator gain `kv`, derived from the MuJoCo dynamics `M·q̈ = τ_actuator − qfrc_bias`. This keeps the arm at its target pose without sagging. The left arm is held at HOME with the same controller throughout.

- **CTRL_GAIN:** 6.0 rad/s per rad
- **N_SUBSTEPS:** 5 physics steps per control step (10 ms effective period)
- **Actuator kv:** 30 (shoulder/elbow), 10 (wrist); ctrlrange ±3.0 / ±4.0 rad/s
- **Collision:** arm geoms pass through the table (`contype=2, conaffinity=4`) but collide with the cube (`contype=4`), so approach dynamics are clean and cube contact forces are physical

### Episode structure

Each episode has three phases:

| Phase | Steps | What happens |
|---|---|---|
| **0a** — s0 sweep | 300 | Shoulder joint sweeps to align arm laterally while arm stays folded at home |
| **0b** — arm extension | 400 | Joints 1–6 interpolate from home to approach config at fixed shoulder angle — arm descends toward the workspace |
| **1** — push | ≤ 600 | Shoulder sweeps continuously in −Y, arm pushes cube across the table; terminates early on success |

Only **successful** episodes (cube moved ≥ 6 cm in −Y) are saved. Failed attempts are discarded and retried automatically (~92% yield → ~108 attempts for 100 demos).

### Recorded data

Each `data/demos/episode_NNNN.hdf5` contains:

| Key | Shape | Type | Description |
|---|---|---|---|
| `observations/image` | `(T, 3, 224, 224)` | uint8 | Scene camera (world-fixed, front-right view) |
| `observations/wrist_image` | `(T, 3, 224, 224)` | uint8 | Right wrist camera |
| `observations/state` | `(T, 7)` | float32 | Right arm joint positions (qpos) |
| `actions` | `(T, 7)` | float32 | Actual velocity commands (`d.ctrl[RIGHT_CTRL]`), clipped to ctrlrange |
| `metadata/success` | scalar | bool | Always `True` (failures are not saved) |
| `metadata/episode_length` | scalar | int | T (≈ 761 steps typical) |
| `metadata/cube_start_pos` | `(3,)` | float32 | Cube XYZ at episode start |
| `metadata/language_instruction` | scalar | bytes | `b"move the cube to the right"` |

### Running

```bash
# Collect 100 successful demos, headless (fastest)
python record_demos.py --no-viewer --n-episodes 100

# Collect demos with the MuJoCo viewer open
python record_demos.py --n-episodes 5

# Quick test — 3 episodes, custom output directory
python record_demos.py --no-viewer --n-episodes 3 --out-dir data/test

# Visualise a single episode without saving
python visualize_demo.py
python visualize_demo.py --loop    # repeat until window closed
```

---

## Simulation Model

| Property | Value |
|---|---|
| Degrees of freedom | 15 (1 head + 7 right arm + 7 left arm) |
| Actuator type | Velocity-controlled; kv = 30 (shoulder/elbow), kv = 10 (wrist) |
| ctrlrange | ±3.0 rad/s (shoulder/elbow), ±4.0 rad/s (wrist) |
| Arm control in recorder | Dynamic (d.ctrl) with P + gravity-compensation feedforward |
| Cameras | `scene_camera` (world-fixed, 55° fovy), `right_hand_camera` (90° fovy), `left_hand_camera` (90° fovy) |
| Simulation timestep | 2 ms |
| Home keyframe | s1 = −0.96 rad, e1 = 2.0 rad, w1 = 0.785 rad (both arms) |
| Table height | z = 0.50 m (top surface) |
| Cube | 5 cm box, mass 50 g, free joint, spawned at ~(0.65, −0.20 ± 0.04, 0.525) m |
| Self-collision | Disabled (arm `contype=2, conaffinity=4`); arm–table also disabled |

---

## Next Steps

1. Convert HDF5 demos to LeRobot format
2. Write `BaxterInputs` / `BaxterOutputs` / `TrainConfig` for π0.5 fine-tuning
3. Compute normalisation statistics
4. Run LoRA fine-tuning on `pi05_base`
