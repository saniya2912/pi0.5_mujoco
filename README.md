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
baxter_env.py         — MuJoCo environment wrapper (reset / step / get_obs / render)
record_demos.py       — Scripted demo recorder (see below)
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

`record_demos.py` runs fully scripted, kinematic demonstrations of the right arm pushing a red cube in the −Y direction ("to the right" from Baxter's perspective).

**Task:** `"move the cube to the right"`
**Robot:** right arm only (7 DOF, joint-space kinematic control)
**Cube:** spawned at a randomised Y position (±4 cm), fixed X

### Episode structure

Each episode has three phases:

| Phase | Steps | What happens |
|---|---|---|
| **0a** — s0 sweep | 200 | Shoulder joint sweeps to align arm laterally while arm stays folded at home — endpoint never approaches the cube |
| **0b** — arm extension | 300 | Joints 1–6 interpolate from home to approach config at fixed shoulder angle — arm descends toward the workspace |
| **1** — push | ≤ 600 | Shoulder sweeps continuously in −Y, arm pushes cube across the table; terminates early on success |

Only **successful** episodes (cube moved ≥ 6 cm in −Y) are saved. Failed attempts are discarded and retried automatically.

### Recorded data

Each `data/demos/episode_NNNN.hdf5` contains:

| Key | Shape | Type | Description |
|---|---|---|---|
| `observations/image` | `(T, 3, 224, 224)` | uint8 | Scene camera (world-fixed, front-right view) |
| `observations/wrist_image` | `(T, 3, 224, 224)` | uint8 | Right wrist camera |
| `observations/state` | `(T, 7)` | float32 | Right arm joint positions (qpos) |
| `actions` | `(T, 7)` | float32 | Implied joint velocities (Δqpos / dt), clipped to ±4 rad/s |
| `metadata/success` | scalar | bool | Always `True` (failures are not saved) |
| `metadata/episode_length` | scalar | int | T |
| `metadata/cube_start_pos` | `(3,)` | float32 | Cube XYZ at episode start |
| `metadata/language_instruction` | scalar | bytes | `b"move the cube to the right"` |

### Running

```bash
# Collect 100 successful demos, headless (fastest)
python record_demos.py --no-viewer --n-episodes 100

# Collect 5 demos with the MuJoCo viewer open
python record_demos.py --n-episodes 5

# Quick test — 3 episodes, custom output directory
python record_demos.py --no-viewer --n-episodes 3 --out-dir data/test
```

The recorder will run as many attempts as needed until `--n-episodes` successes are saved (~77% yield → ~130 attempts for 100 demos at default settings).

---

## Simulation Model

| Property | Value |
|---|---|
| Degrees of freedom | 15 (1 head + 7 right arm + 7 left arm) |
| Actuator type | Velocity-controlled (kv = 10 shoulder/elbow, kv = 5 wrist) |
| Arm control in recorder | Kinematic (direct qpos) — bypasses weak velocity actuators |
| Cameras | `scene_camera` (world-fixed, 55° fovy), `right_hand_camera` (90° fovy), `left_hand_camera` (90° fovy) |
| Simulation timestep | 2 ms |
| Home keyframe | s1 = −0.96 rad, e1 = 2.0 rad, w1 = 0.785 rad (both arms) |
| Table height | z = 0.50 m (top surface) |
| Cube | 5 cm box, mass 50 g, free joint, spawned at ~(0.65, −0.20 ± 0.04, 0.52) m |

---

## Why Kinematic Control?

Baxter's velocity actuators (kv = 10) produce a maximum torque of ~15 Nm.
The gravity torque on the s1 joint with the arm extended is ~31 Nm — the arm collapses regardless of control input.
The demo recorder bypasses this by setting `d.qpos` directly each step, giving perfectly smooth, reproducible trajectories while keeping the cube fully dynamic so it responds to real contact forces.

---

## Next Steps

1. Convert HDF5 demos to LeRobot format
2. Write `BaxterInputs` / `BaxterOutputs` / `TrainConfig` for π0.5 fine-tuning
3. Compute normalisation statistics
4. Run LoRA fine-tuning on `pi05_base`
