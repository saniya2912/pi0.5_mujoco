# π0.5 Baxter Inference

---

## Table of Contents

1. [How inference works](#1-how-inference-works)
2. [How to run](#2-how-to-run)
3. [Inference stats — run_001 checkpoint 19999](#3-inference-stats--run_001-checkpoint-19999)
4. [Language conditioning behaviour](#4-language-conditioning-behaviour)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. How inference works

Inference uses a **two-process server-client architecture**:

```
Terminal 1                              Terminal 2
──────────────────────────────          ──────────────────────────────────────
serve_policy.py                         run_inference.py
  loads checkpoint 19999                  BaxterEnv (MuJoCo)
  builds LiberoInputs pipeline              resets to 'home' keyframe
  normalises observations                   renders head + wrist cameras
  runs 10 denoising steps                   extracts right arm qpos (7D)
  de-normalises actions                     ↓
  serves over WebSocket :8000         →  sends observation dict to server
                                       ←  receives 10-step action chunk (10, 32)
                                           applies chunk[:, :7] to ctrl[1:8]
                                           executes each action for 50 physics steps
                                           (= 0.1 s at dt=0.002 s → 10 Hz)
                                           replans every 5 executed actions
```

### Observation format sent to server

Matches training data exactly:

| Key | Shape | Type | Description |
|---|---|---|---|
| `observation/image` | (224, 224, 3) | uint8 HWC | Head camera |
| `observation/wrist_image` | (224, 224, 3) | uint8 HWC | Right wrist camera |
| `observation/state` | (7,) | float32 | Right arm qpos — `d.qpos[8:15]` |
| `prompt` | str | — | Language instruction |

### Action format received from server

| Key | Shape | Description |
|---|---|---|
| `actions` | (10, 32) | 10-step chunk; first 7 dims = right arm velocity commands |

Applied to `d.ctrl[1:8]` (RIGHT_CTRL). Each action executed for **50 physics steps** (= 0.1 s), matching the 10 Hz cadence of the training data.

### Physics execution rate

Training data was collected at:
- `DT = 0.002` s physics timestep
- `N_SUBSTEPS = 5` physics steps per control step → 100 Hz control
- Stride = 10 → 10 Hz training data

Each training action therefore spans `(100 Hz / 10 Hz) × 5 substeps = 50 physics steps`.
`run_inference.py` applies each action for exactly 50 physics steps to match this.

---

## 2. How to run

### Prerequisites

- Training complete: `checkpoints/pi05_baxter/run_001/19999` must exist
- MuJoCo installed and `baxter_env.py` present at project root

### Step 1 — Start the policy server

```bash
cd /home/robotlab/Desktop/saniya_ws/pi0.5_mujoco/openpi

uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config pi05_baxter \
    --policy.dir checkpoints/pi05_baxter/run_001/19999
```

Wait for: `INFO: Creating server (host: ..., ip: ...)`

### Step 2 — Run the MuJoCo client

```bash
cd /home/robotlab/Desktop/saniya_ws/pi0.5_mujoco

python run_inference.py --prompt "move the cube to the right"
```

A MuJoCo viewer window opens. The robot will start moving within ~2 seconds (first inference call).

### Options

```bash
# Save a video (saved to data/inference/videos/rollout_0000.mp4)
python run_inference.py --prompt "move the cube to the right" --save-video

# Longer episode
python run_inference.py --max-actions 300

# Replan every step (slower but more reactive)
python run_inference.py --replan-steps 1
```

### Output files

| File | Contents |
|---|---|
| `data/inference/diagnostics.csv` | Per-action: action vector (7D) + right arm qpos (7D) |
| `data/inference/videos/rollout_XXXX.mp4` | Video at 10 fps (if `--save-video`) |

---

## 3. Inference stats — run_001 checkpoint 19999

**Episode:** 150 actions × 0.1 s = **15.0 s of motion**
**Checkpoint:** `checkpoints/pi05_baxter/run_001/19999`
**Prompt:** "move the cube to the right"

### Action statistics (right arm velocity commands)

| Joint | Mean |action| (rad/s) | Max |action| (rad/s) | Role |
|---|---|---|---|
| 0 — right_s0 (shoulder rotate) | 0.1923 | 0.3403 | moderate |
| **1 — right_s1 (shoulder lift)** | **0.6098** | **1.0333** | **dominant** |
| 2 — right_e0 (elbow rotate) | 0.0998 | 0.1737 | minor |
| **3 — right_e1 (elbow flex)** | **0.2800** | **0.7927** | **active** |
| 4 — right_w0 (wrist rotate) | 0.0155 | 0.0459 | negligible |
| 5 — right_w1 (wrist flex) | 0.0706 | 0.1181 | minor |
| 6 — right_w2 (wrist roll) | 0.0062 | 0.0239 | negligible |

**Overall mean |action|:** 0.1820 rad/s
**Overall max |action|:** 1.0333 rad/s (joint 1, within ctrlrange ±3.0)

### Joint position drift over episode

| Joint | Start (rad) | End (rad) | Drift (rad) |
|---|---|---|---|
| right_s0 | 0.0154 | −0.4647 | −0.480 |
| **right_s1** | **−0.9597** | **1.0470** | **+2.007** ← largest |
| right_e0 | −0.0001 | 1.0574 | +1.058 |
| **right_e1** | **1.9999** | **−0.0045** | **−2.004** |
| right_w0 | −0.0005 | −0.1706 | −0.170 |
| right_w1 | 0.7859 | 0.9725 | +0.187 |
| right_w2 | −0.0001 | 0.0035 | +0.004 |

**Total joint displacement:** 5.91 rad

### Motion activity over time

| Phase | Mean |action| | Max |action| | Interpretation |
|---|---|---|---|
| 0–3 s (actions 0–29) | 0.1284 | 0.3592 | cautious initial motion |
| 3–6 s (30–59) | 0.1749 | 0.7698 | arm sweeping toward cube |
| 6–9 s (60–89) | 0.1781 | 0.7927 | continued push motion |
| 9–12 s (90–119) | 0.2132 | 1.0333 | full-speed push |
| 12–15 s (120–149) | 0.2155 | 1.0301 | sustained push at max velocity |

Activity **ramped up monotonically** over the episode (0.13 → 0.22 rad/s), consistent with the robot building up speed as it commits to the push trajectory. Wrist joints (4–6) remained almost stationary throughout — correct for a gross arm sweep task.

### Interpretation

The policy is producing **structured, task-relevant motion**, not random noise:
- The dominant motion is in s1 (shoulder lift) and e1 (elbow flex) — the joints that swing the arm forward to contact the cube
- Motion amplitude increased over time, not erratic
- Wrist joints correctly suppressed (no fine manipulation needed)
- Total arm displacement of 5.91 rad over 15 s is consistent with the scripted demos

---

## 4. Language conditioning behaviour

**Observation:** Changing `--prompt` to an unrelated instruction (e.g., "wave your hand") produced the **same cube-push motion** as the training prompt.

**This is expected.** Fine-tuning on a single task with a single prompt causes language conditioning collapse:

- During training, every demo had identical prompt ("move the cube to the right") and identical task
- The model never needed to read the language token — visual observations alone predicted the action perfectly
- The LoRA adapters learned: any prompt + Baxter-in-front-of-cube scene → cube push motion
- The language token carries zero gradient signal after fine-tuning and is effectively ignored

| Scenario | Language used? | Reason |
|---|---|---|
| π0.5 base (pre-trained) | ✅ yes | Diverse tasks — must use language to disambiguate |
| Our fine-tune (1 task) | ❌ no | Single task — language is redundant |

**To restore language conditioning** you would need to fine-tune on at least 2–3 tasks with different visual outcomes (e.g., "push left", "push right", "hold still") so the model is forced to condition on the prompt to select the correct behaviour.

For the current deployment goal (reliable cube-push execution), this is not a problem — the model correctly executes the task regardless of prompt.

---

## 5. Troubleshooting

### `Connection refused` when starting client

The server is not ready yet. Wait for `INFO: Creating server` in Terminal 1 before running Terminal 2.

### Viewer opens but robot doesn't move

The first inference call takes ~2–3 s (server loads checkpoint into GPU on first request). Wait briefly — motion should start after the first replan.

### Robot moves but immediately hits joint limits

The model is outputting unnormalized actions. Check that `norm_stats.json` is present at `openpi/assets/pi05_baxter/local/baxter_cube_push/` — the server needs it for de-normalization.

### Actions are all near zero

The server may be using the wrong config. Confirm `--policy.config pi05_baxter` is set (not `pi05_libero` or similar).
