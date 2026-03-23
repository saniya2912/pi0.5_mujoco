"""Scripted demonstration recorder for Baxter right-arm cube pushing.

Controller strategy
-------------------
The right arm is driven **kinematically** (qpos set directly each step).
This bypasses gravity issues with the weak velocity actuators and gives
perfectly smooth, reproducible trajectories.

The cube is fully dynamic — it responds to contact forces from the arm.

Pipeline per episode
--------------------
  Phase 0a  s0 sweep (arm folded) : move s0 from home→approach while joints 1-6
                                    stay at HOME values; endpoint stays far from cube
  Phase 0b  arm extension         : interpolate joints 1-6 from HOME→APPROACH_BASE
                                    at fixed s0_approach; arm descends toward workspace
  Phase 1   push                  : sweep s0 from approach → past cube in -Y

Recorded actions = implied joint velocities (Δqpos/dt), clipped to ctrlrange.

Language instruction: "move the cube to the right"

Output
------
  data/demos/episode_NNNN.hdf5

Each HDF5 contains:
  observations/image        (T, 3, 224, 224) uint8
  observations/wrist_image  (T, 3, 224, 224) uint8
  observations/state        (T, 7)           float32  right arm qpos
  actions                   (T, 7)           float32  right arm Δqpos/dt
  metadata/success, episode_length, cube_start_pos, language_instruction

Usage
-----
  python record_demos.py                     # 100 episodes with viewer
  python record_demos.py --no-viewer         # headless (faster)
  python record_demos.py --n-episodes 5      # quick test
"""

import dataclasses
import pathlib

import h5py
import mujoco
import mujoco.viewer
import numpy as np
import tyro

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

XML_PATH  = "models/baxter.xml"
OUT_DIR   = pathlib.Path("data/demos")
LANGUAGE  = "move the cube to the right"
IMG_H, IMG_W = 224, 224

# Right arm slices (verified from joint ordering with free-joint cube)
RIGHT_QPOS = slice(8, 15)   # in d.qpos
RIGHT_CTRL = slice(1, 8)    # in d.ctrl

DT = 0.002   # simulation timestep

# ── Keyframe configs (right arm: s0 s1 e0 e1 w0 w1 w2) ──────────────────────
HOME = np.array([0.0,   -0.9599, 0.0, 2.0,  0.0, 0.7854, 0.0])

# Approach config: arm endpoint at ~(0.644, -0.111, 0.543)
# s0 controls Y sweep; other joints set height and orientation
APPROACH_BASE = np.array([0.886, 1.04, 0.5, 0.0, 0.0, 0.929, 0.0])

# FK-derived mapping: ep_y ≈ BASE_Y + DY_DS0*(s0 - BASE_S0)
BASE_S0  =  0.886
BASE_Y   = -0.111   # ep_y at BASE_S0 with other joints = APPROACH_BASE[1:]
DY_DS0   =  0.593   # ∂ep_y/∂s0 (from FK scan, constant while other joints fixed)

# Cube defaults
CUBE_DEFAULT = np.array([0.65, -0.20, 0.55])
CUBE_RAND_XY = 0.04   # ±4 cm in Y only (X fixed — arm aligns with cube in X)

# Episode timing
REACH_STEPS_A = 200   # phase 0a: sweep s0 with arm folded (HOME joints 1-6)
REACH_STEPS_B = 300   # phase 0b: extend arm at fixed s0_approach
PUSH_STEPS    = 600   # max steps for push sweep (phase 1)
MAX_STEPS     = REACH_STEPS_A + REACH_STEPS_B + PUSH_STEPS

PUSH_RATE     = -0.003  # Δs0 per step during push (≈ -1.5 rad/s at dt=0.002)

SUCCESS_DIST  = 0.06   # m  cube must move in -Y to count as success


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def s0_for_y(target_y: float) -> float:
    """Return s0 that puts endpoint at target_y (other joints at APPROACH_BASE)."""
    return BASE_S0 + (target_y - BASE_Y) / DY_DS0


def get_cube_pos(m, d) -> np.ndarray:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
    return d.xpos[bid].copy()


def set_arm_kinematic(d, qpos_right: np.ndarray) -> None:
    """Directly set right arm qpos and zero its velocity (kinematic step)."""
    d.qpos[RIGHT_QPOS] = qpos_right
    d.qvel[7:14] = 0.0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def reset_episode(m, d, rng) -> np.ndarray:
    """Reset to home keyframe + randomise cube Y only. Returns settled cube pos."""
    mujoco.mj_resetDataKeyframe(m, d, 0)
    dy = rng.uniform(-CUBE_RAND_XY, CUBE_RAND_XY)
    d.qpos[0] = CUBE_DEFAULT[0]           # X fixed — arm aligns in X at APPROACH_BASE
    d.qpos[1] = CUBE_DEFAULT[1] + dy
    d.qpos[2] = CUBE_DEFAULT[2]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    # Let cube settle on table (arm stays at home kinematically)
    for _ in range(60):
        set_arm_kinematic(d, HOME)
        mujoco.mj_step(m, d)

    return get_cube_pos(m, d).copy()


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def run_episode(m, d, rng, renderer, viewer=None) -> dict | None:
    cube_start = reset_episode(m, d, rng)

    # Compute s0 targets based on cube's actual Y position
    cube_y = cube_start[1]
    s0_approach = s0_for_y(cube_y + 0.09)   # 9 cm +Y of cube face
    s0_push_end = s0_for_y(cube_y - 0.16)   # push 16 cm through cube in -Y

    # Clamp to joint limit
    s0_approach = float(np.clip(s0_approach, -1.70, 1.70))
    s0_push_end  = float(np.clip(s0_push_end,  -1.70, 1.70))

    approach_config = APPROACH_BASE.copy()
    approach_config[0] = s0_approach

    # ── Phase 0a: sweep s0 with arm folded (joints 1-6 stay at HOME) ─────────
    # Endpoint stays at x ≈ -0.05 to -0.16, far from cube at x ≈ 0.65
    obs_imgs, obs_wrists, obs_states, obs_actions = [], [], [], []

    prev_qpos = HOME.copy()
    for t in range(REACH_STEPS_A):
        if viewer is not None and not viewer.is_running():
            return None

        alpha = (t + 1) / REACH_STEPS_A
        target = HOME.copy()
        target[0] = (1.0 - alpha) * HOME[0] + alpha * s0_approach
        set_arm_kinematic(d, target)
        mujoco.mj_step(m, d)
        if viewer is not None:
            viewer.sync()

        implied_vel = (target - prev_qpos) / DT
        _record(renderer, d, implied_vel, obs_imgs, obs_wrists, obs_states, obs_actions)
        prev_qpos = target.copy()

    # ── Phase 0b: extend arm at fixed s0_approach (HOME[1:] → APPROACH_BASE[1:]) ──
    # Arm descends toward workspace; stays >8 cm from cube in XY until final position
    for t in range(REACH_STEPS_B):
        if viewer is not None and not viewer.is_running():
            return None

        alpha = (t + 1) / REACH_STEPS_B
        target = approach_config.copy()
        # Interpolate joints 1-6 from HOME to APPROACH_BASE; s0 is fixed at s0_approach
        target[1:] = (1.0 - alpha) * HOME[1:] + alpha * approach_config[1:]
        set_arm_kinematic(d, target)
        mujoco.mj_step(m, d)
        if viewer is not None:
            viewer.sync()

        implied_vel = (target - prev_qpos) / DT
        _record(renderer, d, implied_vel, obs_imgs, obs_wrists, obs_states, obs_actions)
        prev_qpos = target.copy()

    # ── Phase 1: sweep s0 from approach → push end ───────────────────────────
    current_s0 = s0_approach
    s0_step = PUSH_RATE   # Δs0 per step (constant rate)

    for t in range(PUSH_STEPS):
        if viewer is not None and not viewer.is_running():
            return None

        current_s0 = max(current_s0 + s0_step, s0_push_end)
        target = approach_config.copy()
        target[0] = current_s0
        set_arm_kinematic(d, target)
        mujoco.mj_step(m, d)
        if viewer is not None:
            viewer.sync()

        implied_vel = np.zeros(7)
        implied_vel[0] = s0_step / DT   # only s0 moves
        _record(renderer, d, implied_vel, obs_imgs, obs_wrists, obs_states, obs_actions)
        prev_qpos = target.copy()

        # Early success
        cube_now = get_cube_pos(m, d)
        y_moved = cube_start[1] - cube_now[1]
        if y_moved >= SUCCESS_DIST:
            break

        if current_s0 <= s0_push_end:
            break

    T = len(obs_imgs)
    cube_final = get_cube_pos(m, d)
    success = (cube_start[1] - cube_final[1]) >= SUCCESS_DIST

    return {
        "observations/image":       np.stack(obs_imgs),
        "observations/wrist_image": np.stack(obs_wrists),
        "observations/state":       np.stack(obs_states),
        "actions":                  np.stack(obs_actions),
        "metadata/success":         success,
        "metadata/episode_length":  T,
        "metadata/cube_start_pos":  cube_start.astype(np.float32),
        "metadata/language_instruction": LANGUAGE.encode(),
    }


def _record(renderer, d, implied_vel, imgs, wrists, states, actions):
    renderer.update_scene(d, camera="scene_camera")
    imgs.append(np.transpose(renderer.render(), (2, 0, 1)))
    renderer.update_scene(d, camera="right_hand_camera")
    wrists.append(np.transpose(renderer.render(), (2, 0, 1)))
    states.append(d.qpos[RIGHT_QPOS].astype(np.float32))
    actions.append(np.clip(implied_vel, -4.0, 4.0).astype(np.float32))


# ---------------------------------------------------------------------------
# HDF5 save
# ---------------------------------------------------------------------------

def save_hdf5(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, val in data.items():
            arr = np.asarray(val)
            if arr.ndim == 0:
                f.create_dataset(key, data=val)
            else:
                f.create_dataset(key, data=val, compression="gzip", compression_opts=4)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    n_episodes: int = 100
    viewer: bool = True
    seed: int = 42
    xml_path: str = XML_PATH
    out_dir: pathlib.Path = OUT_DIR


def main(args: Args) -> None:
    rng = np.random.default_rng(args.seed)
    m   = mujoco.MjModel.from_xml_path(args.xml_path)
    d   = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, IMG_H, IMG_W)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    n_success = 0
    n_attempts = 0

    def _run_all(viewer=None):
        nonlocal n_success, n_attempts
        while n_success < args.n_episodes:
            n_attempts += 1
            print(f"[{n_success+1:3d}/{args.n_episodes}] attempt {n_attempts}", end=" ", flush=True)
            result = run_episode(m, d, rng, renderer, viewer)
            if result is None:
                print("viewer closed — stopping.")
                return
            ok = result["metadata/success"]
            T   = result["metadata/episode_length"]
            pos = result["metadata/cube_start_pos"]
            if ok:
                n_success += 1
                path = args.out_dir / f"episode_{n_success-1:04d}.hdf5"
                save_hdf5(result, path)
                print(f"T={T:4d}  ✓  cube_y={pos[1]:.3f}  ({n_success}/{args.n_episodes} saved)")
            else:
                print(f"T={T:4d}  ✗  cube_y={pos[1]:.3f}  (skip)")

    if args.viewer:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            _run_all(viewer)
    else:
        _run_all()

    renderer.close()
    print(f"\nDone. {n_success} successful demos saved ({n_attempts} attempts) → {args.out_dir}/")


if __name__ == "__main__":
    main(tyro.cli(Args))
