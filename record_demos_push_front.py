"""Scripted demonstration recorder for Baxter right-arm cube push — forward (+X).

The arm pushes the cube away from the robot base along the +X axis ("to the front").

Arm configuration (e0=0.0)
--------------------------
  With e0=0.0 the elbow bend (e1) unfolds primarily in the X-Z plane.
  As e1 decreases from FRONT_E1_APPROACH → FRONT_E1_END, the hand sweeps
  forward in +X and slightly downward in -Z, contacting the cube's back face
  and pushing it in +X.

  With e0=0.0, the e1-induced Y-shift is tiny (~0.008 m/rad), so no s0
  correction for e1 is needed (E1_Y_CORRECTION ≈ 0).

  FRONT_S1 = 1.10 lowers the arm endpoint to z ≈ 0.520 at e1=0 (cube midpoint
  z = 0.525), ensuring the hand contacts the cube's back face rather than
  sliding over the top.

  FRONT_E1_END = −0.2 targets beyond the joint limit (−0.05 rad), causing the
  arm to press against the limit and sustain contact force throughout the push.

Phase plan
----------
  Phase 0a  s0 sweep (arm folded) : HOME → s0_cube                        300 steps
  Phase 0b  arm extension         : joints 1-6  HOME → approach config     400 steps
                                    (e1=1.0, arm retracted behind cube)
  Phase 1   push forward (+X)     : e1 decreases 1.0 → FRONT_E1_END       ~600 steps
                                    s0 held at s0_cube (Y-correction ≈ 0)
                                    → arm contacts cube back face, pushes in +X

Language instruction: "push the block to the front"

Output
------
  data/demos_push_front/episode_NNNN.hdf5

Usage
-----
  python record_demos_push_front.py                     # 100 episodes with viewer
  python record_demos_push_front.py --no-viewer         # headless (faster)
  python record_demos_push_front.py --n-episodes 5      # quick test
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
OUT_DIR   = pathlib.Path("data/demos_push_front")
LANGUAGE  = "push the block to the front"
IMG_H, IMG_W = 224, 224

RIGHT_QPOS = slice(8, 15)   # [s0,s1,e0,e1,w0,w1,w2] in d.qpos
RIGHT_CTRL = slice(1, 8)

LEFT_QPOS  = slice(15, 22)
LEFT_CTRL  = slice(8, 15)
KV_LEFT    = np.array([30., 30., 30., 30., 10., 10., 10.])

DT = 0.002

HOME = np.array([0.0, -0.9599, 0.0, 2.0, 0.0, 0.7854, 0.0])

# ── FK: s0 → endpoint Y at e1=0 (with e0=0.0, s1=1.10) ─────────────────────
BASE_S0 =  0.886
BASE_Y  = -0.111
DY_DS0  =  0.593

# ── Y-alignment correction for elbow bend ────────────────────────────────────
# With e0=0.0, the e1-induced Y-shift is only ~0.008 m/rad (negligible).
# No s0 correction is needed; s0 stays at s0_cube throughout the push.
E1_Y_CORRECTION = 0.0

# ── e1 range for forward (+X) push ──────────────────────────────────────────
FRONT_E1_APPROACH = 1.0          # elbow bent:   arm retracted behind cube in X
FRONT_E1_END      = -0.2         # target past joint limit (−0.05 rad) to sustain
                                  # contact force at the limit throughout the push
PUSH_RATE_E1      = -0.003       # e1 change per control step

# Other joints fixed during approach and push
# e0=0.0  → elbow bend unfolds in X-Z plane (gives +X push direction)
# s1=1.10 → lowers arm endpoint to z≈0.520 at e1=0 (cube midpoint z=0.525)
FRONT_S1  = 1.10
FRONT_E0  = 0.0
FRONT_W0  = 0.0
FRONT_W1  = 0.929
FRONT_W2  = 0.0

# Cube
CUBE_DEFAULT = np.array([0.65, -0.20, 0.525])
CUBE_RAND_Y  = 0.04    # ±4 cm Y only

N_SUBSTEPS = 5
CTRL_GAIN  = 6.0
NOISE_STD  = 0.01
KV_RIGHT   = np.array([30., 30., 30., 30., 10., 10., 10.])

REACH_STEPS_A = 300
REACH_STEPS_B = 400
PUSH_STEPS    = 600

SUCCESS_DIST  = 0.045  # cube must move +4.5 cm in +X


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def s0_for_y(target_y: float) -> float:
    """s0 that Y-aligns arm endpoint at e1=0 with target_y."""
    return BASE_S0 + (target_y - BASE_Y) / DY_DS0


def get_cube_pos(m, d) -> np.ndarray:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
    return d.xpos[bid].copy()


def joint_vel_toward(d, target_qpos: np.ndarray, gain: float = CTRL_GAIN) -> np.ndarray:
    feedback    = gain * (target_qpos - d.qpos[RIGHT_QPOS])
    feedforward = d.qfrc_bias[7:14] / KV_RIGHT
    return feedback + feedforward


def _apply_ctrl(m, d, dq: np.ndarray) -> np.ndarray:
    ctrl_min = m.actuator_ctrlrange[RIGHT_CTRL, 0]
    ctrl_max = m.actuator_ctrlrange[RIGHT_CTRL, 1]
    dq = dq + np.random.normal(0, NOISE_STD, size=7)
    dq_clipped = np.clip(dq, ctrl_min, ctrl_max)

    left_fb = CTRL_GAIN * (HOME - d.qpos[LEFT_QPOS])
    left_ff = d.qfrc_bias[14:21] / KV_LEFT
    left_ctrl_min = m.actuator_ctrlrange[LEFT_CTRL, 0]
    left_ctrl_max = m.actuator_ctrlrange[LEFT_CTRL, 1]
    left_dq = np.clip(left_fb + left_ff, left_ctrl_min, left_ctrl_max)

    d.ctrl[:] = 0.0
    d.ctrl[RIGHT_CTRL] = dq_clipped
    d.ctrl[LEFT_CTRL]  = left_dq
    return dq_clipped


def _record(renderer, d, implied_vel, imgs, wrists, states, actions):
    renderer.update_scene(d, camera="scene_camera")
    imgs.append(np.transpose(renderer.render(), (2, 0, 1)))
    renderer.update_scene(d, camera="right_hand_camera")
    wrists.append(np.transpose(renderer.render(), (2, 0, 1)))
    states.append(d.qpos[RIGHT_QPOS].astype(np.float32))
    actions.append(np.clip(implied_vel, -4.0, 4.0).astype(np.float32))


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def reset_episode(m, d, rng) -> np.ndarray:
    mujoco.mj_resetDataKeyframe(m, d, 0)
    dy = rng.uniform(-CUBE_RAND_Y, CUBE_RAND_Y)
    d.qpos[0] = CUBE_DEFAULT[0]
    d.qpos[1] = CUBE_DEFAULT[1] + dy
    d.qpos[2] = CUBE_DEFAULT[2]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    for _ in range(60):
        dq = joint_vel_toward(d, HOME)
        _apply_ctrl(m, d, dq)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(m, d)

    return get_cube_pos(m, d).copy()


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def run_episode(m, d, rng, renderer, viewer=None) -> dict | None:
    cube_start = reset_episode(m, d, rng)
    cube_y = cube_start[1]

    # s0 that Y-aligns endpoint with cube (E1_Y_CORRECTION=0 so no bend correction)
    s0_cube = float(np.clip(s0_for_y(cube_y), -1.70, 1.70))
    # 7-DOF approach config: arm retracted behind cube in X with e1=1.0
    approach_config = np.array([
        s0_cube, FRONT_S1, FRONT_E0, FRONT_E1_APPROACH,
        FRONT_W0, FRONT_W1, FRONT_W2
    ])

    obs_imgs, obs_wrists, obs_states, obs_actions = [], [], [], []

    # ── Phase 0a: s0 sweep (arm folded, joints 1-6 at HOME) ──────────────
    for t in range(REACH_STEPS_A):
        if viewer is not None and not viewer.is_running():
            return None

        alpha = (t + 1) / REACH_STEPS_A
        target = HOME.copy()
        target[0] = (1.0 - alpha) * HOME[0] + alpha * s0_cube

        dq = joint_vel_toward(d, target)
        ctrl = _apply_ctrl(m, d, dq)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(m, d)
        if viewer is not None:
            viewer.sync()
        _record(renderer, d, ctrl, obs_imgs, obs_wrists, obs_states, obs_actions)

    # ── Phase 0b: extend arm to approach config (e1=1.0, behind cube) ────
    for t in range(REACH_STEPS_B):
        if viewer is not None and not viewer.is_running():
            return None

        alpha = (t + 1) / REACH_STEPS_B
        target = approach_config.copy()
        target[1:] = (1.0 - alpha) * HOME[1:] + alpha * approach_config[1:]

        dq = joint_vel_toward(d, target)
        ctrl = _apply_ctrl(m, d, dq)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(m, d)
        if viewer is not None:
            viewer.sync()
        _record(renderer, d, ctrl, obs_imgs, obs_wrists, obs_states, obs_actions)

    # ── Phase 1: push forward ─────────────────────────────────────────────
    # e1 decreases toward FRONT_E1_END (past joint limit −0.05).
    # With e0=0.0, the elbow unfolds in the X-Z plane: hand sweeps +X and
    # contacts the cube back face.  s0 stays at s0_cube (Y drift ≈ 0).
    current_e1 = FRONT_E1_APPROACH

    for t in range(PUSH_STEPS):
        if viewer is not None and not viewer.is_running():
            return None

        current_e1 = max(current_e1 + PUSH_RATE_E1, FRONT_E1_END)
        target = approach_config.copy()
        target[0] = s0_cube   # s0 constant (E1_Y_CORRECTION = 0)
        target[3] = current_e1

        dq = joint_vel_toward(d, target)
        ctrl = _apply_ctrl(m, d, dq)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(m, d)
        if viewer is not None:
            viewer.sync()
        _record(renderer, d, ctrl, obs_imgs, obs_wrists, obs_states, obs_actions)

        cube_now = get_cube_pos(m, d)
        if cube_now[0] - cube_start[0] >= SUCCESS_DIST:
            break
        if current_e1 <= FRONT_E1_END:
            break

    T = len(obs_imgs)
    cube_final = get_cube_pos(m, d)
    success = (cube_final[0] - cube_start[0]) >= SUCCESS_DIST

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
            ok  = result["metadata/success"]
            T   = result["metadata/episode_length"]
            pos = result["metadata/cube_start_pos"]
            if ok:
                n_success += 1
                path = args.out_dir / f"episode_{n_success-1:04d}.hdf5"
                save_hdf5(result, path)
                print(f"T={T:4d}  ✓  cube_x={pos[0]:.3f}  ({n_success}/{args.n_episodes} saved)")
            else:
                print(f"T={T:4d}  ✗  cube_x={pos[0]:.3f}  (skip)")

    if args.viewer:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            _run_all(viewer)
    else:
        _run_all()

    renderer.close()
    print(f"\nDone. {n_success} successful demos saved ({n_attempts} attempts) → {args.out_dir}/")


if __name__ == "__main__":
    main(tyro.cli(Args))
