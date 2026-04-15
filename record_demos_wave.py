"""Scripted demonstration recorder for Baxter right-arm waving.

Controller strategy
-------------------
Same proportional velocity + gravity compensation as record_demos.py.

Motion plan  (mirrors push task phase structure)
-----------
  Phase 0a  s0 sweep (arm folded) : HOME → s0_wave_center               300 steps
  Phase 0b  arm extension         : joints 1-6  HOME → WAVE_READY        400 steps
  Phase 2   wave                  : s0 oscillates ±WAVE_AMP around
                                    s0_wave_center for WAVE_CYCLES cycles 600 steps

WAVE_READY raises the arm above the table so there is no cube interaction.
The cube stays at its default keyframe position throughout.
All episodes succeed — there is no success criterion.

Language instruction: "wave your hand"

Output
------
  data/demos_wave/episode_NNNN.hdf5

Each HDF5 contains:
  observations/image        (T, 3, 224, 224) uint8
  observations/wrist_image  (T, 3, 224, 224) uint8
  observations/state        (T, 7)           float32  right arm qpos
  actions                   (T, 7)           float32  velocity commands (d.ctrl)
  metadata/success, episode_length, language_instruction

Usage
-----
  python record_demos_wave.py                     # 100 episodes with viewer
  python record_demos_wave.py --no-viewer         # headless (faster)
  python record_demos_wave.py --n-episodes 5      # quick test
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
OUT_DIR   = pathlib.Path("data/demos_wave")
LANGUAGE  = "wave your hand"
IMG_H, IMG_W = 224, 224

RIGHT_QPOS = slice(8, 15)
RIGHT_CTRL = slice(1, 8)

LEFT_QPOS  = slice(15, 22)
LEFT_CTRL  = slice(8, 15)
KV_LEFT    = np.array([30., 30., 30., 30., 10., 10., 10.])

DT = 0.002

HOME = np.array([0.0, -0.9599, 0.0, 2.0, 0.0, 0.7854, 0.0])

# Wave centre s0: arm positioned forward-centre for waving
WAVE_CENTER_S0 = 0.3

# Wave-ready pose: arm extended and raised above table level.
# s1=-0.2 (arm above horizontal), e1=1.0 (elbow bent, forearm angled up).
# The arm clears the table so there is no cube contact.
WAVE_READY = np.array([WAVE_CENTER_S0, -0.2, 0.3, 1.0, 0.0, 0.4, 0.0])

WAVE_AMP       = 0.40   # s0 amplitude (rad) around WAVE_CENTER_S0
WAVE_CYCLES    = 3
STEPS_PER_HALF = 100    # control steps per half-cycle (centre → peak)

N_SUBSTEPS = 5
CTRL_GAIN  = 6.0
NOISE_STD  = 0.01
KV_RIGHT   = np.array([30., 30., 30., 30., 10., 10., 10.])

REACH_STEPS_A = 300
REACH_STEPS_B = 400


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def reset_episode(m, d) -> None:
    """Reset to home keyframe. Cube stays at keyframe position."""
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    for _ in range(60):
        dq = joint_vel_toward(d, HOME)
        _apply_ctrl(m, d, dq)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(m, d)


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def run_episode(m, d, renderer, viewer=None) -> dict | None:
    reset_episode(m, d)

    obs_imgs, obs_wrists, obs_states, obs_actions = [], [], [], []

    # ── Phase 0a: sweep s0 from HOME → WAVE_CENTER_S0 (arm still folded) ─
    for t in range(REACH_STEPS_A):
        if viewer is not None and not viewer.is_running():
            return None

        alpha = (t + 1) / REACH_STEPS_A
        target = HOME.copy()
        target[0] = (1.0 - alpha) * HOME[0] + alpha * WAVE_CENTER_S0

        dq = joint_vel_toward(d, target)
        ctrl = _apply_ctrl(m, d, dq)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(m, d)
        if viewer is not None:
            viewer.sync()

        _record(renderer, d, ctrl, obs_imgs, obs_wrists, obs_states, obs_actions)

    # ── Phase 0b: extend/raise arm from HOME joints → WAVE_READY joints ──
    for t in range(REACH_STEPS_B):
        if viewer is not None and not viewer.is_running():
            return None

        alpha = (t + 1) / REACH_STEPS_B
        target = WAVE_READY.copy()
        target[1:] = (1.0 - alpha) * HOME[1:] + alpha * WAVE_READY[1:]

        dq = joint_vel_toward(d, target)
        ctrl = _apply_ctrl(m, d, dq)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(m, d)
        if viewer is not None:
            viewer.sync()

        _record(renderer, d, ctrl, obs_imgs, obs_wrists, obs_states, obs_actions)

    # ── Phase 2: wave — oscillate s0 around WAVE_CENTER_S0 ───────────────
    for cycle in range(WAVE_CYCLES):
        # centre → +peak
        for t in range(STEPS_PER_HALF):
            if viewer is not None and not viewer.is_running():
                return None
            alpha = (t + 1) / STEPS_PER_HALF
            target = WAVE_READY.copy()
            target[0] = WAVE_CENTER_S0 + alpha * WAVE_AMP
            dq = joint_vel_toward(d, target)
            ctrl = _apply_ctrl(m, d, dq)
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(m, d)
            if viewer is not None:
                viewer.sync()
            _record(renderer, d, ctrl, obs_imgs, obs_wrists, obs_states, obs_actions)

        # +peak → -peak
        for t in range(STEPS_PER_HALF * 2):
            if viewer is not None and not viewer.is_running():
                return None
            alpha = (t + 1) / (STEPS_PER_HALF * 2)
            target = WAVE_READY.copy()
            target[0] = WAVE_CENTER_S0 + (1.0 - alpha) * WAVE_AMP + alpha * (-WAVE_AMP)
            dq = joint_vel_toward(d, target)
            ctrl = _apply_ctrl(m, d, dq)
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(m, d)
            if viewer is not None:
                viewer.sync()
            _record(renderer, d, ctrl, obs_imgs, obs_wrists, obs_states, obs_actions)

        # -peak → centre
        for t in range(STEPS_PER_HALF):
            if viewer is not None and not viewer.is_running():
                return None
            alpha = (t + 1) / STEPS_PER_HALF
            target = WAVE_READY.copy()
            target[0] = WAVE_CENTER_S0 + (1.0 - alpha) * (-WAVE_AMP)
            dq = joint_vel_toward(d, target)
            ctrl = _apply_ctrl(m, d, dq)
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(m, d)
            if viewer is not None:
                viewer.sync()
            _record(renderer, d, ctrl, obs_imgs, obs_wrists, obs_states, obs_actions)

    T = len(obs_imgs)
    return {
        "observations/image":       np.stack(obs_imgs),
        "observations/wrist_image": np.stack(obs_wrists),
        "observations/state":       np.stack(obs_states),
        "actions":                  np.stack(obs_actions),
        "metadata/success":         True,
        "metadata/episode_length":  T,
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
    n_episodes: int = 10
    viewer: bool = True
    xml_path: str = XML_PATH
    out_dir: pathlib.Path = OUT_DIR


def main(args: Args) -> None:
    m   = mujoco.MjModel.from_xml_path(args.xml_path)
    d   = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, IMG_H, IMG_W)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    n_saved = 0

    def _run_all(viewer=None):
        nonlocal n_saved
        while n_saved < args.n_episodes:
            print(f"[{n_saved+1:3d}/{args.n_episodes}]", end=" ", flush=True)
            result = run_episode(m, d, renderer, viewer)
            if result is None:
                print("viewer closed — stopping.")
                return
            T = result["metadata/episode_length"]
            n_saved += 1
            path = args.out_dir / f"episode_{n_saved-1:04d}.hdf5"
            save_hdf5(result, path)
            print(f"T={T:4d}  ✓  ({n_saved}/{args.n_episodes} saved)")

    if args.viewer:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            _run_all(viewer)
    else:
        _run_all()

    renderer.close()
    print(f"\nDone. {n_saved} demos saved → {args.out_dir}/")


if __name__ == "__main__":
    main(tyro.cli(Args))
