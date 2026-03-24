"""Visualise a single demo episode in the MuJoCo viewer (no data saved).

Uses the same dynamic P-controller + gravity compensation as record_demos.py.

Usage
-----
  python visualize_demo.py              # random cube position
  python visualize_demo.py --seed 7     # reproducible episode
  python visualize_demo.py --loop       # repeat forever until window closed
"""

import dataclasses
import mujoco
import mujoco.viewer
import numpy as np
import tyro

from record_demos import (
    XML_PATH,
    HOME,
    APPROACH_BASE,
    CUBE_DEFAULT,
    CUBE_RAND_XY,
    REACH_STEPS_A,
    REACH_STEPS_B,
    PUSH_STEPS,
    PUSH_RATE,
    SUCCESS_DIST,
    BASE_S0, BASE_Y, DY_DS0,
    RIGHT_QPOS,
    N_SUBSTEPS,
    get_cube_pos,
    reset_episode,
    joint_vel_toward,
    _apply_ctrl,
    s0_for_y,
)


@dataclasses.dataclass
class Args:
    seed: int = 42
    xml_path: str = XML_PATH
    loop: bool = False


def run_episode_viewer(m, d, rng, viewer) -> bool | None:
    """Run one episode with viewer. Returns True/False on success/fail, None if viewer closed."""
    cube_start = reset_episode(m, d, rng)
    cube_y = cube_start[1]

    s0_approach = float(np.clip(s0_for_y(cube_y + 0.09), -1.70, 1.70))
    s0_push_end  = float(np.clip(s0_for_y(cube_y - 0.16), -1.70, 1.70))
    approach_config = APPROACH_BASE.copy()
    approach_config[0] = s0_approach

    print(f"  cube start: ({cube_start[0]:.3f}, {cube_start[1]:.3f}, {cube_start[2]:.3f})")
    print(f"  s0_approach={s0_approach:.3f}  s0_push_end={s0_push_end:.3f}")

    # Phase 0a: sweep s0 with arm folded
    print("  phase 0a: shoulder sweep...")
    for t in range(REACH_STEPS_A):
        if not viewer.is_running():
            return None
        alpha = (t + 1) / REACH_STEPS_A
        target = HOME.copy()
        target[0] = (1.0 - alpha) * HOME[0] + alpha * s0_approach
        dq = joint_vel_toward(d, target)
        _apply_ctrl(m, d, dq)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(m, d)
        viewer.sync()

    # Phase 0b: extend arm at fixed s0
    print("  phase 0b: arm extension...")
    for t in range(REACH_STEPS_B):
        if not viewer.is_running():
            return None
        alpha = (t + 1) / REACH_STEPS_B
        target = approach_config.copy()
        target[1:] = (1.0 - alpha) * HOME[1:] + alpha * approach_config[1:]
        dq = joint_vel_toward(d, target)
        _apply_ctrl(m, d, dq)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(m, d)
        viewer.sync()

    # Phase 1: push
    print("  phase 1: push...")
    current_s0 = s0_approach
    for t in range(PUSH_STEPS):
        if not viewer.is_running():
            return None
        current_s0 = max(current_s0 + PUSH_RATE, s0_push_end)
        target = approach_config.copy()
        target[0] = current_s0
        dq = joint_vel_toward(d, target)
        _apply_ctrl(m, d, dq)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(m, d)
        viewer.sync()

        cube_now = get_cube_pos(m, d)
        if cube_start[1] - cube_now[1] >= SUCCESS_DIST:
            break
        if current_s0 <= s0_push_end:
            break

    cube_final = get_cube_pos(m, d)
    success = (cube_start[1] - cube_final[1]) >= SUCCESS_DIST
    moved = cube_start[1] - cube_final[1]
    print(f"  {'SUCCESS' if success else 'FAIL'}  cube moved {moved*100:.1f} cm in -Y")
    return success


def main(args: Args) -> None:
    rng = np.random.default_rng(args.seed)
    m = mujoco.MjModel.from_xml_path(args.xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    ep = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            ep += 1
            print(f"\nEpisode {ep}")
            result = run_episode_viewer(m, d, rng, viewer)
            if result is None:
                print("Viewer closed.")
                break
            if not args.loop:
                print("\nDone. Close the viewer window to exit, or run with --loop to repeat.")
                while viewer.is_running():
                    viewer.sync()
                break


if __name__ == "__main__":
    main(tyro.cli(Args))
