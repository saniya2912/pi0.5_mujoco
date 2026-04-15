"""Visualise the push-block-to-front task in the MuJoCo viewer (no data saved).

Usage
-----
  python visualize_demo_push_front.py              # random cube position
  python visualize_demo_push_front.py --seed 7     # reproducible episode
  python visualize_demo_push_front.py --loop       # repeat until window closed
"""

import dataclasses
import mujoco
import mujoco.viewer
import numpy as np
import tyro

from record_demos_push_front import (
    XML_PATH,
    HOME,
    FRONT_S1, FRONT_E0, FRONT_W0, FRONT_W1, FRONT_W2,
    FRONT_E1_APPROACH,
    FRONT_E1_END,
    PUSH_RATE_E1,
    E1_Y_CORRECTION,
    SUCCESS_DIST,
    REACH_STEPS_A,
    REACH_STEPS_B,
    PUSH_STEPS,
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
    cube_start = reset_episode(m, d, rng)
    cube_y = cube_start[1]

    s0_cube = float(np.clip(s0_for_y(cube_y), -1.70, 1.70))
    approach_config = np.array([
        s0_cube, FRONT_S1, FRONT_E0, FRONT_E1_APPROACH,
        FRONT_W0, FRONT_W1, FRONT_W2
    ])

    print(f"  cube start: ({cube_start[0]:.3f}, {cube_start[1]:.3f}, {cube_start[2]:.3f})")
    print(f"  s0_cube={s0_cube:.3f}  e1: {FRONT_E1_APPROACH:.1f} → {FRONT_E1_END:.1f}")

    # Phase 0a: s0 sweep (arm folded, joints 1-6 at HOME)
    print("  phase 0a: s0 sweep (Y-align with cube)...")
    for t in range(REACH_STEPS_A):
        if not viewer.is_running():
            return None
        alpha = (t + 1) / REACH_STEPS_A
        target = HOME.copy()
        target[0] = (1.0 - alpha) * HOME[0] + alpha * s0_cube
        dq = joint_vel_toward(d, target)
        _apply_ctrl(m, d, dq)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(m, d)
        viewer.sync()

    # Phase 0b: extend arm with elbow bent (stays behind cube in X)
    print("  phase 0b: arm extension (elbow bent, arm behind cube)...")
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

    # Phase 1: decrease e1; s0 stays at s0_cube (E1_Y_CORRECTION=0) → push in +X
    print("  phase 1: push forward (+X)  [e1 extends arm, s0 constant]...")
    current_e1 = FRONT_E1_APPROACH
    for t in range(PUSH_STEPS):
        if not viewer.is_running():
            return None
        current_e1 = max(current_e1 + PUSH_RATE_E1, FRONT_E1_END)
        target = approach_config.copy()
        target[0] = s0_cube
        target[3] = current_e1
        dq = joint_vel_toward(d, target)
        _apply_ctrl(m, d, dq)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(m, d)
        viewer.sync()

        cube_now = get_cube_pos(m, d)
        if cube_now[0] - cube_start[0] >= SUCCESS_DIST:
            break
        if current_e1 <= FRONT_E1_END:
            break

    cube_final = get_cube_pos(m, d)
    success = (cube_final[0] - cube_start[0]) >= SUCCESS_DIST
    moved = cube_final[0] - cube_start[0]
    print(f"  {'SUCCESS' if success else 'FAIL'}  cube moved {moved*100:.1f} cm in +X")
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
            print(f"\nEpisode {ep} — push the block to the front")
            result = run_episode_viewer(m, d, rng, viewer)
            if result is None:
                print("Viewer closed.")
                break
            if not args.loop:
                print("\nDone. Close viewer to exit, or run with --loop to repeat.")
                while viewer.is_running():
                    viewer.sync()
                break


if __name__ == "__main__":
    main(tyro.cli(Args))
