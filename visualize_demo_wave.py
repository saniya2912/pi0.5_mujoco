"""Visualise the wave-your-hand task in the MuJoCo viewer (no data saved).

Usage
-----
  python visualize_demo_wave.py          # single episode
  python visualize_demo_wave.py --loop   # repeat until window closed
"""

import dataclasses
import mujoco
import mujoco.viewer
import numpy as np
import tyro

from record_demos_wave import (
    XML_PATH,
    HOME,
    WAVE_READY,
    WAVE_CENTER_S0,
    WAVE_AMP,
    WAVE_CYCLES,
    STEPS_PER_HALF,
    REACH_STEPS_A,
    REACH_STEPS_B,
    N_SUBSTEPS,
    reset_episode,
    joint_vel_toward,
    _apply_ctrl,
)


@dataclasses.dataclass
class Args:
    xml_path: str = XML_PATH
    loop: bool = False


def run_episode_viewer(m, d, viewer) -> None:
    reset_episode(m, d)

    # Phase 0a: s0 sweep
    print("  phase 0a: s0 sweep...")
    for t in range(REACH_STEPS_A):
        if not viewer.is_running():
            return
        alpha = (t + 1) / REACH_STEPS_A
        target = HOME.copy()
        target[0] = (1.0 - alpha) * HOME[0] + alpha * WAVE_CENTER_S0
        dq = joint_vel_toward(d, target)
        _apply_ctrl(m, d, dq)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(m, d)
        viewer.sync()

    # Phase 0b: raise arm to WAVE_READY
    print("  phase 0b: arm extension / raise...")
    for t in range(REACH_STEPS_B):
        if not viewer.is_running():
            return
        alpha = (t + 1) / REACH_STEPS_B
        target = WAVE_READY.copy()
        target[1:] = (1.0 - alpha) * HOME[1:] + alpha * WAVE_READY[1:]
        dq = joint_vel_toward(d, target)
        _apply_ctrl(m, d, dq)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(m, d)
        viewer.sync()

    # Phase 2: wave
    print(f"  phase 2: waving ({WAVE_CYCLES} cycles)...")
    for cycle in range(WAVE_CYCLES):
        for t in range(STEPS_PER_HALF):
            if not viewer.is_running():
                return
            alpha = (t + 1) / STEPS_PER_HALF
            target = WAVE_READY.copy()
            target[0] = WAVE_CENTER_S0 + alpha * WAVE_AMP
            dq = joint_vel_toward(d, target)
            _apply_ctrl(m, d, dq)
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(m, d)
            viewer.sync()

        for t in range(STEPS_PER_HALF * 2):
            if not viewer.is_running():
                return
            alpha = (t + 1) / (STEPS_PER_HALF * 2)
            target = WAVE_READY.copy()
            target[0] = WAVE_CENTER_S0 + (1.0 - alpha) * WAVE_AMP + alpha * (-WAVE_AMP)
            dq = joint_vel_toward(d, target)
            _apply_ctrl(m, d, dq)
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(m, d)
            viewer.sync()

        for t in range(STEPS_PER_HALF):
            if not viewer.is_running():
                return
            alpha = (t + 1) / STEPS_PER_HALF
            target = WAVE_READY.copy()
            target[0] = WAVE_CENTER_S0 + (1.0 - alpha) * (-WAVE_AMP)
            dq = joint_vel_toward(d, target)
            _apply_ctrl(m, d, dq)
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(m, d)
            viewer.sync()

    print("  done.")


def main(args: Args) -> None:
    m = mujoco.MjModel.from_xml_path(args.xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    ep = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            ep += 1
            print(f"\nEpisode {ep} — wave your hand")
            run_episode_viewer(m, d, viewer)
            if not viewer.is_running():
                break
            if not args.loop:
                print("\nDone. Close viewer to exit, or run with --loop to repeat.")
                while viewer.is_running():
                    viewer.sync()
                break


if __name__ == "__main__":
    main(tyro.cli(Args))
