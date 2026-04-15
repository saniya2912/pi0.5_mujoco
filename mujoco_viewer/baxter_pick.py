#!/usr/bin/env python3
"""
Scripted pick demo for Baxter with Rethink parallel-jaw grippers.

  Phase 0 — settle + open right gripper
  Phase 1 — joint-space P control → Q_REACH
             (places grip site near grasp height with horizontal finger closing)
  Phase 2 — Cartesian DLS IK → center grip site over cube in x/y
  Phase 3 — close right gripper  (fingers close in ±y, squeezing cube faces)
  Phase 4 — Cartesian DLS IK → lift 22 cm

The cube is placed at (0.55, -0.30, 0.525), on the table surface.
Q_REACH (w2=-1.78) orients fingers to close in the ±y direction (nearly horizontal),
so l_tip and r_tip straddle the cube y-span before closing.

Run from repo root:
    python mujoco_viewer/baxter_pick.py
"""

import pathlib
import time

import mujoco
import mujoco.viewer
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).parent.parent
XML_PATH  = REPO_ROOT / "models" / "baxter_withgripper.xml"

# ── Cube position (on table, within right arm workspace) ──────────────────────
CUBE_POS = np.array([0.55, -0.30, 0.525])

# ── Gripper open / closed ─────────────────────────────────────────────────────
OPEN   = (+0.020833, -0.020833)
CLOSED = (-0.0115,    +0.0115)

# ── Actuator indices  (nu = 19) ───────────────────────────────────────────────
# 0       : head_pan_v
# 1 .. 7  : right_s0_v … right_w2_v   (velocity)
# 8,  9   : right_gripper_l/r          (position)
# 10 .. 16: left_s0_v  … left_w2_v    (velocity)
# 17, 18  : left_gripper_l/r           (position)
CTRL_RARM = slice(1, 8)
CTRL_RG_L = 8
CTRL_RG_R = 9

# ── qvel / qpos indices for right arm  (nv=25, nq=26) ────────────────────────
QVEL_RARM = slice(7, 14)   # nv: after cube_free(6) + head(1)
QPOS_RARM = slice(8, 15)   # nq: after cube_free(7) + head(1)

# ── Reaching configuration ────────────────────────────────────────────────────
# Grid-searched (orientation-aware, w2 swept for horizontal finger closing):
#   grip site  ≈ (0.514, -0.307, 0.521)  — at grasp height
#   gripper z-axis ≈ (0.47, -0.02, -0.88)  — mostly pointing downward
#   fingers close in y-direction (gripper y-axis ≈ [0.06, 1.00, 0.005]) — HORIZONTAL
#   l_tip open at y≈-0.259, r_tip open at y≈-0.355 → straddle cube y-span [-0.325,-0.275]
#   after closing: l_tip→y≈-0.291, r_tip→y≈-0.322 (both contact cube faces)
#   r_tip z clearance ≈ 20 mm above table ✓
Q_REACH = np.array([0.409, 0.667, -0.333, 0.0, 0.5, 2.0, -1.78])

# Null-space centering: keep IK near Q_REACH so arm stays in this configuration
Q_MID   = Q_REACH.copy()

# ── IK parameters ─────────────────────────────────────────────────────────────
KP_CART   = 5.0    # Cartesian proportional gain [1/s]
KP_JOINT  = 4.0    # joint-space proportional gain [1/s]
K_NULL    = 2.0    # null-space centering gain
LAMBDA    = 0.05   # DLS damping factor
VEL_LIMIT = 1.5    # joint velocity clamp [rad/s]
SUBSTEPS  = 8      # sim steps per viewer.sync()
SIM_DT    = 0.002  # XML timestep


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_gripper(ctrl: np.ndarray, state: str) -> None:
    v = OPEN if state == "open" else CLOSED
    ctrl[CTRL_RG_L], ctrl[CTRL_RG_R] = v


def dls_ik(model: mujoco.MjModel, data: mujoco.MjData,
           site_id: int, target: np.ndarray) -> np.ndarray:
    """Damped least-squares Jacobian IK with null-space joint centering."""
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    J = jacp[:, QVEL_RARM]                                   # (3, 7)

    JJT   = J @ J.T                                          # (3, 3)
    J_dls = J.T @ np.linalg.inv(JJT + LAMBDA**2 * np.eye(3))  # (7, 3)

    err  = target - data.site_xpos[site_id]
    qdot = J_dls @ (KP_CART * err)

    # Null-space: keep joints near Q_REACH
    q_arm = data.qpos[QPOS_RARM]
    N     = np.eye(7) - J_dls @ J
    qdot += N @ (K_NULL * (Q_MID - q_arm))

    return np.clip(qdot, -VEL_LIMIT, VEL_LIMIT)


def joint_p(data: mujoco.MjData, q_target: np.ndarray) -> np.ndarray:
    """Joint-space proportional velocity command."""
    return np.clip(KP_JOINT * (q_target - data.qpos[QPOS_RARM]),
                   -VEL_LIMIT, VEL_LIMIT)


def tick(model, data, viewer) -> bool:
    """Advance SUBSTEPS and sync viewer. Returns viewer.is_running()."""
    for _ in range(SUBSTEPS):
        mujoco.mj_step(model, data)
    viewer.sync()
    return viewer.is_running()


def run_joint(model, data, viewer, q_target: np.ndarray,
              tol: float = 0.04, timeout: float = 10.0) -> bool:
    t0 = data.time
    while viewer.is_running():
        if np.linalg.norm(q_target - data.qpos[QPOS_RARM]) < tol:
            return True
        if data.time - t0 > timeout:
            return False
        data.ctrl[CTRL_RARM] = joint_p(data, q_target)
        tick(model, data, viewer)
    return False


def run_cart(model, data, viewer, site_id: int,
             target: np.ndarray, tol: float = 0.008,
             timeout: float = 10.0) -> bool:
    t0 = data.time
    while viewer.is_running():
        if np.linalg.norm(target - data.site_xpos[site_id]) < tol:
            return True
        if data.time - t0 > timeout:
            return False
        data.ctrl[CTRL_RARM] = dls_ik(model, data, site_id, target)
        tick(model, data, viewer)
    return False


def hold(model, data, viewer, n_syncs: int = 30) -> None:
    data.ctrl[CTRL_RARM] = 0.0
    for _ in range(n_syncs):
        if not viewer.is_running():
            break
        tick(model, data, viewer)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data  = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)

    # Place cube at the workspace position
    cube_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_adr = model.jnt_qposadr[cube_jid]
    data.qpos[cube_adr:cube_adr + 3] = CUBE_POS
    data.qpos[cube_adr + 3]          = 1.0
    data.qpos[cube_adr + 4:cube_adr + 7] = 0.0

    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grip_site")
    print(f"Grip site @ home : {data.site_xpos[site_id].round(4)}")
    print(f"Cube pos         : {CUBE_POS}")
    print(f"Pre-grasp target : {(CUBE_POS + [0, 0, 0.12]).round(4)}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.45, -0.25, 0.55]
        viewer.cam.distance  = 1.6
        viewer.cam.elevation = -18
        viewer.cam.azimuth   = 150

        # ── Phase 0: settle, open gripper ─────────────────────────────────────
        print("\n[0] Settle + open gripper")
        set_gripper(data.ctrl, "open")
        hold(model, data, viewer, n_syncs=50)

        # ── Phase 1: joint-space reach-out ────────────────────────────────────
        # After this, grip site is at ≈(0.514, -0.307, 0.521) — near grasp height.
        # Fingers close in y-direction (y-axis≈[0.06,1.00,0.005]); gripper z points mostly down.
        print(f"[1] Joint reach-out  Q_REACH = {Q_REACH}")
        ok = run_joint(model, data, viewer, Q_REACH, tol=0.04, timeout=10.0)
        grip = data.site_xpos[site_id].round(4)
        print(f"    done ok={ok}   grip site = {grip}")
        hold(model, data, viewer, n_syncs=20)

        # ── Phase 2: Cartesian correction — center over cube in x/y ─────────
        # Fingers close in y-direction (gripper y-axis ≈ [0.06, 1.00, 0.005]).
        # l_tip and r_tip straddle cube y-span; fingers approach from ±y.
        # Keep current z (r_tip already has ~20 mm clearance above table).
        cube_pos   = data.qpos[cube_adr:cube_adr + 3].copy()
        grasp_tgt  = np.array([cube_pos[0], cube_pos[1],
                                data.site_xpos[site_id][2]])
        print(f"[2] Center over cube → {grasp_tgt.round(4)}")
        ok = run_cart(model, data, viewer, site_id, grasp_tgt, tol=0.008, timeout=8.0)
        grip = data.site_xpos[site_id].round(4)
        print(f"    done ok={ok}   grip site = {grip}")
        hold(model, data, viewer, n_syncs=15)

        # ── Phase 3: close gripper ─────────────────────────────────────────────
        print("[3] Close gripper")
        set_gripper(data.ctrl, "closed")
        hold(model, data, viewer, n_syncs=100)   # 100 * 8 * 0.002 = 1.6 s
        cube_z = data.qpos[cube_adr + 2]
        print(f"    cube z after grasp = {cube_z:.4f}")

        # ── Phase 4: lift ──────────────────────────────────────────────────────
        lift_tgt = data.site_xpos[site_id].copy() + np.array([0.0, 0.0, 0.22])
        print(f"[4] Lift       → {lift_tgt.round(4)}")
        ok = run_cart(model, data, viewer, site_id, lift_tgt, tol=0.012, timeout=8.0)
        cube_z_final = data.qpos[cube_adr + 2]
        print(f"    done ok={ok}   cube z = {cube_z_final:.4f}  (start = {CUBE_POS[2]:.4f})")
        if cube_z_final > CUBE_POS[2] + 0.05:
            print("    *** PICK SUCCESSFUL ***")
        else:
            print("    *** cube did not lift — gripper may need orientation tuning ***")

        # ── Hold for 2 s then exit ─────────────────────────────────────────────
        print("\nHolding for 2 s ...")
        hold(model, data, viewer, n_syncs=int(2.0 / (SUBSTEPS * SIM_DT)))


if __name__ == "__main__":
    main()
