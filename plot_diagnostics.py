"""Generate diagnostic plots from a run_baxter_vla.py CSV log.

Usage:
    python plot_diagnostics.py
    python plot_diagnostics.py --csv data/baxter/diagnostics.csv --out data/baxter/plots
"""

import dataclasses
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tyro

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

JOINT_NAMES = [
    "head_pan",
    "right_s0", "right_s1", "right_e0", "right_e1",
    "right_w0", "right_w1", "right_w2",
    "left_s0",  "left_s1",  "left_e0",  "left_e1",
    "left_w0",  "left_w1",  "left_w2",
]

POLICY_ACTION_LABELS = [
    "Δeef_x", "Δeef_y", "Δeef_z",
    "Δroll",  "Δpitch", "Δyaw",
    "gripper",
]


@dataclasses.dataclass
class Args:
    csv: pathlib.Path = pathlib.Path("data/baxter/diagnostics.csv")
    out: pathlib.Path = pathlib.Path("data/baxter/plots")


def load(csv_path: pathlib.Path):
    import csv
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    steps        = np.array([r["step"] for r in rows])
    policy_act   = np.array([[r[f"policy_action_{i}"] for i in range(7)]  for r in rows])
    qpos         = np.array([[r[f"qpos_{i}"]          for i in range(15)] for r in rows])
    ctrl         = np.array([[r[f"ctrl_{i}"]          for i in range(15)] for r in rows])
    return steps, policy_act, qpos, ctrl


# ─── Plot 1: ctrl sent to all 15 joints ──────────────────────────────────────

def plot_ctrl_all_joints(steps, ctrl, out: pathlib.Path):
    fig, axes = plt.subplots(3, 5, figsize=(18, 9), sharex=True)
    fig.suptitle(
        "Control signals sent to all 15 Baxter joints\n"
        "(joints 7–14 = left arm, receive zero signal — never active)",
        fontsize=13, fontweight="bold"
    )
    for i, ax in enumerate(axes.flat):
        color = "#e05c5c" if i >= 7 else "#4c8fcc"
        ax.plot(steps, ctrl[:, i], color=color, lw=1.2)
        ax.set_title(JOINT_NAMES[i])
        ax.set_ylim(-1.6, 1.6)
        if i >= 7:
            ax.set_facecolor("#fff0f0")
            ax.axhline(0, color="#e05c5c", lw=1.5, ls="--")
    for ax in axes[-1]:
        ax.set_xlabel("Step")
    fig.text(0.01, 0.5, "ctrl (rad/s)", va="center", rotation="vertical")
    fig.tight_layout()
    path = out / "01_ctrl_all_joints.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ─── Plot 2: Right arm vs left arm activity comparison ────────────────────────

def plot_arm_activity(steps, ctrl, out: pathlib.Path):
    right_rms = np.sqrt(np.mean(ctrl[:, 1:8] ** 2, axis=1))
    left_rms  = np.sqrt(np.mean(ctrl[:, 8:15] ** 2, axis=1))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, right_rms, label="Right arm (joints 1–7)", color="#4c8fcc", lw=1.5)
    ax.plot(steps, left_rms,  label="Left arm  (joints 8–14)", color="#e05c5c", lw=1.5)
    ax.set_title(
        "RMS control signal: right arm vs left arm\n"
        "Left arm is entirely inactive (all zeros) — 8 of 15 joints receive no policy signal",
        fontweight="bold"
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("RMS ctrl (rad/s)")
    ax.legend()
    fig.tight_layout()
    path = out / "02_arm_activity_rms.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ─── Plot 3: Raw 7D policy actions over time ─────────────────────────────────

def plot_policy_actions(steps, policy_act, out: pathlib.Path):
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(
        "Raw 7-D policy actions output by pi05_libero\n"
        "(steps-within-chunk are identical — action refreshes every 10 steps)",
        fontsize=13, fontweight="bold"
    )
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    for i, ax in enumerate(axes):
        ax.plot(steps, policy_act[:, i], color=colors[i], lw=1.2)
        ax.set_ylabel(POLICY_ACTION_LABELS[i], fontsize=9)
        ax.set_ylim(-1.2, 1.2)
        # shade chunk boundaries
        for chunk_start in range(0, int(steps[-1]), 10):
            ax.axvline(chunk_start, color="gray", lw=0.5, ls=":", alpha=0.6)
    axes[-1].set_xlabel("Step")
    # highlight gripper always at -1
    axes[6].axhline(-1.0, color="red", lw=1, ls="--", alpha=0.7, label="gripper=-1 (always closed)")
    axes[6].legend(fontsize=8)
    fig.tight_layout()
    path = out / "03_policy_actions_7d.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ─── Plot 4: State mismatch — what policy "thinks" vs actual joint angles ─────

def plot_state_mismatch(steps, qpos, out: pathlib.Path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    fig.suptitle(
        "State mismatch: Baxter joint angles (qpos) sent as LIBERO end-effector state\n"
        "Policy expects [eef_x, eef_y, eef_z, roll, pitch, yaw, grip_L, grip_R] — receives Baxter joint angles instead",
        fontsize=12, fontweight="bold"
    )
    libero_state_labels = [
        "eef_x (m)", "eef_y (m)", "eef_z (m)",
        "roll (rad)", "pitch (rad)", "yaw (rad)",
        "gripper_L", "gripper_R"
    ]
    baxter_sent_labels = JOINT_NAMES[:8]
    colors_actual  = "#4c8fcc"
    for idx, ax in enumerate(axes.flat):
        ax.plot(steps, qpos[:, idx], color=colors_actual, lw=1.2)
        ax.set_title(
            f"Policy expects: {libero_state_labels[idx]}\n"
            f"Receives: {baxter_sent_labels[idx]}",
            fontsize=9
        )
    for ax in axes[-1]:
        ax.set_xlabel("Step")
    fig.tight_layout()
    path = out / "04_state_mismatch.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ─── Plot 5: Joint drift — how much each joint drifted from home pose ─────────

def plot_joint_drift(steps, qpos, out: pathlib.Path):
    HOME = np.array([0, 0, -0.9599, 0, 2.0, 0, 0.7854, 0,
                     0, -0.9599, 0, 2.0, 0, 0.7854, 0])
    drift = qpos - HOME[np.newaxis, :]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        "Joint drift from home pose over 300 steps\n"
        "Right arm drifts (policy active); left arm stays frozen at home",
        fontsize=13, fontweight="bold"
    )
    colors_r = plt.cm.Blues(np.linspace(0.4, 1.0, 7))
    colors_l = plt.cm.Reds(np.linspace(0.4, 1.0, 7))

    ax = axes[0]
    ax.set_title("Right arm joints (1–7)")
    for i, c in zip(range(1, 8), colors_r):
        ax.plot(steps, drift[:, i], label=JOINT_NAMES[i], color=c, lw=1.3)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Δqpos (rad)")
    ax.legend(ncol=4, fontsize=8)

    ax = axes[1]
    ax.set_title("Left arm joints (8–14) — should be zero drift, confirming no policy signal")
    for i, c in zip(range(8, 15), colors_l):
        ax.plot(steps, drift[:, i], label=JOINT_NAMES[i], color=c, lw=1.3)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Δqpos (rad)")
    ax.legend(ncol=4, fontsize=8)

    fig.tight_layout()
    path = out / "05_joint_drift.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ─── Plot 6: Action chunk repetition visualised ───────────────────────────────

def plot_chunk_repetition(steps, policy_act, out: pathlib.Path):
    """Show that within each 10-step chunk the same action is repeated."""
    fig, ax = plt.subplots(figsize=(12, 4))
    action_norm = np.linalg.norm(policy_act, axis=1)
    ax.plot(steps, action_norm, color="#4c8fcc", lw=1.2, label="‖policy action‖")
    for chunk_start in range(0, int(steps[-1]), 10):
        ax.axvspan(chunk_start, chunk_start + 10,
                   alpha=0.08 if (chunk_start // 10) % 2 == 0 else 0.0,
                   color="gray")
        ax.axvline(chunk_start, color="gray", lw=0.6, ls=":")
    ax.set_title(
        "L2 norm of policy action over time\n"
        "Flat segments = same action repeated for 10 steps (one chunk). "
        "Policy replans every 10 steps regardless of scene change.",
        fontweight="bold"
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("‖action‖")
    ax.legend()
    fig.tight_layout()
    path = out / "06_chunk_repetition.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main(args: Args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    steps, policy_act, qpos, ctrl = load(args.csv)
    print(f"Loaded {len(steps)} steps from {args.csv}")

    plot_ctrl_all_joints(steps, ctrl, args.out)
    plot_arm_activity(steps, ctrl, args.out)
    plot_policy_actions(steps, policy_act, args.out)
    plot_state_mismatch(steps, qpos, args.out)
    plot_joint_drift(steps, qpos, args.out)
    plot_chunk_repetition(steps, policy_act, args.out)

    print(f"\nAll plots saved to {args.out}/")


if __name__ == "__main__":
    main(tyro.cli(Args))
