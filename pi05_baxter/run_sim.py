"""
run_sim.py - Main simulation loop.

Wires together BaxterEnv and Pi05Policy into a closed-loop inference cycle:

    observation → policy → action → MuJoCo step → repeat

Run from the project root:

    # With a real π0.5 checkpoint:
    python -m pi05_baxter.run_sim --checkpoint checkpoints/pi05_baxter

    # Without a checkpoint (random actions for testing):
    python -m pi05_baxter.run_sim --dummy-policy
"""

import argparse
import time

import numpy as np

from pi05_baxter.env import BaxterEnv
from pi05_baxter.utils import extract_ctrl_range, scale_action


# ---------------------------------------------------------------------------
# Dummy policy — used when --dummy-policy is set or openpi is unavailable.
# Outputs small smooth random velocities so the arms move visibly but gently.
# ---------------------------------------------------------------------------
class _DummyPolicy:
    def __init__(self, nu: int, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._nu  = nu
        self._vel = np.zeros(nu)          # current velocity command
        self._t   = 0

    def act(self, obs: dict) -> np.ndarray:  # noqa: ARG002
        # Slowly drift in random directions; change target every 60 steps
        if self._t % 60 == 0:
            self._target = self._rng.uniform(-0.3, 0.3, self._nu)
        self._vel += 0.05 * (self._target - self._vel)   # low-pass filter
        self._t  += 1
        return np.clip(self._vel, -1.0, 1.0)


def parse_args():
    p = argparse.ArgumentParser(description="π0.5 + MuJoCo Baxter sim loop")
    p.add_argument("--model",         default="models/baxter.xml",
                   help="Path to baxter.xml")
    p.add_argument("--checkpoint",    default="checkpoints/pi05_baxter",
                   help="π0.5 checkpoint path or HuggingFace repo id")
    p.add_argument("--instruction",   default="reach the object",
                   help="Language instruction sent to the policy")
    p.add_argument("--steps",         type=int, default=500,
                   help="Number of simulation steps")
    p.add_argument("--no-viewer",     action="store_true",
                   help="Disable the interactive MuJoCo viewer window")
    p.add_argument("--step-sleep",    type=float, default=0.0,
                   help="Optional sleep (s) between steps to slow down playback")
    p.add_argument("--dummy-policy",  action="store_true",
                   help="Use random actions instead of π0.5 (no checkpoint needed)")
    return p.parse_args()


def run(args):
    # ------------------------------------------------------------------
    # 1. Build environment
    # ------------------------------------------------------------------
    print("[run_sim] Loading MuJoCo environment …")
    env = BaxterEnv(
        model_path=args.model,
        img_size=(224, 224),
        show_viewer=not args.no_viewer,
    )

    # Actuator control range for action re-scaling
    ctrl_lower, ctrl_upper = extract_ctrl_range(env.model)

    # ------------------------------------------------------------------
    # 2. Load policy
    # ------------------------------------------------------------------
    if args.dummy_policy:
        print("[run_sim] Using dummy policy (smooth random actions).")
        policy = _DummyPolicy(nu=env.nu)
    else:
        from pi05_baxter.policy_wrapper import Pi05Policy
        print("[run_sim] Loading π0.5 policy …")
        policy = Pi05Policy(
            checkpoint_path=args.checkpoint,
            instruction=args.instruction,
        )

    # ------------------------------------------------------------------
    # 3. Reset and run loop
    # ------------------------------------------------------------------
    obs = env.reset()
    print(f"[run_sim] Starting simulation for {args.steps} steps …")

    t0 = time.monotonic()
    for step_idx in range(args.steps):
        # --- validate observation shapes before inference ---
        if obs["image"].shape != (224, 224, 3):
            raise RuntimeError(
                f"Unexpected image shape {obs['image'].shape}; expected (224, 224, 3)"
            )

        # --- policy inference ---
        raw_action = policy.act(obs)          # (nu,) in [-1, 1]

        # --- clip raw action to [-1, 1] before scaling (safety) ---
        raw_action = np.clip(raw_action, -1.0, 1.0)

        # --- scale from [-1, 1] to actuator control range ---
        action = scale_action(raw_action, (ctrl_lower, ctrl_upper))

        # --- step physics ---
        obs = env.step(action)

        if args.step_sleep > 0:
            time.sleep(args.step_sleep)

        if step_idx % 50 == 0:
            elapsed = time.monotonic() - t0
            print(f"  step {step_idx:4d}/{args.steps}"
                  f"  |  qpos[:3] = {obs['qpos'][:3].round(3)}"
                  f"  |  {elapsed:.1f}s elapsed")

    print("[run_sim] Done.")
    env.close()


if __name__ == "__main__":
    run(parse_args())
