"""
Baxter multitask inference client.

Connects to a running openpi policy server and runs one episode in the
MuJoCo Baxter sim, executing model actions in a live viewer window.

Tasks (choose via PROMPT):
    "move the cube to the right"
    "wave your hand"
    "push the block to the front"

Run:
    # Terminal 1 — policy server
    cd /home/robotlab/Desktop/saniya_ws/pi0.5_mujoco/openpi
    uv run scripts/serve_policy.py \
        policy:checkpoint \
        --policy.config pi05_baxter_multitask \
        --policy.dir checkpoints/pi05_baxter_multitask/run_002/39999

    # Terminal 2 — this script
    cd /home/robotlab/Desktop/saniya_ws/pi0.5_mujoco
    python mujoco_viewer/baxter_inference.py
"""

import argparse
import collections
import pathlib
import sys

import mujoco
import mujoco.viewer
import numpy as np

# Add openpi-client to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "openpi" / "packages" / "openpi-client" / "src"))
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

TASKS = [
    "move the cube to the right",   # task 0
    "wave your hand",               # task 1
    "push the block to the front",  # task 2
]

# ── Configuration ─────────────────────────────────────────────────────────────
HOST, PORT    = "0.0.0.0", 8000
MAX_STEPS     = 200   # policy steps (each = SUBSTEPS physics steps)
REPLAN_STEPS  = 5     # execute this many actions before querying server again
SUBSTEPS      = 50    # physics steps per action (SIM_HZ=100 / dataset fps=10 * N_SUBSTEPS=5)
IMG_SIZE      = 224

XML_PATH      = pathlib.Path(__file__).parent.parent / "models" / "baxter.xml"

# Indices (same as demo recorder)
RIGHT_QPOS = slice(8, 15)
RIGHT_CTRL = slice(1, 8)

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-p", type=str, default=TASKS[0],
                        help="Language prompt. Options: " + " | ".join(f'"{t}"' for t in TASKS))
    parser.add_argument("--task", "-t", type=int, default=None,
                        help="Task index 0/1/2 (shortcut instead of typing the full prompt)")
    args = parser.parse_args()

    prompt = TASKS[args.task] if args.task is not None else args.prompt

    print(f"Connecting to policy server at {HOST}:{PORT} ...")
    client = _websocket_client_policy.WebsocketClientPolicy(HOST, PORT)
    print(f"Prompt: '{prompt}'")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data  = mujoco.MjData(model)

    # Reset to keyframe
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=IMG_SIZE, width=IMG_SIZE)

    action_plan = collections.deque()
    t = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.5, -0.2, 0.6]
        viewer.cam.distance  = 2.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 160

        print("Running episode ...")
        while viewer.is_running() and t < MAX_STEPS:

            # ── Render cameras ────────────────────────────────────────────────
            renderer.update_scene(data, camera="scene_camera")
            img_scene = renderer.render().copy()   # HWC uint8

            renderer.update_scene(data, camera="right_hand_camera")
            img_wrist = renderer.render().copy()   # HWC uint8

            img_scene = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_scene, IMG_SIZE, IMG_SIZE)
            )
            img_wrist = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_wrist, IMG_SIZE, IMG_SIZE)
            )

            # ── Query policy server ───────────────────────────────────────────
            if not action_plan:
                obs = {
                    "observation/image":       np.transpose(img_scene, (2, 0, 1)),  # CHW
                    "observation/wrist_image": np.transpose(img_wrist, (2, 0, 1)),  # CHW
                    "observation/state":       data.qpos[RIGHT_QPOS].astype(np.float32),
                    "prompt":                  prompt,
                }
                action_chunk = client.infer(obs)["actions"]
                print(f"  step {t:4d}  action sample: {action_chunk[0, :7].round(3)}")
                action_plan.extend(action_chunk[:REPLAN_STEPS])

            action = action_plan.popleft()

            # ── Apply action for SUBSTEPS physics steps ───────────────────────
            data.ctrl[RIGHT_CTRL] = action[:7]
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
            viewer.sync()
            t += 1

    print(f"Episode ended at step {t}.")
    del renderer


if __name__ == "__main__":
    main()
