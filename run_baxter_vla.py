"""Run a Pi0.5 VLA policy on the Baxter MuJoCo simulation.

Architecture
------------
MuJoCo Baxter simulator
        ↓
capture camera image + robot state  (BaxterEnv.get_obs)
        ↓
build observation dict with flat "/" keys expected by the server
        ↓
send to policy server over WebSocket  (WebsocketClientPolicy.infer)
        ↓
receive action chunk  (shape: [chunk_size, 15])
        ↓
execute actions one-at-a-time via a deque (action_plan)
        ↓
sync the passive MuJoCo viewer so the window updates
        ↓
repeat until max_steps or viewer is closed

Usage
-----
    python run_baxter_vla.py --prompt "move cube to the right"

The policy server must already be running at --host:--port before this
script is started.  WebsocketClientPolicy blocks until the server responds.
"""

import collections
import dataclasses
import logging
import mujoco.viewer
import pathlib
import sys

import numpy as np
import tyro

# ---------------------------------------------------------------------------
# Locate the openpi_client package.  It lives inside openpi/packages/… and
# may not be on PYTHONPATH yet, so we add it here.
# ---------------------------------------------------------------------------
_OPENPI_CLIENT = (
    pathlib.Path(__file__).parent
    / "openpi"
    / "packages"
    / "openpi-client"
    / "src"
)
if str(_OPENPI_CLIENT) not in sys.path:
    sys.path.insert(0, str(_OPENPI_CLIENT))

# WebSocket policy client — connects to a running serve_policy.py server
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402

# Local environment wrapper around baxter.xml
from baxter_env import BaxterEnv  # noqa: E402


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Args:
    """Command-line arguments for the Baxter VLA evaluation loop."""

    # Path to the Baxter MJCF file
    xml_path: str = "models/baxter.xml"

    # Policy server address (host of the machine running serve_policy.py)
    host: str = "0.0.0.0"
    # Policy server port
    port: int = 8000

    # Natural-language task prompt sent to the VLA model on every inference call
    prompt: str = "pick up the cube"

    # Total number of simulator steps to run
    max_steps: int = 300

    # How many steps to execute from one action chunk before requesting a new one.
    # Setting this equal to the chunk size means every step replans;
    # setting it smaller recycles the tail of each chunk.
    replan_steps: int = 10


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run(args: Args) -> None:
    """Connect to the policy server and run a single Baxter episode."""

    # -----------------------------------------------------------------------
    # 1.  Create the MuJoCo environment
    # -----------------------------------------------------------------------
    logging.info("Loading Baxter environment from %s", args.xml_path)
    env = BaxterEnv(args.xml_path)

    # -----------------------------------------------------------------------
    # 2.  Connect to the VLA policy server over WebSocket.
    #
    #     WebsocketClientPolicy blocks here until the server is reachable.
    #     The server must be started separately:
    #         python openpi/scripts/serve_policy.py --port 8000
    # -----------------------------------------------------------------------
    logging.info("Connecting to policy server at ws://%s:%d …", args.host, args.port)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("Connected.  Server metadata: %s", client.get_server_metadata())

    # -----------------------------------------------------------------------
    # 3.  Reset the environment and open the interactive MuJoCo viewer.
    #
    #     launch_passive() opens the window without blocking — the physics
    #     loop below drives the simulation while viewer.sync() pushes each
    #     new state to the renderer.  Closing the window sets
    #     viewer.is_running() to False, which exits the loop cleanly.
    # -----------------------------------------------------------------------
    obs = env.reset()

    # action_plan holds the remaining actions from the most recent chunk.
    # When empty a new inference call is made to the server.
    action_plan: collections.deque[np.ndarray] = collections.deque()

    logging.info("Starting rollout — max_steps=%d, replan_steps=%d", args.max_steps, args.replan_steps)
    logging.info("Close the viewer window (or press Escape) to stop.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

        # -------------------------------------------------------------------
        # 4.  Main control loop
        # -------------------------------------------------------------------
        for step in range(args.max_steps):

            # Exit early if the user closes the viewer window
            if not viewer.is_running():
                logging.info("Viewer closed — stopping at step %d.", step)
                break

            # ---------------------------------------------------------------
            # 4a.  Build the observation dict.
            #
            #      The server expects flat "/" separated keys and channel-first
            #      images.  The "prompt" field is the natural-language
            #      instruction that conditions the policy.
            #
            #      obs["image"]  : uint8 (3, 224, 224) — channel-first RGB
            #      obs["state"]  : float64 (15,)        — joint positions
            # ---------------------------------------------------------------
            element = {
                # Camera image in channel-first format (C, H, W)
                "observation/image": obs["image"],
                # Joint-position state vector
                "observation/state": obs["state"].astype(np.float32),
                # Task instruction for the language-conditioned policy
                "prompt": args.prompt,
            }

            # ---------------------------------------------------------------
            # 4b.  Action-chunk broker: request a new chunk when the plan
            #      runs out.
            #
            #      The server returns a dict whose "actions" value has shape
            #      (chunk_size, action_dim).  We slice it into individual
            #      steps and push them into the deque so the simulator can
            #      consume them one at a time without re-querying the server
            #      each step.
            # ---------------------------------------------------------------
            if not action_plan:
                logging.info("Step %d: requesting new action chunk from server", step)
                response = client.infer(element)

                # response["actions"] shape: (chunk_size, 15)
                action_chunk: np.ndarray = response["actions"]

                # Only keep the first replan_steps actions from the chunk
                for i in range(min(args.replan_steps, len(action_chunk))):
                    action_plan.append(action_chunk[i])

            # Pop the next action from the front of the plan
            action = action_plan.popleft()

            # ---------------------------------------------------------------
            # 4c.  Step the simulator with the chosen action.
            #
            #      BaxterEnv.step() clips the action to ctrlrange, writes it
            #      to data.ctrl, and advances physics by one timestep.
            # ---------------------------------------------------------------
            obs, reward, done, info = env.step(action)

            # ---------------------------------------------------------------
            # 4d.  Push the new physics state to the viewer window.
            #
            #      viewer.sync() reads env.data (shared with the viewer) and
            #      redraws the scene.  This is the only call needed — no
            #      separate render() required for visualisation.
            # ---------------------------------------------------------------
            viewer.sync()

            if done:
                logging.info("Episode finished early at step %d.", step)
                break

    logging.info("Done.")
    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    args = tyro.cli(Args)
    run(args)
