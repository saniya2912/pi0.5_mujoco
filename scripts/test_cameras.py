"""Iterate on camera euler angles and save renders to verify orientation.

Run from repo root:
    python scripts/test_cameras.py
"""
import os
import pathlib
import re
import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco  # noqa: E402

REPO  = pathlib.Path(__file__).parent.parent
MODEL = REPO / "models" / "baxter.xml"
OUT   = REPO / "scripts" / "cam_tests"
OUT.mkdir(exist_ok=True)

xml_orig = MODEL.read_text()


def render_with_patches(patches: dict[str, str], tag: str) -> None:
    """patch is {old: new}; write to models/test_camera.xml, load, render."""
    xml = xml_orig
    for old, new in patches.items():
        xml = xml.replace(old, new)

    # Write alongside the model so relative mesh paths resolve
    tmp = REPO / "models" / "test_camera.xml"
    tmp.write_text(xml)

    try:
        m = mujoco.MjModel.from_xml_path(str(tmp))
        d = mujoco.MjData(m)
        # Set home keyframe
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)

        r = mujoco.Renderer(m, height=480, width=640)
        for cam_name in ("head_camera", "right_hand_camera"):
            try:
                cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                if cam_id < 0:
                    continue
                r.update_scene(d, camera=cam_name)
                img = r.render()
                fname = OUT / f"{tag}_{cam_name}.png"
                import PIL.Image
                PIL.Image.fromarray(img).save(fname)
                print(f"  saved {fname}")
            except Exception as e:
                print(f"  {cam_name}: {e}")
        r.close()
    finally:
        tmp.unlink(missing_ok=True)


# ─── head camera candidates ───────────────────────────────────────────────────
# Goal: camera at (~0.19, 0, ~0.75) looking forward (+X) and down toward table
# With euler="0 β 0" (XYZ intrinsic), camera -Z = (sin β, 0, -cos β)
# Table at (0.65, -0.20, 0.30).  Ideal pitch: ~42° below horizontal → β≈0.83
# Also test with yaw (γ) to aim at table in Y

HEAD_ORIG = '<body name="head_camera_body" pos="0.12839 0 0.06368" euler="1.75057 0 1.5708">'

candidates_head = {
    "h_b083"      : HEAD_ORIG.replace('euler="1.75057 0 1.5708"', 'euler="0 0.83 0"'),
    "h_b083_g-04" : HEAD_ORIG.replace('euler="1.75057 0 1.5708"', 'euler="0 0.83 -0.40"'),
    "h_b090"      : HEAD_ORIG.replace('euler="1.75057 0 1.5708"', 'euler="0 0.90 -0.40"'),
    "h_b100"      : HEAD_ORIG.replace('euler="1.75057 0 1.5708"', 'euler="0 1.00 -0.40"'),
}

# ─── right_hand camera candidates ─────────────────────────────────────────────
# Goal: camera on the wrist looking forward along the arm / down toward gripper
# right_hand body is at the wrist tip.  Camera needs to be outside the mesh.
# Current pos="0.03825 0.012 0.015355" is inside the wrist mesh → push out in Z.
# Try pos="0 0 0.08" pointing along wrist axis (local -Z of camera = wrist +Z?).

WRIST_ORIG = '<body name="right_hand_camera_body"\n                                pos="0.03825 0.012 0.015355" euler="0 0 -1.5708">'

candidates_wrist = {
    "w_out_z"    : WRIST_ORIG.replace('pos="0.03825 0.012 0.015355" euler="0 0 -1.5708"',
                                      'pos="0 0 0.09" euler="0 0 0"'),
    "w_out_z_r90": WRIST_ORIG.replace('pos="0.03825 0.012 0.015355" euler="0 0 -1.5708"',
                                      'pos="0 0 0.09" euler="3.14159 0 0"'),
    "w_orig_down": WRIST_ORIG.replace('pos="0.03825 0.012 0.015355" euler="0 0 -1.5708"',
                                      'pos="0.04 0 0.07" euler="1.5708 0 1.5708"'),
    "w_fwd"      : WRIST_ORIG.replace('pos="0.03825 0.012 0.015355" euler="0 0 -1.5708"',
                                      'pos="0.06 0 0.04" euler="0 1.5708 0"'),
}

print("=== Head camera candidates ===")
for tag, new_body in candidates_head.items():
    print(f"\n{tag}")
    render_with_patches({HEAD_ORIG: new_body}, tag)

print("\n=== Wrist camera candidates ===")
for tag, new_body in candidates_wrist.items():
    print(f"\n{tag}")
    render_with_patches({WRIST_ORIG: new_body}, tag)

print("\nDone. Images saved to", OUT)
