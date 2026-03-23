"""Round 2: fix head camera position (push further from head mesh)."""
import os
import pathlib

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import PIL.Image

REPO  = pathlib.Path(__file__).parent.parent
MODEL = REPO / "models" / "baxter.xml"
OUT   = REPO / "scripts" / "cam_tests"
OUT.mkdir(exist_ok=True)

xml_orig = MODEL.read_text()

HEAD_ORIG = '<body name="head_camera_body" pos="0.12839 0 0.06368" euler="1.75057 0 1.5708">'
WRIST_ORIG = ('<body name="right_hand_camera_body"\n'
              '                                pos="0.03825 0.012 0.015355" euler="0 0 -1.5708">')


def render(patches, tag):
    xml = xml_orig
    for old, new in patches.items():
        xml = xml.replace(old, new)
    tmp = REPO / "models" / "test_camera.xml"
    tmp.write_text(xml)
    try:
        m = mujoco.MjModel.from_xml_path(str(tmp))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, height=480, width=640)
        for cam in ("head_camera", "right_hand_camera"):
            cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            if cam_id < 0:
                continue
            r.update_scene(d, camera=cam)
            img = r.render()
            p = OUT / f"{tag}_{cam}.png"
            PIL.Image.fromarray(img).save(p)
            print(f"  {p.name}")
        r.close()
    finally:
        tmp.unlink(missing_ok=True)


# Head camera: push position forward (+x) and up (+z) to clear the head mesh
# Also try negative yaw (γ < 0) to look slightly right toward the table
# euler="0 β γ": look direction = (sin β · cos γ, -sin γ, -cos β · cos γ)
head_candidates = {
    # forward + down 40°, slight right yaw
    "h2_fwd_down40": HEAD_ORIG.replace('pos="0.12839 0 0.06368" euler="1.75057 0 1.5708"',
                                       'pos="0.20 0 0.07" euler="0 0.70 -0.35"'),
    # push even further, steeper down
    "h2_fwd_down50": HEAD_ORIG.replace('pos="0.12839 0 0.06368" euler="1.75057 0 1.5708"',
                                       'pos="0.22 0 0.07" euler="0 0.87 -0.35"'),
    # elevated mount, looking down more steeply (like an over-shoulder cam)
    "h2_elevated":   HEAD_ORIG.replace('pos="0.12839 0 0.06368" euler="1.75057 0 1.5708"',
                                       'pos="0.15 0 0.20" euler="0 1.10 -0.30"'),
    # direct overhead-ish but facing forward
    "h2_high_down":  HEAD_ORIG.replace('pos="0.12839 0 0.06368" euler="1.75057 0 1.5708"',
                                       'pos="0.18 0 0.25" euler="0 1.20 -0.25"'),
}

# Wrist camera: w_fwd was promising; refine it
WRIST_FWD = WRIST_ORIG.replace('pos="0.03825 0.012 0.015355" euler="0 0 -1.5708"',
                               'pos="0.06 0 0.04" euler="0 1.5708 0"')

wrist_candidates = {
    # slight tilt to look more down-forward (more toward gripper workspace)
    "w2_fwd_tilt": WRIST_ORIG.replace('pos="0.03825 0.012 0.015355" euler="0 0 -1.5708"',
                                      'pos="0.05 0 0.04" euler="-0.30 1.5708 0"'),
    # from above, angled down like a hand-eye camera
    "w2_above_down": WRIST_ORIG.replace('pos="0.03825 0.012 0.015355" euler="0 0 -1.5708"',
                                        'pos="0 0 0.10" euler="3.14159 0 0"'),
    # from front of hand, looking back toward fingertips
    "w2_front": WRIST_ORIG.replace('pos="0.03825 0.012 0.015355" euler="0 0 -1.5708"',
                                   'pos="0.09 0 0.03" euler="0 2.094 0"'),
}

print("=== Head camera round 2 ===")
for tag, new in head_candidates.items():
    print(tag)
    render({HEAD_ORIG: new}, tag)

print("\n=== Wrist camera round 2 ===")
for tag, new in wrist_candidates.items():
    print(tag)
    render({WRIST_ORIG: new}, tag)
