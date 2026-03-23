"""Round 3: world-fixed scene camera + finalise wrist camera.

A world-fixed camera (not attached to any moving body) guarantees a clear view
of the workspace regardless of the robot pose.
"""
import os
import pathlib

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import PIL.Image

REPO  = pathlib.Path(__file__).parent.parent
MODEL = REPO / "models" / "baxter.xml"
OUT   = REPO / "scripts" / "cam_tests"
OUT.mkdir(exist_ok=True)

xml_orig = MODEL.read_text()

WRIST_ORIG = ('<body name="right_hand_camera_body"\n'
              '                                pos="0.03825 0.012 0.015355" euler="0 0 -1.5708">')

# ── World-fixed scene cameras ─────────────────────────────────────────────────
# Inject a world-body camera just after the <worldbody> tag.
# Using xyaxes="x1 y1 z1  x2 y2 z2" where first triplet = camera X (right),
# second triplet = camera Y (up).  Camera looks along -Z = X cross Y.
#
# Camera at (1.5, 0.4, 1.4) facing the table at (0.65, -0.20, 0.30):
#   look_dir = (0.65-1.5, -0.2-0.4, 0.3-1.4) = (-0.85, -0.6, -1.1)
#   normalize: mag=1.49, dir=(-0.570, -0.403, -0.738)
#   camera_right = world_up × look_dir = (0,0,1)×(-0.570,-0.403,-0.738)
#     = (-0.403*1-(-0.738)*0, (-0.738)*(-0.570)... actually:
#     = (0*(-0.738)-1*(-0.403), 1*(-0.570)-0*(-0.738), 0*(-0.403)-0*(-0.570))
#     = (0.403, -0.570, 0), norm=0.697, normalised=(0.578, -0.818, 0)
#   camera_up = look_dir × camera_right  (but we want up in camera, pointing roughly +Z world)
#     = (-0.570,-0.403,-0.738) × (0.578,-0.818,0)
#     = (-0.403*0-(-0.738)*(-0.818), (-0.738)*0.578-(-0.570)*0, (-0.570)*(-0.818)-(-0.403)*0.578)
#     = (-0.603, -0.427, 0.466), normalize: mag=0.856, = (-0.705,-0.499,0.544)
# So xyaxes = "0.578 -0.818 0  -0.705 -0.499 0.544"

WORLD_CAM_FRONT = """
    <!-- Fixed scene camera: front-right elevated view of workspace -->
    <body name="scene_camera_body" pos="1.5 0.4 1.4">
      <camera name="scene_camera" xyaxes="0.578 -0.818 0  -0.705 -0.499 0.544" fovy="55"/>
    </body>"""

WORLD_CAM_SIDE = """
    <!-- Fixed scene camera: side view from the right -->
    <body name="scene_camera_body" pos="1.0 -1.2 1.2">
      <camera name="scene_camera" xyaxes="0.832 0.555 0  -0.400 0.600 0.693" fovy="55"/>
    </body>"""

# Simpler: just use euler on the world body (pos + euler approach)
# Camera at (1.4, 0.3, 1.5), euler such that it looks toward (0.65, -0.2, 0.3)
# Look dir = (-0.75, -0.5, -1.2), normalize: mag=1.46, = (-0.514,-0.342,-0.821)
# With euler="α β γ", local -Z = -(row2 of R) = (cx*sy*cz - sx*sz, -cx*sy*sz - sx*cz, -cx*cy)
# Try α=0.4, β=1.0, γ=-0.6 as an empirical guess:
WORLD_CAM_EULER = """
    <!-- Fixed scene camera: slightly elevated front view -->
    <body name="scene_camera_body" pos="1.4 0.3 1.5" euler="0.35 0.95 -0.60">
      <camera name="scene_camera" fovy="60"/>
    </body>"""


def render_with_world_cam(cam_xml_block, tag):
    xml = xml_orig.replace(
        "<worldbody>",
        "<worldbody>" + cam_xml_block
    )
    # Also fix wrist camera with best candidate from round 1
    xml = xml.replace(
        WRIST_ORIG,
        WRIST_ORIG.replace('pos="0.03825 0.012 0.015355" euler="0 0 -1.5708"',
                           'pos="0.06 0 0.04" euler="0 1.5708 0"')
    )
    tmp = REPO / "models" / "test_camera.xml"
    tmp.write_text(xml)
    try:
        m = mujoco.MjModel.from_xml_path(str(tmp))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, height=480, width=640)
        for cam in ("scene_camera", "right_hand_camera"):
            cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            if cam_id < 0:
                print(f"  {cam}: not found")
                continue
            r.update_scene(d, camera=cam)
            img = r.render()
            p = OUT / f"{tag}_{cam}.png"
            PIL.Image.fromarray(img).save(p)
            print(f"  {p.name}")
        r.close()
    finally:
        tmp.unlink(missing_ok=True)


print("=== World-fixed scene camera candidates ===")
for label, block in [("front_right", WORLD_CAM_FRONT),
                     ("side",        WORLD_CAM_SIDE),
                     ("euler",       WORLD_CAM_EULER)]:
    print(label)
    render_with_world_cam(block, f"wc_{label}")
