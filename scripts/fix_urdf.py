"""
scripts/fix_urdf.py

Prepares models/urdf/baxter.urdf for direct MuJoCo loading:

  1. Strips  package://baxter_description/  ROS package prefix from all
     mesh filenames so paths become relative (e.g. meshes/torso/base_link.STL).

  2. Replaces .DAE collision mesh references with .STL equivalents — MuJoCo
     3.x does not reliably parse DAE files for collision geometry.

  3. Injects a  <mujoco><compiler meshdir="..."/></mujoco>  element into the
     URDF so MuJoCo knows the absolute base directory for mesh lookup,
     making the file loadable from any working directory.

Usage (run from project root):
    python scripts/fix_urdf.py

Input  : models/urdf/baxter.urdf
Output : models/baxter_mj.urdf
"""

import re
from pathlib import Path

SRC  = Path("models/urdf/baxter.urdf")
DEST = Path("models/baxter_mj.urdf")
MESHDIR = Path("models").resolve()  # mesh files are at models/meshes/…


def fix(text: str) -> str:
    # 1. Strip ROS package prefix
    text = text.replace("package://baxter_description/", "")

    # 2. In <collision> blocks, replace .DAE meshes with .STL
    #    We only target the filename inside <collision>…</collision> sections.
    def collision_dae_to_stl(m: re.Match) -> str:
        return m.group(0).replace(".DAE", ".STL").replace(".dae", ".stl")

    text = re.sub(
        r"<collision>.*?</collision>",
        collision_dae_to_stl,
        text,
        flags=re.DOTALL,
    )

    # 3. Inject <mujoco> compiler hint just inside the <robot …> opening tag
    mujoco_hint = f'\n  <mujoco><compiler meshdir="{MESHDIR}" strippath="false"/></mujoco>'
    text = re.sub(r"(<robot\b[^>]*>)", r"\1" + mujoco_hint, text, count=1)

    return text


if __name__ == "__main__":
    original = SRC.read_text()
    fixed    = fix(original)
    DEST.write_text(fixed)

    pkg_fixes  = original.count("package://baxter_description/")
    dae_colls  = len(re.findall(r"<collision>.*?\.DAE", original, re.DOTALL))
    print(f"Stripped {pkg_fixes} package:// prefixes")
    print(f"Replaced {dae_colls} DAE collision meshes → STL")
    print(f"Injected meshdir = {MESHDIR}")
    print(f"Written → {DEST}")
