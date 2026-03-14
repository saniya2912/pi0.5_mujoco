#!/usr/bin/env python3
"""
View the Baxter robot in MuJoCo.

This script:
  1. Creates  mujoco_viewer/meshes/  with symlinks to all STL files
     (MuJoCo's URDF importer resolves meshes by basename + meshdir)
  2. Reads the original Baxter URDF from  models/urdf/baxter.urdf
  3. Replaces .DAE mesh references with .STL equivalents
     (MuJoCo 3.x does not support Collada meshes)
  4. Strips ROS/Gazebo-only elements  (<gazebo>, <transmission>, <plugin>)
  5. Injects a <mujoco> block pointing meshdir at the flat symlinks directory
  6. Saves the cleaned URDF as  mujoco_viewer/baxter_cleaned.urdf
  7. Loads into MjSpec, adds a floor / lights / overview camera
  8. Saves the complete scene as  mujoco_viewer/baxter_scene.xml
  9. Compiles the model and opens the interactive MuJoCo viewer

Usage:
    python mujoco_viewer/view_baxter.py

Requirements:
    pip install mujoco
"""

import os
import re
import sys

import mujoco
import mujoco.viewer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(SCRIPT_DIR)

# Source URDF – never modified
URDF_SRC     = os.path.join(REPO_ROOT, "models", "urdf", "baxter.urdf")

# Original mesh tree (all STL/DAE files live under here)
MESHES_ROOT  = os.path.join(REPO_ROOT, "models", "meshes")

# Flat symlinks directory – MuJoCo finds meshes here by basename
FLAT_MESHES  = os.path.join(SCRIPT_DIR, "meshes")

# Generated output files – all inside mujoco_viewer/
CLEANED_URDF = os.path.join(SCRIPT_DIR, "baxter_cleaned.urdf")
SCENE_XML    = os.path.join(SCRIPT_DIR, "baxter_scene.xml")

# ROS/Gazebo top-level tags MuJoCo cannot parse
_STRIP_TAGS = ("gazebo", "transmission", "plugin")


# ---------------------------------------------------------------------------
# Step 1 – build a flat meshes/ directory with symlinks to all STL files
# ---------------------------------------------------------------------------
def setup_flat_meshes() -> None:
    """
    Create FLAT_MESHES/ and populate it with symlinks to every *.STL file
    found anywhere under MESHES_ROOT.

    MuJoCo's URDF importer resolves mesh filenames by basename only; it
    ignores subdirectory structure.  A flat directory lets it find everything
    with a single meshdir setting.
    """
    if not os.path.isdir(MESHES_ROOT):
        sys.exit(
            f"ERROR: mesh directory not found:\n  {MESHES_ROOT}\n"
            "Make sure  models/meshes/  exists in the repository."
        )

    os.makedirs(FLAT_MESHES, exist_ok=True)

    linked, skipped = 0, 0
    missing_stl: list[str] = []

    for dirpath, _dirs, fnames in os.walk(MESHES_ROOT):
        for fname in fnames:
            if not fname.upper().endswith(".STL"):
                continue
            src = os.path.join(dirpath, fname)
            dst = os.path.join(FLAT_MESHES, fname)
            if os.path.lexists(dst):
                skipped += 1
            else:
                os.symlink(src, dst)
                linked += 1

    # Sanity check – all expected mesh basenames present?
    expected = {
        "base_link.STL", "base_link_collision.STL",
        "PEDESTAL.STL", "pedestal_link_collision.STL",
        "H0.STL", "H1.STL",
        "S0.STL", "S1.STL", "E0.STL", "E1.STL",
        "W0.STL", "W1.STL", "W2.STL",
    }
    present = set(os.listdir(FLAT_MESHES))
    for name in sorted(expected - present):
        missing_stl.append(name)

    if missing_stl:
        print(f"  WARNING: expected STL files not found in {FLAT_MESHES}:")
        for m in missing_stl:
            print(f"    {m}")
    else:
        print(f"[1/4] Mesh symlinks ready  ({linked} new, {skipped} existing)")


# ---------------------------------------------------------------------------
# Step 2 – pre-process the URDF
# ---------------------------------------------------------------------------
def preprocess_urdf() -> None:
    """
    Read the original URDF, apply these transforms, and write CLEANED_URDF:
      • .DAE → .STL  (MuJoCo 3.x has no Collada support)
      • strip <gazebo>, <transmission>, <plugin> blocks
      • inject a <mujoco><compiler …/></mujoco> extension block so that
        MuJoCo uses FLAT_MESHES as the mesh search directory
    """
    if not os.path.isfile(URDF_SRC):
        sys.exit(
            f"ERROR: URDF not found:\n  {URDF_SRC}\n"
            "Make sure  models/urdf/baxter.urdf  exists."
        )

    print(f"[2/4] Pre-processing URDF …")
    with open(URDF_SRC) as f:
        xml = f.read()

    # ---- .DAE → .STL -----------------------------------------------------
    dae_n = xml.count(".DAE")
    xml = xml.replace(".DAE", ".STL")
    print(f"      Replaced {dae_n} .DAE references → .STL")

    # ---- strip unsupported tags ------------------------------------------
    before = xml.count("<")
    for tag in _STRIP_TAGS:
        xml = re.sub(rf"\s*<{tag}(\s[^/]*)*/\s*>", "", xml)
        xml = re.sub(rf"\s*<{tag}(\s[^>]*)?>.*?</{tag}>", "", xml, flags=re.DOTALL)
    print(f"      Stripped {before - xml.count('<')} tags  {_STRIP_TAGS}")

    # ---- inject <mujoco> extension block ---------------------------------
    # Placed immediately after <robot ...> so it is the first child element.
    #
    # meshdir   – where MuJoCo looks for mesh files by basename
    # balanceinertia  – silently fix near-zero/invalid inertia values common
    #                   in ROS robot descriptions
    # discardvisual   – keep visual geoms (default is to discard them)
    mujoco_block = (
        f'\n  <mujoco>\n'
        f'    <compiler meshdir="{FLAT_MESHES}"\n'
        f'              balanceinertia="true"\n'
        f'              discardvisual="false"/>\n'
        f'  </mujoco>'
    )
    xml, n_subs = re.subn(r"(<robot\b[^>]*>)", r"\1" + mujoco_block, xml, count=1)
    if n_subs == 0:
        sys.exit("ERROR: could not find <robot …> root element in URDF.")

    with open(CLEANED_URDF, "w") as f:
        f.write(xml)
    print(f"      Saved → {CLEANED_URDF}")


# ---------------------------------------------------------------------------
# Step 3 – load into MjSpec and add scene elements
# ---------------------------------------------------------------------------
def build_spec() -> mujoco.MjSpec:
    """
    Load the cleaned URDF into an MjSpec, add floor / lights / camera,
    and return the spec ready to compile.
    """
    print(f"[3/4] Loading URDF into MuJoCo {mujoco.__version__} …")

    try:
        spec = mujoco.MjSpec.from_file(CLEANED_URDF)
    except Exception as exc:
        sys.exit(
            f"ERROR: MuJoCo could not parse the cleaned URDF:\n  {exc}\n\n"
            "Possible fixes:\n"
            "  • Verify all STL symlinks in  mujoco_viewer/meshes/\n"
            "  • Re-run the script to regenerate baxter_cleaned.urdf\n"
            "  • Check for remaining unsupported URDF elements"
        )

    worldbody = spec.worldbody

    # ---- ground plane -----------------------------------------------------
    floor = worldbody.add_geom()
    floor.name  = "floor"
    floor.type  = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size  = [6.0, 6.0, 0.1]
    floor.rgba  = [0.80, 0.80, 0.80, 1.0]
    # Baxter's pedestal bottom is ~z=-0.924 m; place floor just below it
    floor.pos   = [0.0, 0.0, -0.93]

    # ---- key light (upper-right-front, casts shadows) ---------------------
    key = worldbody.add_light()
    key.name       = "key_light"
    key.pos        = [3.0, -3.0, 5.0]
    key.dir        = [-1.0,  1.0, -2.0]   # normalised by MuJoCo
    key.diffuse    = [1.0, 0.95, 0.9]
    key.specular   = [0.4,  0.4,  0.4]
    key.castshadow = True

    # ---- fill light (upper-left-back, no shadows) ------------------------
    fill = worldbody.add_light()
    fill.name       = "fill_light"
    fill.pos        = [-3.0,  3.0, 4.0]
    fill.dir        = [ 1.0, -1.0, -1.5]
    fill.diffuse    = [0.5,  0.5,  0.6]
    fill.specular   = [0.0,  0.0,  0.0]
    fill.castshadow = False

    # ---- overview camera --------------------------------------------------
    # Position [3, -3, 2.5] looking toward the robot centre ~[0, 0, 1].
    #
    # Camera frame (MuJoCo convention: X=right, Y=up, -Z=forward):
    #   forward  = normalise([0,0,1] - [3,-3,2.5]) = normalise([-3, 3,-1.5])
    #   X(right) = normalise(forward × world_Z)    ≈ (0.707, 0.707, 0)
    #   Y(up)    = normalise(X × forward)          ≈ (-0.236, 0.236, 0.942)
    cam = worldbody.add_camera()
    cam.name = "overview"
    cam.pos  = [3.0, -3.0, 2.5]
    cam.alt.xyaxes = [0.707, 0.707, 0.0, -0.236, 0.236, 0.942]

    print("      Added: floor, 2 lights, overview camera")
    return spec


# ---------------------------------------------------------------------------
# Step 4 – compile, save scene XML, launch viewer
# ---------------------------------------------------------------------------
def launch(spec: mujoco.MjSpec) -> None:
    # compile
    try:
        model = spec.compile()
    except Exception as exc:
        sys.exit(f"ERROR: model compilation failed:\n  {exc}")

    print(
        f"      nq={model.nq}  nv={model.nv}  "
        f"nbody={model.nbody}  nmesh={model.nmesh}"
    )

    # save scene XML (informational – viewer uses compiled model directly)
    print(f"[4/4] Saving scene XML …")
    try:
        spec.to_file(SCENE_XML)
        print(f"      Saved → {SCENE_XML}")
    except Exception as exc:
        print(f"  WARNING: could not save scene XML: {exc}")

    data = mujoco.MjData(model)
    print(
        "\nLaunching MuJoCo viewer"
        " — close the window or press Escape to exit.\n"
    )
    mujoco.viewer.launch(model, data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    setup_flat_meshes()
    preprocess_urdf()
    spec  = build_spec()
    launch(spec)


if __name__ == "__main__":
    main()
