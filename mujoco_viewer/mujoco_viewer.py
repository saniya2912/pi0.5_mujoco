import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("/home/robotlab/Desktop/saniya_ws/pi0.5_mujoco/models/baxter_withgripper.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
