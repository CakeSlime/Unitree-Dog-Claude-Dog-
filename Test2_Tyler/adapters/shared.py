"""Import shared Phase 2 runtime without importing/modifying Connor's Test2.

Shared simulator/controller/sensors: Phase_2_Investigator and its pinned submodule.
Only generated scene-overlay paths are redirected, in memory, to Tyler's cache.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "Test2_Tyler"


def configure():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    cache = WORKSPACE / ".cache"
    (cache / "tmp").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(cache / "tmp"))
    phase2 = str(ROOT / "Phase_2_Investigator")
    if phase2 not in sys.path:
        sys.path.insert(0, phase2)
    from sim_env import environments
    environments.OVERLAY = cache / "scene_overlay"


class Go2Simulation:
    """Physical scripted-trot Go2. No hardware SDK or Connor policy is imported."""

    def __init__(self, config):
        configure()
        from sim_env.runtime import SimSession
        self.config = config
        self.session = SimSession(config["environment"], spawn=config["spawn"], scene_patch=self._lights)
        camera = self.session.sensors.cameras[self.session.camera_key]
        camera["width"], camera["height"] = config.get("camera_size", (640, 480))
        self.session.settle(config["settle_s"])

    def _lights(self, scene_path):
        import xml.etree.ElementTree as ET
        tree = ET.parse(scene_path)
        world = tree.getroot().find("worldbody")
        if self.config.get("chair_pose"):
            import math
            x, y, yaw = self.config["chair_pose"]
            chair = world.find(".//body[@name='living_room_task_chair_0']")
            if chair is None:
                raise ValueError("Configured chair not present in selected scene")
            chair.set("pos", f"{x} {y} -0.00175591")
            chair.set("quat", f"{math.cos(yaw / 2)} 0 0 {math.sin(yaw / 2)}")
        for i, pos in enumerate(("3 2 2.4", "5 3 2.4")):
            ET.SubElement(world, "light", name=f"tyler_ceiling_{i}", pos=pos, dir="0 0 -1",
                          diffuse="0.35 0.35 0.35", castshadow="false")
        tree.write(scene_path)

    def sample(self):
        # No physics advance between modalities: all timestamps refer to one simulator state.
        s = self.session
        return s.camera(), s.lidar(), s.state(), s.contacts()

    def step(self, command, dt):
        self.session.cmd_vel(*command)
        self.session.step(dt)

    def close(self):
        try:
            self.session.stop()
        finally:
            self.session.close()
