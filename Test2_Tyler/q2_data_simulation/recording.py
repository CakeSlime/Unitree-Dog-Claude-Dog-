"""Tyler's sensor-aligned trial format. Each directory is exclusive and never reused.

Inspired by Phase_2_Investigator/sim_env/recorder.py; adds detector, task and applied-command
provenance and stores raw float depth without a lossy millimetre conversion.
"""
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image


def clean(obj):
    if is_dataclass(obj):
        return clean(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list, np.ndarray)):
        return [clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


class TrialRecorder:
    def __init__(self, path, config, model_sha256):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        for name in ("rgb", "depth", "lidar", "inspection"):
            (self.path / name).mkdir()
        self.started = time.monotonic()
        self.events = (self.path / "events.jsonl").open("x")
        self.states = (self.path / "states.csv").open("x", newline="")
        self.writer = None
        self.count = 0
        self.meta = {"format": "tyler-object-search-1", "started_utc": datetime.now(timezone.utc).isoformat(),
                     "simulation_only": True, "clock": "synchronized simulation seconds; wall_elapsed_s separate",
                     "config": config, "model_sha256": model_sha256, "status": "running",
                     "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                     "controller": "shared physical scripted trot (no trained high-level policy)",
                     "depth": "optical z, float32 metres; inf is invalid; raycast 160x120 upsampled to RGB"}
        write_json(self.path / "metadata.json", self.meta)

    def record(self, frame, scan, state, detections, decision, requested, applied, safety, contacts, inference_s):
        if max(abs(frame.t-state.t), abs(scan.t-state.t), abs(state.imu.t-state.t)) > 1e-5:
            raise ValueError("Sensors not synchronized")
        i = self.count
        Image.fromarray(frame.rgb).save(self.path / "rgb" / f"{i:05d}.png")
        np.savez_compressed(self.path / "depth" / f"{i:05d}.npz", depth_m=frame.depth_m,
                            K=frame.K, T_world_cam=frame.T_world_cam, timestamp=frame.t)
        np.savez_compressed(self.path / "lidar" / f"{i:05d}.npz", ranges=scan.ranges,
                            points_lidar=scan.points_lidar, T_world_lidar=scan.T_world_lidar,
                            angle_min=scan.angle_min, angle_increment=scan.angle_increment,
                            range_min=scan.range_min, range_max=scan.range_max, timestamp=scan.t)
        row = {"frame": i, "t": state.t, "camera_t": frame.t, "lidar_t": scan.t, "imu_t": state.imu.t,
               "wall_elapsed_s": time.monotonic()-self.started, "x": state.position[0],
               "y": state.position[1], "z": state.position[2], "yaw": state.yaw,
               "upright": state.upright, "state": decision.state.value, "reason": decision.reason,
               "contacts": len(contacts), "inference_s": inference_s,
               "detections": len(detections), "safety_limited": bool(safety["limited"])}
        for key, values, labels in (("imu_gyro", state.imu.gyro, "xyz"), ("imu_acc", state.imu.accel, "xyz"),
                                     ("quat", state.quat_wxyz, ("w", "x", "y", "z")),
                                     ("velocity_world", state.lin_vel_world, "xyz"),
                                     ("command", applied, ("vx", "vy", "wz")),
                                     ("requested", requested, ("vx", "vy", "wz"))):
            row.update({f"{key}_{lab}": v for lab, v in zip(labels, values)})
        if self.writer is None:
            self.writer = csv.DictWriter(self.states, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(clean(row))
        event = {"frame": i, "t": frame.t, "detections": detections, "decision": decision,
                 "safety": safety, "contacts": contacts, "command_applied": applied}
        self.events.write(json.dumps(clean(event), allow_nan=False) + "\n")
        self.states.flush()
        self.events.flush()
        self.count += 1

    def finish(self, summary):
        self.states.close()
        self.events.close()
        self.meta.update(status="finished", frames=self.count, outcome=summary)
        write_json(self.path / "metadata.json", self.meta)
        write_json(self.path / "summary.json", summary)
        files = sorted(p for p in self.path.rglob("*") if p.is_file() and p.name != "sha256.json")
        write_json(self.path / "sha256.json", {str(p.relative_to(self.path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
