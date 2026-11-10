"""Read-only recording verification and task replay, optionally repeating actual YOLO inference.

Writes a NEW report path outside the recording. Does not create a MuJoCo session or advance physics.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..adapters.shared import configure
from ..adapters.yolo import ChairDetector
from ..q1_vision.mission import SearchMission
from ..q1_vision.observations import targets_from_depth
from .recording import write_json
from go2_stream.perception.detector import Detection


def verify(trial, repeat_yolo=False):
    trial = Path(trial)
    hashes = json.loads((trial / "sha256.json").read_text())
    for name, expected in hashes.items():
        actual = hashlib.sha256((trial / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Recording checksum mismatch: {name}")
    metadata = json.loads((trial / "metadata.json").read_text())
    events = [json.loads(x) for x in (trial / "events.jsonl").read_text().splitlines()]
    with (trial / "states.csv").open() as f:
        states = list(csv.DictReader(f))
    assert len(events) == len(states) == metadata["frames"]
    configure()
    from sim_env.interfaces import CameraFrame
    mission = SearchMission(metadata["config"]["task"], start_time=float(states[0]["t"]))
    detector = ChairDetector(metadata["config"]["detector"]["confidence"]) if repeat_yolo else None
    blocked = False
    intervals, valid_depth, valid_lidar = [], [], []
    for i, (row, event) in enumerate(zip(states, events)):
        assert int(row["frame"]) == event["frame"] == i
        now = float(row["t"])
        if i:
            intervals.append(now-float(states[i-1]["t"]))
        assert all(abs(float(row[k])-now)<1e-5 for k in ("camera_t", "lidar_t", "imu_t"))
        rgb = np.asarray(Image.open(trial / "rgb" / f"{i:05d}.png").convert("RGB"))
        with np.load(trial / "depth" / f"{i:05d}.npz", allow_pickle=False) as d:
            assert abs(float(d["timestamp"])-now) < 1e-5
            assert rgb.shape[:2] == d["depth_m"].shape
            frame = CameraFrame(now, rgb, d["depth_m"], d["K"], d["T_world_cam"])
            valid_depth.append(float(np.isfinite(frame.depth_m).mean()))
        with np.load(trial / "lidar" / f"{i:05d}.npz", allow_pickle=False) as l:
            assert abs(float(l["timestamp"])-now) < 1e-5
            assert l["ranges"].shape == (720,) and l["points_lidar"].shape[1] == 3
            valid_lidar.append(int(np.isfinite(l["ranges"]).sum()))
        detections = [Detection(**d) for d in event["detections"]]
        if detector:
            actual = detector.detect_rgb(rgb)
            assert len(actual) == len(detections), f"YOLO count mismatch at frame {i}"
            for a, b in zip(actual, detections):
                assert abs(a.conf-b.conf)<1e-5 and np.allclose(a.bbox, b.bbox, atol=1e-3), f"YOLO mismatch {i}"
        targets = targets_from_depth(frame, detections)
        speed = np.linalg.norm([float(row["velocity_world_x"]), float(row["velocity_world_y"])])
        decision = mission.update(now, now, targets, int(row["contacts"]), float(row["upright"]), blocked, speed)
        assert decision.state.value == row["state"] and decision.reason == row["reason"], f"Task replay mismatch {i}"
        assert np.allclose(decision.command, event["decision"]["command"])
        applied = np.asarray(event["command_applied"])
        blocked = bool(np.linalg.norm(decision.command)>.02 and np.linalg.norm(applied)<.005)
    expected_dt = metadata["config"]["simulation"]["dt"]
    assert np.allclose(intervals, expected_dt)
    summary = metadata["outcome"]
    assert summary["final_state"] == mission.state.value
    if summary["success"]:
        assert (trial / "inspection/rgb.png").exists() and (trial / "inspection/observation.json").exists()
        assert np.allclose(events[-1]["command_applied"], 0.)
    return {"trial": str(trial), "frames": len(states), "checksum_files_verified": len(hashes),
            "timestamps_aligned": True, "sim_interval_s": expected_dt,
            "task_states_and_requested_commands_reproduced": True,
            "yolo_reinference_matches": True if detector else None,
            "minimum_valid_depth_fraction": min(valid_depth), "minimum_lidar_scan_returns": min(valid_lidar),
            "success": summary["success"], "final_state": mission.state.value}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeat-yolo", action="store_true")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(f"Preserving prior verification report: {a.out}")
    result = verify(a.trial, a.repeat_yolo)
    write_json(a.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
