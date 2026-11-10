"""Bounded, stationary Go2-camera YOLO gate. No navigation is enabled here."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import yaml

from .adapters.shared import Go2Simulation, WORKSPACE
from .adapters.yolo import ChairDetector


def annotated(rgb, detections):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for d in detections:
        draw.rectangle(d.bbox, outline="lime", width=2)
        draw.text((d.bbox[0], max(0, d.bbox[1] - 12)), f"{d.name} {d.conf:.3f}", fill="lime")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaw", type=float, default=1.3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load((WORKSPACE / "config/chair.yaml").read_text())
    cfg["simulation"]["spawn"][2] = args.yaw
    args.out.mkdir(parents=True, exist_ok=False)
    detector = ChairDetector(cfg["detector"]["confidence"])
    sim = Go2Simulation(cfg["simulation"])
    try:
        f, scan, state, contacts = sim.sample()
        detections = detector.detect_rgb(f.rgb)
        Image.fromarray(f.rgb).save(args.out / "camera.png")
        annotated(f.rgb, detections).save(args.out / "detections.png")
        np.savez_compressed(args.out / "depth.npz", depth_m=f.depth_m, K=f.K, T_world_cam=f.T_world_cam)
        np.savez_compressed(args.out / "lidar.npz", ranges=scan.ranges, points_lidar=scan.points_lidar,
                            T_world_lidar=scan.T_world_lidar)
        report = {"mode": "stationary_yolo_gate", "passed": bool(detections), "config": cfg,
                  "simulation_time_s": f.t, "actual_pose": state.pose2d.tolist(),
                  "upright": state.upright, "contacts": len(contacts), "model_sha256": detector.sha256,
                  "inference_s": detector.last_latency_s, "detections": [asdict(d) for d in detections]}
        (args.out / "gate.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        sim.close()
    raise SystemExit(0 if detections else 2)


if __name__ == "__main__":
    main()
