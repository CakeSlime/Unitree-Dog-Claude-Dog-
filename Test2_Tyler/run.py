"""Bounded detector-driven simulated task. Run from repository root with python -m Test2_Tyler.run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import yaml

from .adapters.shared import Go2Simulation, WORKSPACE
from .adapters.yolo import ChairDetector, MODEL_SHA256
from .probe import annotated
from .q1_vision.mission import SearchMission, State
from .q1_vision.observations import targets_from_depth, depth_obstacles
from .q2_data_simulation.recording import TrialRecorder, write_json


def check_gate(path, config):
    gate = json.loads(Path(path).read_text())
    if not gate.get("passed") or gate.get("model_sha256") != MODEL_SHA256:
        raise ValueError("A successful stationary YOLO gate with this model is required")
    for k in ("environment", "camera_size", "chair_pose"):
        if gate["config"]["simulation"].get(k) != config["simulation"].get(k):
            raise ValueError(f"Probe and mission scene differ: {k}")
    if not any(d["name"] == "chair" and d["conf"] >= config["detector"]["confidence"]
               for d in gate["detections"]):
        raise ValueError("Probe has no chair at the mission's confidence threshold")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def demo(config, gate_path, output, wall_limit_s=90.):
    if Path(output).exists():
        raise FileExistsError(f"Preserving existing trial: {output}")
    gate_hash = check_gate(gate_path, config)
    detector = ChairDetector(config["detector"]["confidence"])
    sim = Go2Simulation(config["simulation"])
    # Reuse the shared sensor-based local safety layer (not a ground-truth cost map).
    from navigation.local import SafetyLayer, SafetyConfig
    c = config["task"]
    safety = SafetyLayer(SafetyConfig(corridor_margin=c["corridor_half_width_m"]-.155,
                                     stop_dist=c["obstacle_margin_m"]))
    recorder = None
    start_wall = time.monotonic()
    path, states, latencies, gif = [], Counter(), [], []
    interventions = contact_steps = 0
    minimum_upright = 1.
    was_blocked = False
    inspection = None
    try:
        recorder = TrialRecorder(output, config, detector.sha256)
        mission = SearchMission(c, sim.session.time)
        start_sim = sim.session.time
        previous_state = None
        while True:
            frame, scan, state, contacts = sim.sample()
            detections = detector.detect_rgb(frame.rgb)
            targets = targets_from_depth(frame, detections)
            decision = mission.update(state.t, frame.t, targets, len(contacts), state.upright,
                                      was_blocked, float(np.linalg.norm(state.lin_vel_world[:2])))
            if time.monotonic()-start_wall > wall_limit_s and not mission.terminal:
                decision = mission.fail("wall_clock_budget")
            safety.observe(scan)
            requested = decision.command
            applied = safety.filter(state, *requested, map_points=depth_obstacles(frame))
            was_blocked = bool(np.linalg.norm(requested) > .02 and np.linalg.norm(applied) < .005)
            interventions += int(safety.last["limited"])
            contact_steps += int(bool(contacts))
            minimum_upright = min(minimum_upright, state.upright)
            states[decision.state.value] += 1
            path.append(state.position[:2].copy())
            latencies.append(detector.last_latency_s)
            recorder.record(frame, scan, state, detections, decision, requested, applied,
                            safety.last, contacts, detector.last_latency_s)
            if decision.state != previous_state:
                print(f"t={state.t-start_sim:.1f}s {decision.state.value}: {decision.reason}", flush=True)
                previous_state = decision.state
            if recorder.count % 2 == 1 or mission.terminal:
                panel = annotated(frame.rgb, detections)
                panel.thumbnail((480, 360))
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, panel.width, 24), fill="black")
                draw.text((4, 6), f"t={state.t-start_sim:.1f}s | {decision.state.value} | YOLO chair", fill="white")
                gif.append(panel)
            if decision.state == State.COMPLETE:
                # Three stopped, aligned, stable target frames preceded this decision.
                Image.fromarray(frame.rgb).save(recorder.path / "inspection/rgb.png")
                annotated(frame.rgb, detections).save(recorder.path / "inspection/annotated.png")
                inspection = {"frame": recorder.count-1, "timestamp": frame.t, "target": decision.target,
                              "robot_pose": state.pose2d, "speed_mps": float(np.linalg.norm(state.lin_vel_world[:2])),
                              "method": "YOLO + registered depth; no ground-truth target label/coordinate input"}
                write_json(recorder.path / "inspection/observation.json", inspection)
            if mission.terminal:
                break
            sim.step(applied, config["simulation"]["dt"])
        # Stop before analysis or file export. This does not advance physics.
        sim.session.stop()
        summary = {"success": decision.state == State.COMPLETE and inspection is not None,
                   "final_state": decision.state.value, "reason": decision.reason,
                   "frames": recorder.count, "duration_sim_s": state.t-start_sim,
                   "wall_time_s": time.monotonic()-start_wall, "states": dict(states),
                   "path_m": float(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1).sum()) if len(path)>1 else 0.,
                   "start_pose": path[0], "end_pose": state.pose2d,
                   "min_upright": minimum_upright, "contact_steps": contact_steps,
                   "safety_intervention_steps": interventions, "mean_inference_s": float(np.mean(latencies)),
                   "inspection_range_m": decision.target.range_m if decision.target else None,
                   "final_applied_command": applied, "stationary_gate_sha256": gate_hash,
                   "simulation_only": True, "policy": "Tyler classical visual state machine; PPO not trained"}
        if gif:
            gif[0].save(recorder.path / "demo.gif", save_all=True, append_images=gif[1:],
                        duration=400, loop=0)
        # Small task-trajectory figure: camera search and inspection, no oracle target plot.
        plot = Image.new("RGB", (520, 420), "white")
        d = ImageDraw.Draw(plot)
        points = [(int(40+(p[0]-4)*190), int(380-(p[1]-.5)*110)) for p in path]
        if len(points)>1:
            d.line(points, fill="blue", width=3)
        d.ellipse((points[0][0]-4, points[0][1]-4, points[0][0]+4, points[0][1]+4), fill="green")
        d.ellipse((points[-1][0]-4, points[-1][1]-4, points[-1][0]+4, points[-1][1]+4), fill="red")
        d.text((10, 10), "Measured robot path (world XY). Green=start, red=end.\nDisplay: x=4..6.5 m; y=0.5..4 m.", fill="black")
        plot.save(recorder.path / "trajectory.png")
        recorder.finish(summary)
        print(json.dumps(summary, default=lambda o: o.tolist(), indent=2), flush=True)
        return summary
    finally:
        sim.close()
        if recorder is not None:
            recorder.states.close()
            recorder.events.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=WORKSPACE / "config/chair.yaml")
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Must not exist; preserves all earlier trials")
    p.add_argument("--wall-limit", type=float, default=90.)
    a = p.parse_args()
    if a.wall_limit <= 0 or a.wall_limit > 120:
        p.error("wall-limit must be in (0,120]")
    config = yaml.safe_load(a.config.read_text())
    result = demo(config, a.gate, a.out, a.wall_limit)
    raise SystemExit(0 if result["success"] else 2)


if __name__ == "__main__":
    main()
