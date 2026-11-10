"""Tyler's pure task state machine. No simulator, hidden target or trained policy is imported.

Approach pattern is adapted from go2_stream/behavior/control.py (turn, then bounded forward
motion). This task adds consecutive confirmation, world-observation identity gating and
inspection evidence readiness; it does not execute the group's approach-and-act actions.
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np


class State(str, Enum):
    SEARCH = "search"
    CONFIRM = "confirm"
    APPROACH = "approach"
    INSPECT = "inspect"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Decision:
    state: State
    command: tuple = (0., 0., 0.)
    reason: str = ""
    target: object = None


class SearchMission:
    def __init__(self, config, start_time=0.):
        self.c = config
        positive = ("confirm_frames", "inspect_frames", "standoff_m", "distance_tolerance_m",
                    "max_forward_mps", "max_yaw_rps", "scan_yaw_rps", "max_time_s", "target_lost_s",
                    "blocked_timeout_s", "continuity_gate_m", "max_frame_age_s")
        if any(not np.isfinite(config[k]) or config[k] <= 0 for k in positive):
            raise ValueError("Task limits must be finite and positive")
        if config["max_forward_mps"] > .3 or config["max_yaw_rps"] > .5 or config["scan_yaw_rps"] > config["max_yaw_rps"]:
            raise ValueError("Task speeds exceed simulation validation limits")
        if config["standoff_m"] - config["distance_tolerance_m"] < .6:
            raise ValueError("Inspection band too close to the target")
        tolerance = config.get("deadline_tolerance_s", 0.)
        if not np.isfinite(tolerance) or not 0 <= tolerance <= 1e-6:
            raise ValueError("Deadline tolerance must be at most one microsecond")
        self.start = start_time
        self.state, self.reason = State.SEARCH, "scanning"
        self.anchor = None
        self.last_seen = start_time
        self.hits = self.inspect_hits = 0
        self.blocked_since = None
        self.last_frame_time = -np.inf

    @property
    def terminal(self):
        return self.state in (State.COMPLETE, State.FAILED)

    def fail(self, reason):
        self.state, self.reason = State.FAILED, reason
        return Decision(self.state, reason=reason)

    def update(self, now, frame_time, observations, contacts=0, upright=1., blocked=False, speed_mps=0.):
        if self.terminal:
            return Decision(self.state, reason=self.reason)
        if not np.isfinite([now, frame_time, upright, speed_mps]).all():
            return self.fail("invalid_sensor_state")
        if contacts:
            return self.fail("robot_scene_contact")
        if upright < .92:
            return self.fail("excessive_tilt")
        # Old recorded configs default to zero, preserving exact replay of their decisions.
        if now - self.start >= self.c["max_time_s"] - self.c.get("deadline_tolerance_s", 0.):
            return self.fail("task_timeout")
        if frame_time > now + 1e-5 or now - frame_time > self.c["max_frame_age_s"]:
            return self.fail("stale_or_future_frame")
        if frame_time <= self.last_frame_time:
            return self.fail("non_increasing_frame_timestamp")
        self.last_frame_time = frame_time
        if blocked:
            self.blocked_since = now if self.blocked_since is None else self.blocked_since
            if now - self.blocked_since >= self.c["blocked_timeout_s"]:
                return self.fail("blocked_path")
        else:
            self.blocked_since = None
        valid = [o for o in observations if np.isfinite([o.range_m, o.bearing, *o.world_xyz, o.confidence]).all()
                 and o.confidence >= self.c.get("min_confidence", .3)
                 and o.range_m > 0 and abs(o.timestamp-frame_time) < 1e-5]
        if self.anchor is not None:
            valid = [o for o in valid if np.linalg.norm(np.array(o.world_xyz)-self.anchor) <= self.c["continuity_gate_m"]]
        target = max(valid, key=lambda o: o.confidence) if valid else None
        if target is None:
            self.hits = self.inspect_hits = 0
            if self.state in (State.APPROACH, State.INSPECT):
                if now - self.last_seen >= self.c["target_lost_s"]:
                    return self.fail("target_lost")
                return Decision(self.state, reason="target_missing_stop")
            self.state = State.SEARCH
            self.anchor = None
            return Decision(self.state, (0., 0., self.c["scan_yaw_rps"]), "scanning")
        if self.anchor is None:
            self.anchor = np.array(target.world_xyz)
        self.last_seen = now
        self.hits += 1
        if self.state in (State.SEARCH, State.CONFIRM):
            self.state = State.CONFIRM
            if self.hits < self.c["confirm_frames"]:
                return Decision(self.state, reason="confirming", target=target)
            self.state = State.APPROACH
        lo, hi = self.c["standoff_m"]-self.c["distance_tolerance_m"], self.c["standoff_m"]+self.c["distance_tolerance_m"]
        if target.range_m < lo:
            return self.fail("target_too_close")
        if target.range_m <= hi and abs(target.bearing) <= .10:
            self.state = State.INSPECT
            self.inspect_hits = self.inspect_hits + 1 if speed_mps < .06 else 0
            if self.inspect_hits >= self.c["inspect_frames"]:
                self.state, self.reason = State.COMPLETE, "inspection_ready"
            return Decision(self.state, reason=self.reason if self.terminal else "holding_for_inspection", target=target)
        self.state = State.APPROACH
        self.inspect_hits = 0
        yaw = float(np.clip(1.5*target.bearing, -self.c["max_yaw_rps"], self.c["max_yaw_rps"]))
        vx = min(self.c["max_forward_mps"], .45*max(0, target.range_m-self.c["standoff_m"])) if abs(target.bearing) < .12 else 0.
        return Decision(self.state, (0. if blocked else vx, 0., yaw), "visual_approach", target)
