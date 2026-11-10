"""Meaningful task/geometry/data contracts; no model download or simulator is needed."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from Test2_Tyler.adapters.shared import configure, WORKSPACE
from Test2_Tyler.adapters.yolo import decode
from Test2_Tyler.q1_vision.mission import SearchMission, State
from Test2_Tyler.q1_vision.observations import TargetObservation, targets_from_depth
from Test2_Tyler.q2_data_simulation.recording import TrialRecorder
from Test2_Tyler.q2_data_simulation.replay import verify
from Test2_Tyler.q3_rl.observations import observable_vector
from Test2_Tyler.run import check_gate
from Test2_Tyler.adapters.yolo import MODEL_SHA256
from go2_stream.perception.detector import Detection

configure()
from sim_env.interfaces import CameraFrame, RobotState, ImuReading, LidarScan


@pytest.fixture
def config():
    return yaml.safe_load((WORKSPACE / "config/chair.yaml").read_text())


def target(t, distance=2., xyz=(4., 3., .5)):
    return TargetObservation(t, .8, (30., 20., 80., 90.), 0., distance, xyz, 100)


def confirmed(m):
    for t in (0., .2, .4):
        d = m.update(t, t, [target(t)])
    assert d.state == State.APPROACH
    return d


def test_model_output_coordinates_class_and_fragment_filter():
    out = decode([[200, 100, 400, 400, .8, 56], [200, 100, 400, 400, .9, 57],
                  [0, 10, 20, 100, .99, 56]], 320, 240)
    assert len(out) == 1 and out[0].name == "chair" and out[0].source == "yolo"
    assert out[0].bbox == (100., 50., 200., 200.)


def frame(t=0.):
    return CameraFrame(t, np.zeros((120, 160, 3), np.uint8), np.full((120, 160), 2., np.float32),
                       np.array([[100, 0, 80], [0, 100, 60], [0, 0, 1.]]), np.eye(4))


def test_depth_geometry_and_invalid_returns():
    f = frame()
    det = Detection(56, "chair", .8, (40., 20., 80., 100.))
    obs = targets_from_depth(f, [det])[0]
    assert obs.bearing > 0 and obs.range_m == pytest.approx(np.hypot(.4, 2.))
    assert obs.world_xyz == pytest.approx((-.4, 0., 2.))
    f.depth_m[:] = np.inf
    assert targets_from_depth(f, [det]) == []
    assert targets_from_depth(frame(), [replace(det, source="ground_truth")]) == []


def test_consecutive_confirmation_and_no_unconfirmed_forward_motion(config):
    m = SearchMission(config["task"])
    d = m.update(0., 0., [target(0.)])
    assert d.state == State.CONFIRM and d.command == (0., 0., 0.)
    assert m.update(.2, .2, []).state == State.SEARCH
    assert m.update(.4, .4, [target(.4)]).state == State.CONFIRM
    assert m.update(.6, .6, [target(.6)]).state == State.CONFIRM
    assert m.update(.8, .8, [target(.8)]).state == State.APPROACH


def test_same_class_target_cannot_switch_and_missing_target_stops(config):
    m = SearchMission(config["task"])
    confirmed(m)
    d = m.update(.6, .6, [target(.6, xyz=(7., 3., .5))])
    assert d.command == (0., 0., 0.) and d.reason == "target_missing_stop"
    d = m.update(2.6, 2.6, [])
    assert d.state == State.FAILED and d.reason == "target_lost"


@pytest.mark.parametrize("kwargs,reason", [({"contacts": 1}, "robot_scene_contact"),
                                            ({"upright": .8}, "excessive_tilt"),
                                            ({"speed_mps": float("nan")}, "invalid_sensor_state")])
def test_stop_on_sensor_or_contact_failure(config, kwargs, reason):
    m = SearchMission(config["task"])
    confirmed(m)
    d = m.update(.6, .6, [target(.6)], **kwargs)
    assert d.state == State.FAILED and d.reason == reason and d.command == (0., 0., 0.)


def test_repeated_or_stale_frame_cannot_confirm(config):
    m = SearchMission(config["task"])
    m.update(0., 0., [target(0.)])
    assert m.update(.2, 0., [target(0.)]).reason == "non_increasing_frame_timestamp"
    assert SearchMission(config["task"]).update(1., 0., [target(0.)]).reason == "stale_or_future_frame"


def test_too_close_timeout_and_invalid_limits(config):
    m = SearchMission(config["task"])
    confirmed(m)
    assert m.update(.6, .6, [target(.6, distance=.3)]).reason == "target_too_close"
    m = SearchMission(config["task"])
    assert m.update(36., 36., []).reason == "task_timeout"
    c = dict(config["task"], max_forward_mps=1.)
    with pytest.raises(ValueError):
        SearchMission(c)


def test_inspection_requires_stopped_consecutive_frames(config):
    m = SearchMission(config["task"])
    confirmed(m)
    for t in (.6, .8, 1.):
        d = m.update(t, t, [target(t, distance=.95)], speed_mps=.1)
        assert d.state == State.INSPECT and d.command == (0., 0., 0.)
    for t in (1.2, 1.4, 1.6):
        d = m.update(t, t, [target(t, distance=.95)], speed_mps=.01)
    assert d.state == State.COMPLETE and m.terminal
    assert m.update(2., 2., []).command == (0., 0., 0.)


def test_blocked_path_terminates(config):
    m = SearchMission(config["task"])
    confirmed(m)
    assert m.update(.6, .6, [target(.6)], blocked=True).command[0] == 0
    assert m.update(2.8, 2.8, [target(2.8)], blocked=True).reason == "blocked_path"


def test_float_roundoff_does_not_delay_deadline(config):
    c = dict(config["task"], max_time_s=1.)
    m = SearchMission(c, start_time=1.500000000000001)
    assert m.update(2.5, 2.5, []).reason == "task_timeout"


def test_future_rl_vector_does_not_expose_hidden_or_world_target_position():
    t = target(0.)
    a = observable_vector(np.full(720, 3.), t, [0, 0, 0], [0, 0])
    b = observable_vector(np.full(720, 3.), replace(t, world_xyz=(999, 999, 999)), [0, 0, 0], [0, 0])
    assert a.shape == (34,) and a.dtype == np.float32 and np.array_equal(a, b)
    assert np.isfinite(observable_vector(np.full(720, np.inf), None, [0, 0, 0], [0, 0])).all()


def test_recorder_sensor_alignment_roundtrip_and_no_overwrite(tmp_path, config):
    path = tmp_path / "trial"
    r = TrialRecorder(path, config, "test-model")
    f = frame()
    state = RobotState(0., np.array([0., 0., .3]), np.array([1., 0, 0, 0]), 0., np.zeros(3),
                       np.zeros(3), np.zeros(3), ImuReading(0., np.zeros(3), np.array([0., 0., 9.81]), np.array([1., 0, 0, 0])), 1.)
    scan = LidarScan(0., np.ones(720), -np.pi, 2*np.pi/720, .1, 30., np.ones((20, 3)), np.eye(4))
    m = SearchMission(config["task"])
    d = m.update(0., 0., [])
    with pytest.raises(ValueError, match="synchronized"):
        r.record(f, replace(scan, t=.5), state, [], d, d.command, d.command, {"limited": False}, [], .01)
    r.record(f, scan, state, [], d, d.command, d.command, {"limited": False}, [], .01)
    r.finish({"success": False, "reason": "unit_test", "final_state": "search"})
    assert (path / "rgb/00000.png").exists()
    with np.load(path / "depth/00000.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["depth_m"], f.depth_m)
    assert json.loads((path / "metadata.json").read_text())["frames"] == 1
    before = (path / "metadata.json").read_bytes()
    with pytest.raises(FileExistsError):
        TrialRecorder(path, config, "different-model")
    assert (path / "metadata.json").read_bytes() == before
    assert verify(path)["timestamps_aligned"]
    with (path / "states.csv").open("a") as f:
        f.write("tamper\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify(path)


def test_low_confidence_cannot_confirm(config):
    m = SearchMission(config["task"])
    d = m.update(0., 0., [replace(target(0.), confidence=.2)])
    assert d.state == State.SEARCH and m.anchor is None


def test_navigation_gate_refuses_failed_or_different_scene(tmp_path, config):
    p = tmp_path / "gate.json"
    gate = {"passed": False, "model_sha256": MODEL_SHA256,
            "config": config, "detections": [{"name": "chair", "conf": .8}]}
    p.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="successful stationary"):
        check_gate(p, config)
    gate["passed"] = True
    p.write_text(json.dumps(gate))
    assert len(check_gate(p, config)) == 64
    different = dict(config, simulation=dict(config["simulation"], chair_pose=[1., 2., 0.]))
    with pytest.raises(ValueError, match="scene differ"):
        check_gate(p, different)
