"""Future PPO observation boundary. This module does not train or load a policy.

34 floats: 24 measured lidar ranges, 5 target features, 3 measured body velocities,
2 previous applied actions. No hidden target coordinates, scene IDs or scoring labels.
"""
import numpy as np


def observable_vector(ranges, target, body_velocity, previous_action):
    ranges = np.asarray(ranges, float)
    if ranges.ndim != 1 or len(ranges) < 24:
        raise ValueError("At least 24 lidar measurements required")
    lid = ranges[np.linspace(0, len(ranges), 24, endpoint=False).astype(int)]
    lid = np.nan_to_num(lid, nan=0., posinf=3., neginf=0.) / 3.
    features = [0., 0., 0., 0., 0.] if target is None else [1., target.confidence, target.bearing/np.pi, target.range_m/5., 0.]
    velocity = np.asarray(body_velocity, float) / [.4, .15, .5]
    previous = np.asarray(previous_action, float) / [.18, .35]
    if velocity.shape != (3,) or previous.shape != (2,):
        raise ValueError("Expected 3 body velocities and 2 previous actions")
    vector = np.concatenate([lid, features, velocity, previous]).astype(np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Nonfinite observation")
    return np.clip(vector, -1., 1.)
