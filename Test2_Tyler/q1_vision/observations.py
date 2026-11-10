"""Robot-observable target estimates from YOLO boxes and registered optical-z depth."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class TargetObservation:
    timestamp: float
    confidence: float
    bbox: tuple
    bearing: float             # positive left, from the optical camera
    range_m: float             # horizontal distance from camera to observed surface
    world_xyz: tuple           # derived from visible depth, NOT hidden scene object poses
    valid_points: int


def targets_from_depth(frame, detections):
    out = []
    h, w = frame.depth_m.shape
    for d in detections:
        if d.name != "chair" or d.source != "yolo":
            continue
        x1, y1, x2, y2 = d.bbox
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = max(2, (x2 - x1) * .18), max(2, (y2 - y1) * .12)
        patch = frame.depth_m[max(0, int(v-dy)):min(h, int(v+dy)+1),
                              max(0, int(u-dx)):min(w, int(u+dx)+1)]
        z = patch[np.isfinite(patch) & (patch > .15) & (patch < 8)]
        if len(z) < 6:
            continue
        # Median of nearer half reduces background through open chair backs.
        z0 = float(np.median(z[z <= np.quantile(z, .5)]))
        cam_x = (u - frame.K[0, 2]) / frame.K[0, 0] * z0
        world = frame.backproject(u, v, depth=z0)
        if not np.isfinite(world).all():
            continue
        out.append(TargetObservation(frame.t, d.conf, tuple(d.bbox), -math.atan2(cam_x, z0),
                                     math.hypot(cam_x, z0), tuple(map(float, world)), len(z)))
    return out


def depth_obstacles(frame, stride=12):
    """Actual measured depth points, excluding floor/overhead; no scene-geometry oracle."""
    vv, uu = np.mgrid[0:frame.depth_m.shape[0]:stride, 0:frame.depth_m.shape[1]:stride]
    z = frame.depth_m[::stride, ::stride]
    valid = np.isfinite(z) & (z > .15) & (z < 5.)
    if not valid.any():
        return np.empty((0, 3))
    p = frame.backproject(uu[valid], vv[valid], z[valid])
    return p[(p[:, 2] > .07) & (p[:, 2] < .75)]
