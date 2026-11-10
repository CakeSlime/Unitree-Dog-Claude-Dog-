"""Phase 2B depth sampling of the chair upper body, avoiding foreground beds/seat holes.

Adapted from preserved Phase 2A observations.py. Still an RGB-box heuristic,
not segmentation: occlusion/range errors remain measurable failure cases.
"""
from dataclasses import dataclass
import math
import numpy as np
from .observations import TargetObservation


@dataclass(frozen=True)
class SearchTarget(TargetObservation):
    sample_pixel:tuple=(0.,0.)


def search_targets(frame,detections):
    h,w=frame.depth_m.shape
    out=[]
    for d in detections:
        if d.name!='chair' or d.source!='yolo':continue
        x1,y1,x2,y2=d.bbox
        u=(x1+x2)/2;v=y1+.25*(y2-y1)
        dx=max(2,(x2-x1)*.10);dy=max(2,(y2-y1)*.06)
        patch=frame.depth_m[max(0,int(v-dy)):min(h,int(v+dy)+1),max(0,int(u-dx)):min(w,int(u+dx)+1)]
        z=patch[np.isfinite(patch)&(patch>.15)&(patch<8.)]
        if len(z)<6:continue
        z0=float(np.median(z));cam_x=(u-frame.K[0,2])/frame.K[0,0]*z0
        world=frame.backproject(u,v,z0)
        out.append(SearchTarget(frame.t,d.conf,tuple(d.bbox),-math.atan2(cam_x,z0),math.hypot(cam_x,z0),
                                tuple(map(float,world)),len(z),(u,v)))
    return out
