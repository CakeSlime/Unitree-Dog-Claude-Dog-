"""Observable foreground-component association; no scene semantics/mesh queries.

YOLO boxes are not segmentation masks. Find an interior, vertically supported
near-depth component in the upper box, separated from local background. Reject
border-filling foreground occluders, missing/mixed depth and temporal jumps.
"""
from dataclasses import dataclass
import math
import cv2
import numpy as np
from ..q1_vision.search_observations import SearchTarget


@dataclass(frozen=True)
class CalibratedTarget(SearchTarget):
    uncertainty_m:float=.03
    depth_spread_m:float=0.
    source:str='registered_RGBD'


class DepthAssociation:
    def __init__(self):self.previous=None

    def measure(self,frame,detections):
        accepted=[];diagnostics=[];h,w=frame.depth_m.shape
        for d in detections:
            if d.name!='chair' or d.source!='yolo':continue
            log={'bbox':d.bbox,'accepted':False};diagnostics.append(log)
            if getattr(frame,'depth_source',None)!='registered_RGBD':
                log['reason']='unregistered_depth_source';continue
            if getattr(d,'edge_clipped',False):
                log['reason']='edge_clipped_discovery_only';continue
            x1,y1,x2,y2=map(float,d.bbox);bw,bh=x2-x1,y2-y1
            xa,xb=max(0,int(x1+.03*bw)),min(w,int(x2-.03*bw)+1)
            ya,yb=max(0,int(y1+.03*bh)),min(h,int(y1+.43*bh)+1)
            roi=frame.depth_m[ya:yb,xa:xb]
            ok=np.isfinite(roi)&(roi>=.30)&(roi<=6.)
            log['valid_fraction']=float(ok.mean()) if roi.size else 0.
            if not roi.size or ok.mean()<.85 or ok.sum()<30:
                log['reason']='invalid_or_near_depth';continue
            values=np.sort(roi[ok]);groups=np.split(values,np.flatnonzero(np.diff(values)>.12)+1)
            selected=None;failure='no_supported_foreground_component'
            background=float(np.quantile(values,.95))
            for group in groups:
                if len(group)<max(30,.08*roi.size):continue
                depth=float(np.median(group))
                if background-depth<.12:continue
                mask=(ok&(roi>=group[0])&(roi<=group[-1])).astype(np.uint8)
                count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
                for k in range(1,count):
                    x,y,cw,ch,area=stats[k]
                    if area<max(30,.08*roi.size) or cw<.07*bw or ch<.12*bh:continue
                    if x<=1 and x+cw>=roi.shape[1]-1:
                        failure='foreground_occlusion';break
                    comp=(labels==k).astype(np.uint8)
                    interior=cv2.distanceTransform(comp,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                    yy,xx=np.unravel_index(np.argmax(interior),interior.shape)
                    if interior[yy,xx]<2.5:continue
                    selected=(float(xa+xx),float(ya+yy),int(area),depth,background)
                    break
                if failure=='foreground_occlusion' or selected is not None:break
            if selected is None:
                log['reason']=failure;continue
            u,v,support,component_depth,background=selected
            patch=frame.depth_m[max(0,int(v)-2):min(h,int(v)+3),max(0,int(u)-2):min(w,int(u)+3)]
            valid=np.isfinite(patch)&(patch>=.30)&(patch<=6.)
            if valid.mean()<.85:
                log['reason']='invalid_interior_depth';continue
            z=float(np.median(patch[valid]));spread=float(np.quantile(patch[valid],.9)-np.quantile(patch[valid],.1))
            log.update(sample_pixel=[u,v],depth_z_m=z,spread_m=spread,component_support=support,
                       background_z_m=background,component_z_m=component_depth)
            if spread>.10:
                log['reason']='mixed_interior_depth';continue
            world=frame.backproject(u,v,z)
            distance=float(np.linalg.norm(world[:2]-frame.T_world_cam[:2,3]))
            uncertainty=max(.025,spread/2+.015)
            if not np.isfinite([*world,distance,uncertainty]).all() or not .30<=distance<=6.:
                log['reason']='invalid_horizontal_range';continue
            if self.previous is not None:
                t,xyz=self.previous
                # An interior pixel may move across the SAME chair as viewpoint
                # changes. Bound that movement by observable apparent object size,
                # capped at the mission's 0.65 m identity gate, rather than treating
                # every surface point as an immovable landmark.
                limit=max(.30,min(.65,.60*bw*z/frame.K[0,0]+.10))
                log['world_consistency_limit_m']=limit
                if 0<frame.t-t<=.6 and np.linalg.norm(world-xyz)>limit:
                    log['reason']='surface_jump';continue
            target=CalibratedTarget(frame.t,d.conf,tuple(d.bbox),-math.atan2(u-frame.K[0,2],frame.K[0,0]),
                         distance,tuple(map(float,world)),support,(u,v),uncertainty,spread)
            accepted.append(target);log.update(accepted=True,reason='valid',range_m=distance,uncertainty_m=uncertainty)
        if accepted:
            best=max(accepted,key=lambda x:x.confidence);self.previous=(frame.t,np.asarray(best.world_xyz))
        return accepted,diagnostics


def inspection_valid(target,config):
    return (isinstance(target,CalibratedTarget) and target.source=='registered_RGBD'
            and np.isfinite([target.range_m,target.uncertainty_m,target.bearing]).all()
            and target.uncertainty_m>=.025
            and abs(target.range_m-config['standoff_m'])+target.uncertainty_m<=config['distance_tolerance_m']
            and abs(target.bearing)<=.10)
