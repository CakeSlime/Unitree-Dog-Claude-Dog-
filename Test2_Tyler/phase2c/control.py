"""Measured uncertainty guards and sensor-only exploration for Phase 2C."""
import numpy as np
from ..q1_vision.exploration import ExploringMission,SensorExplorer
from ..q1_vision.mission import State
from .perception import CalibratedTarget,inspection_valid


class CalibratedMission(ExploringMission):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.candidate_seen=-np.inf
        from .approach import MappedApproach
        self.approach_route=MappedApproach()

    def update(self,state,frame_time,observations,contacts,blocked,discoveries=()):
        observations=[o for o in observations if isinstance(o,CalibratedTarget)
                      and o.source=='registered_RGBD' and np.isfinite(o.uncertainty_m)
                      and .025<=o.uncertainty_m<=.075]
        # Phase 2C uses three consecutive fresh/stable frames in both real-sensor
        # and proxy tasks. Keep Phase 2B's temporal-window behavior unchanged there.
        self.inspection_votes=[]
        d=super().update(state,frame_time,observations,contacts,blocked)
        if d.target is not None:self.candidate_seen=state.t
        if d.reason=='target_lost_recovery':
            # The inherited eight-second blacklist was designed for unreliable
            # Phase 2B depth. Do not blacklist a newly calibrated reobservation;
            # it must still earn three fresh confirmations after the loss stop.
            self.rejected=[];self.candidate_seen=-np.inf
        partial=[x for x in discoveries if getattr(x,'edge_clipped',False)]
        if d.state==State.SEARCH and partial:
            box=max(partial,key=lambda x:x.conf).bbox
            # The runner supplies measured camera intrinsics, never a target pose.
            cx,fx=self.camera_intrinsics
            bearing=-np.arctan2((box[0]+box[2])/2-cx,fx)
            d.command=(0.,0.,float(np.clip(1.5*bearing,-.35,.35)))
            d.reason='partial_yolo_centering_only'
        if d.state==State.SEARCH and state.t-self.candidate_seen<=1.2:
            d.command=(0.,0.,0.);d.reason='candidate_reacquisition_hold'
        if d.state in (State.INSPECT,State.COMPLETE) and not inspection_valid(d.target,self.visual.c):
            self.visual.state=d.state=State.APPROACH;self.visual.reason='range_uncertainty_guard'
            self.visual.inspect_hits=0;self.inspection_votes=[]
            d.reason='range_uncertainty_guard'
            distance=d.target.range_m if d.target else 0.
            d.command=(min(.06,.45*max(0.,distance-self.visual.c['standoff_m'])) if not blocked else 0.,0.,0.)
        if (d.state==State.APPROACH and d.target is not None and d.target.range_m>1.15
                and hasattr(self.explorer,'grid')):
            command=self.approach_route.command(state,d.target,self.explorer.grid)
            if command is not None:
                d.command=command;d.reason='mapped_visual_approach'
        return d


class ProgressExplorer(SensorExplorer):
    """Keep shared frontier planning, but abandon vantages with no measured progress."""
    def __init__(self,position,config):
        super().__init__(position,config)
        self.progress_position=np.asarray(position[:2]).copy();self.progress_t=None
        self.scan_start=None

    def command(self,state,blocked=False):
        # The previous 1.5-second vantage turn covered only ~26 degrees and could
        # pass a chair on the edge without ever bringing it into a full frame.
        if self.goal is not None and np.linalg.norm(state.position[:2]-self.goal)<.25:
            self.scan_start=state.t
            self.blacklist.append(self.goal.copy())
        if self.scan_start is not None and state.t-self.scan_start<6.:
            self.goal=None;self.path=None
            return (0.,0.,-.35),'vantage_camera_sweep'
        self.scan_start=None
        if self.progress_t is None:self.progress_t=state.t
        if np.linalg.norm(state.position[:2]-self.progress_position)>.12:
            self.progress_position=state.position[:2].copy();self.progress_t=state.t
        stalled=state.t-self.progress_t>10.
        if stalled:
            self.progress_t=state.t
        return super().command(state,blocked or stalled)
