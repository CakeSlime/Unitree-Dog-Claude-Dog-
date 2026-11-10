"""Map-routed approach to an OBSERVED surface; no semantic simulator access.

Uses the attributed Phase 2B occupancy/Dijkstra adaptation. Select reachable
standoff-ring cells around the current depth observation, then follow the safe
route. No command is returned when the caller has no current valid target.
"""
import numpy as np
from ..q1_vision.sensor_map import CostField


class MappedApproach:
    def __init__(self):
        self.path=None;self.next_plan=0.;self.goals=[]

    def command(self,state,target,grid):
        if target is None:return None
        if self.path is None or state.t>=self.next_plan:
            cost=grid.costmap(robot_radius=.42)
            field=CostField(grid,cost,state.position[:2]);point=np.asarray(target.world_xyz[:2])
            angle=np.arctan2(state.position[1]-point[1],state.position[0]-point[0])
            best=None
            for offset in np.linspace(-np.pi/2,np.pi/2,13):
                goal=point+1.30*np.array([np.cos(angle+offset),np.sin(angle+offset)])
                r,c=map(int,grid.to_cell(goal))
                if not grid.inside(r,c) or not np.isfinite(field.cum[r,c]):continue
                score=field.cum[r,c]*grid.res+.1*abs(offset)
                if best is None or score<best[0]:best=(score,(r,c),goal)
            self.path=None if best is None else field.path_to(best[1]);self.next_plan=state.t+1.
            if best is not None:self.goals.append({'t':state.t,'observed_surface':point.tolist(),'goal':best[2].tolist()})
        if self.path is None:return None
        distances=np.linalg.norm(self.path-state.position[:2],axis=1);i=int(np.argmin(distances))
        while i<len(self.path)-1 and distances[i]<.25:i+=1
        delta=self.path[i]-state.position[:2]
        if np.linalg.norm(delta)<.10:return None
        error=float((np.arctan2(delta[1],delta[0])-state.yaw+np.pi)%(2*np.pi)-np.pi)
        return (.14 if abs(error)<.30 else 0.,0.,float(np.clip(1.5*error,-.35,.35)))
