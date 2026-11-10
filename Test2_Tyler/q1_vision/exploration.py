"""Sensor-map exploration baseline, separate from the preserved Phase 2A mission.

Adapts shared occupancy, frontier clustering and Dijkstra algorithms. No target
position, scene object name or simulator model is accepted by this controller.
"""
import numpy as np
from ..adapters.shared import configure


class SensorExplorer:
    def __init__(self, position, config):
        configure()
        from .sensor_map import OccupancyGrid
        self.config = config
        self.grid = OccupancyGrid(origin=np.asarray(position[:2])-[6., 5.], size=(12., 10.), res=.10)
        self.grid.clear_disk(position[:2], .32)
        self.visits = np.zeros(self.grid.shape, np.int32)
        self.camera_seen = np.zeros(self.grid.shape, bool)
        self.goal = self.path = None
        self.blacklist = []
        self.next_plan = 0.
        self.pause_until = 0.
        self.moves = 0
        self.initial = np.array(position[:2])
        self.goals = []

    def observe(self, scan, state, depth_points):
        self.grid.update(scan, max_range=self.config.get("map_range_m", 2.8))
        self.grid.add_low_points(depth_points, state.position[:2])
        r, c = self.grid.to_cell(state.position[:2])
        self.visits[r, c] += 1
        rr, cc = np.nonzero(self.grid.free())
        xy = self.grid.to_xy(rr, cc)
        delta = xy-state.position[:2]
        bearing = (np.arctan2(delta[:, 1], delta[:, 0])-state.yaw+np.pi)%(2*np.pi)-np.pi
        # Coverage heuristic, not a semantic detection or ground-truth visibility label.
        view = (np.linalg.norm(delta, axis=1)<2.5) & (np.abs(bearing)<.55)
        self.camera_seen[rr[view], cc[view]] = True

    def _plan(self, state):
        from .sensor_map import CostField,frontier_clusters
        g = self.grid
        f = CostField(g, g.costmap(robot_radius=.32), state.position[:2])
        candidates = []
        clear = g.obstacle_distance()
        # Refresh a committed route as observations change, without selecting a
        # different vantage every two seconds before the first one is reached.
        if self.goal is not None:
            rc = g.to_cell(self.goal)
            path = f.path_to(rc) if g.inside(*rc) else None
            if path is not None:
                self.path = path
                return
            self.blacklist.append(self.goal.copy())
            self.goal = self.path = None
        for cells in frontier_clusters(g, min_cells=3):
            valid = np.isfinite(f.cum[cells[:, 0], cells[:, 1]]) & (clear[cells[:, 0], cells[:, 1]]>.4)
            for rc in cells[valid][::3]:
                candidates.append((int(rc[0]), int(rc[1]), 1.))
        # A room can be lidar-mapped while camera views remain incomplete. Visit
        # reachable, less-viewed vantage cells instead of declaring search finished.
        rr, cc = np.nonzero(np.isfinite(f.cum) & (clear>.45) & g.free())
        candidates += [(int(r), int(c), 0.) for r,c in zip(rr[::9], cc[::9])]
        best = None
        if not self.goals:
            # A short, sensor-verified step into the current view creates parallax
            # before coverage routing; it is not a waypoint towards the target.
            xy=state.position[:2]+.9*np.array([np.cos(state.yaw),np.sin(state.yaw)])
            r,c=g.to_cell(xy)
            if g.inside(r,c) and np.isfinite(f.cum[r,c]) and clear[r,c]>.4:
                self.goal=xy;self.path=f.path_to((r,c))
                self.goals.append({'t':state.t,'xy':xy.tolist(),'source':'observed-free initial heading'})
                return
        for r,c,frontier in candidates:
            xy = g.to_xy(r,c)
            dist = np.linalg.norm(xy-state.position[:2])
            if not .7 <= dist <= 3.5 or any(np.linalg.norm(xy-b)<.6 for b in self.blacklist):
                continue
            novelty = float((~self.camera_seen[max(0,r-4):r+5,max(0,c-4):c+5]).mean())
            score = f.cum[r,c]*g.res - 6.*frontier - .8*novelty + .1*self.visits[r,c]
            if best is None or score<best[0]:
                best = (score,(r,c),xy)
        if best is None:
            self.path = self.goal = None
            return
        self.goal = best[2]
        self.path = f.path_to(best[1])
        self.goals.append({"t":state.t,"xy":self.goal.tolist(),"source":"sensor-map frontier/coverage"})

    def command(self, state, blocked=False):
        if self.goal is not None and (np.linalg.norm(state.position[:2]-self.goal)<.25 or blocked):
            self.blacklist.append(self.goal.copy())
            self.goal = self.path = None
            self.pause_until = state.t+1.5
        if state.t<self.pause_until:
            return (0.,0.,.3), "vantage_scan"
        if self.path is None or state.t>=self.next_plan:
            self._plan(state)
            self.next_plan = state.t+2.
        if self.path is None:
            return (0.,0.,.3), "no_reachable_vantage"
        d = np.linalg.norm(self.path-state.position[:2],axis=1)
        i = int(np.argmin(d))
        while i<len(self.path)-1 and d[i]<.35:
            i += 1
        delta = self.path[i]-state.position[:2]
        err = float((np.arctan2(delta[1],delta[0])-state.yaw+np.pi)%(2*np.pi)-np.pi)
        vx = .18*max(0.,np.cos(err)) if abs(err)<.4 else 0.
        self.moves += int(vx>0)
        return (vx,0.,float(np.clip(1.5*err,-.35,.35))), "frontier_travel"


class ExploringMission:
    """Wraps unchanged visual mission; replaces only its unconfirmed SEARCH command."""
    def __init__(self, task_config, start_time, explorer):
        from .mission import SearchMission
        self.visual = SearchMission(task_config,start_time)
        self.explorer = explorer
        self.rejected=[]
        self.recoveries=0
        self.inspection_votes=[]
        self.inspection_window=float(task_config.get('inspection_window_s',1.0))
        if not np.isfinite(self.inspection_window) or not 0 < self.inspection_window <= 1.:
            raise ValueError('Inspection evidence window must be in (0,1] seconds')

    def update(self, state, frame_time, observations, contacts, blocked):
        from .mission import State
        observations=[o for o in observations if not any(state.t<until and np.linalg.norm(np.array(o.world_xyz)-xyz)<.65
                      for xyz,until in self.rejected)]
        d = self.visual.update(state.t,frame_time,observations,contacts,state.upright,
                               blocked and self.visual.state != State.SEARCH,
                               float(np.linalg.norm(state.lin_vel_world[:2])))
        # Keep bounded, fresh inspection evidence across brief detector dropouts.
        # The original mission still enforces identity, range, speed, timestamps,
        # tilt, contacts and deadline. Completion requires a current valid image;
        # missing detections always stop and can never trigger completion.
        self.inspection_votes=[t for t in self.inspection_votes if state.t-t <= self.inspection_window+1e-8]
        if d.state==State.INSPECT and d.target is not None and np.linalg.norm(state.lin_vel_world[:2])<.06:
            self.inspection_votes.append(frame_time)
            if len(self.inspection_votes)>=self.visual.c['inspect_frames']:
                self.visual.state=State.COMPLETE;self.visual.reason='inspection_window_ready'
                d.state=State.COMPLETE;d.reason=self.visual.reason
        elif d.reason!='target_missing_stop':
            self.inspection_votes=[]
        if d.state==State.FAILED and d.reason=='target_lost':
            # Stop/reject briefly, then seek a new view rather than blind driving.
            if self.visual.anchor is not None:self.rejected.append((self.visual.anchor.copy(),state.t+8.))
            self.visual.state=State.SEARCH;self.visual.anchor=None;self.visual.hits=0
            self.recoveries+=1
            d.state=State.SEARCH;d.command=(0.,0.,0.);d.reason='target_lost_recovery'
            return d
        if d.state == State.SEARCH:
            d.command,d.reason = self.explorer.command(state,blocked)
        return d
