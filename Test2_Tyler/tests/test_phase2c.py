"""Calibration fail-stop, uncertainty-band and versioned proxy contracts."""
from types import SimpleNamespace
import numpy as np
import yaml
from Test2_Tyler.adapters.shared import WORKSPACE
from Test2_Tyler.phase2c.perception import DepthAssociation,CalibratedTarget,inspection_valid
from Test2_Tyler.phase2c.control import CalibratedMission
from Test2_Tyler.q1_vision.mission import State
from Test2_Tyler.q3_rl.feasible_env import FeasibleSearchEnv


def frame(depth=1.,t=0.):
    z=np.full((100,100),depth,dtype=np.float32)
    if .3<=depth<=6.:
        z[:]=depth+1.;z[16:50,35:65]=depth
    K=np.array([[100,0,50],[0,100,50],[0,0,1.]])
    T=np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,.4],[0,0,0,1.]])
    def backproject(u,v,d):return (T@np.array([(u-50)/100*d,(v-50)/100*d,d,1.]))[:3]
    return SimpleNamespace(t=t,depth_m=z,K=K,T_world_cam=T,backproject=backproject,depth_source='registered_RGBD')


def detection():return SimpleNamespace(name='chair',source='yolo',conf=.8,bbox=(30,10,70,90))


def test_registered_range_and_inspection_interval():
    ts,log=DepthAssociation().measure(frame(),[detection()])
    assert len(ts)==1 and np.isclose(ts[0].range_m,1.,atol=.001) and log[0]['accepted']
    c={'standoff_m':.95,'distance_tolerance_m':.10}
    assert inspection_valid(ts[0],c)
    assert not inspection_valid(CalibratedTarget(0,.8,(0,0,1,1),0,1.04,(1,0,.5),20,(0,0),.025),c)


def test_invalid_near_nonfinite_and_missing_depth_rejected():
    for z in (.17,np.nan,np.inf,0.,-1.,7.):
        ts,log=DepthAssociation().measure(frame(z),[detection()])
        assert not ts and not log[0]['accepted']


def test_foreground_background_mix_and_upper_support_disagreement():
    f=frame(2.)
    f.depth_m[:31]=.7
    ts,log=DepthAssociation().measure(f,[detection()])
    assert not ts and log[0]['reason']=='foreground_occlusion'


def test_observed_surface_jump_stops_association():
    a=DepthAssociation();assert a.measure(frame(2.),[detection()])[0]
    ts,log=a.measure(frame(.8,.2),[detection()])
    assert not ts and log[0]['reason']=='surface_jump'


def mission():
    class Explorer:
        def command(self,*args):return (0.,0.,.3),'scan'
    c=yaml.safe_load((WORKSPACE/'config/chair.yaml').read_text())['task']
    return CalibratedMission(c,0.,Explorer())


def target(t,r=1.):
    return CalibratedTarget(t,.8,(30,10,70,90),0.,r,(1,0,.5),100,(50,30),.025)


def test_false_inspection_on_uncertainty_boundary_is_prevented():
    m=mission();s=SimpleNamespace(t=0.,upright=1.,lin_vel_world=np.zeros(3))
    for i in range(8):
        s.t=i*.2;d=m.update(s,s.t,[target(s.t,1.04)],0,False)
        assert d.state!=State.COMPLETE
    assert d.reason=='range_uncertainty_guard' and d.command[0]>0


def test_current_consecutive_stable_evidence_required():
    m=mission();s=SimpleNamespace(t=0.,upright=1.,lin_vel_world=np.zeros(3))
    for i in range(3):
        s.t=i*.2;m.update(s,s.t,[target(s.t,1.5)],0,False)
    for t,seen in ((.6,True),(.8,False),(1.,True),(1.2,True)):
        s.t=t;d=m.update(s,t,[target(t)] if seen else [],0,False)
        assert d.state!=State.COMPLETE
        if not seen:assert d.command==(0.,0.,0.)
    s.t=1.4;assert m.update(s,1.4,[target(1.4)],0,False).state==State.COMPLETE


def test_noncalibrated_or_nonfinite_target_cannot_complete():
    from Test2_Tyler.q1_vision.observations import TargetObservation
    m=mission();s=SimpleNamespace(t=0.,upright=1.,lin_vel_world=np.zeros(3))
    for i in range(8):
        s.t=i*.2
        old=TargetObservation(s.t,.9,(0,0,2,2),0,.95,(1,0,.5),10)
        d=m.update(s,s.t,[old],0,False)
        assert d.state!=State.COMPLETE


def test_phase2c_proxy_checker_and_unknown_target_masking():
    from stable_baselines3.common.env_checker import check_env
    check_env(FeasibleSearchEnv())
    a,b=FeasibleSearchEnv({'dropout':0}),FeasibleSearchEnv({'dropout':0})
    x,_=a.reset(seed=1,options={'layout':'open','spawn':[1,1,np.pi],'target':[6,4]})
    y,info=b.reset(seed=1,options={'layout':'open','spawn':[1,1,np.pi],'target':[6,5]})
    assert np.array_equal(x,y) and np.all(x[24:29]==0) and 'target_xy' not in info


def test_proxy_uses_camera_surface_band_and_time_budget():
    e=FeasibleSearchEnv({'dropout':0})
    e.reset(seed=2,options={'layout':'open','spawn':[1,1,0],'target':[2.27,1]})
    assert np.isclose(e.target.range_m,.95) and e.cfg['max_steps']==1200
    for i in range(5):o,r,term,trunc,info=e.step([0,0])
    assert term and not trunc and info['success']


def test_legacy_depth_source_and_partial_boxes_are_not_inspection_evidence():
    f=frame();f.depth_source='collision_proxy'
    ts,log=DepthAssociation().measure(f,[detection()])
    assert not ts and log[0]['reason']=='unregistered_depth_source'
    d=detection();d.edge_clipped=True
    ts,log=DepthAssociation().measure(frame(),[d])
    assert not ts and log[0]['reason']=='edge_clipped_discovery_only'


def test_partial_actual_yolo_can_center_but_never_drive_or_complete():
    from Test2_Tyler.phase2c.detector import decode_discovery
    ds=decode_discovery([[0,50,100,300,.8,56]],640,480)
    assert len(ds)==1 and ds[0].edge_clipped and ds[0].source=='yolo'
    m=mission();m.camera_intrinsics=(320.,500.)
    s=SimpleNamespace(t=0.,upright=1.,lin_vel_world=np.zeros(3))
    for i in range(8):
        s.t=i*.2;d=m.update(s,s.t,[],0,False,discoveries=ds)
        assert d.state==State.SEARCH and d.command[0]==0 and d.command[2]>0


def test_historical_floor_lamp_depth_cannot_reproduce_false_range():
    from Test2_Tyler.adapters.shared import configure
    configure()
    from sim_env.interfaces import CameraFrame
    from go2_stream.perception.detector import Detection
    import json
    p=WORKSPACE/'data/phase2b/occluded_20261110'
    e=json.loads((p/'events.jsonl').read_text().splitlines()[155])
    with np.load(p/'depth/00155.npz') as d:
        f=CameraFrame(e['t'],np.zeros((480,640,3),np.uint8),d['depth_m'],d['K'],d['T_world_cam'])
    # Even if mislabeled as registered, the observed near-range values are invalid.
    f.depth_source='registered_RGBD'
    ts,log=DepthAssociation().measure(f,[Detection(**x) for x in e['detections']])
    assert not ts and log[0]['reason']=='invalid_or_near_depth'


def test_classical_controller_can_complete_unknown_target_proxy_episode():
    from Test2_Tyler.q3_rl.search_baseline import ScanVantageController
    env=FeasibleSearchEnv();obs,_=env.reset(seed=902031);c=ScanVantageController()
    for _ in range(env.cfg['max_steps']):
        a=c.predict(obs);assert env.action_space.contains(a)
        obs,r,term,trunc,info=env.step(a)
        if term or trunc:break
    assert info['success'] and not info['collision']
    env.close()


def test_candidate_dropout_holds_camera_without_unconfirmed_forward_motion():
    m=mission();s=SimpleNamespace(t=0.,upright=1.,lin_vel_world=np.zeros(3))
    assert m.update(s,0.,[target(0.,1.5)],0,False).state==State.CONFIRM
    s.t=.2;d=m.update(s,.2,[],0,False)
    assert d.reason=='candidate_reacquisition_hold' and d.command==(0.,0.,0.)
    s.t=1.4;assert m.update(s,1.4,[],0,False).command[2]!=0


def test_point_changes_within_apparent_object_extent_but_large_jump_rejected():
    a=DepthAssociation();assert a.measure(frame(3.),[detection()])[0]
    f=frame(3.,.2);f.T_world_cam[0,3]=.4
    assert a.measure(f,[detection()])[0]
    f=frame(3.,.4);f.T_world_cam[0,3]=1.4
    ts,log=a.measure(f,[detection()])
    assert not ts and log[0]['reason']=='surface_jump'


def test_mapped_approach_routes_around_obstacles_using_only_observed_surface():
    from Test2_Tyler.phase2c.approach import MappedApproach
    from Test2_Tyler.q1_vision.sensor_map import OccupancyGrid
    g=OccupancyGrid((-2,-2),(8,6),.1);g.logodds[:]=-1.2;g.seen[:]=True
    # Obstacle on the direct line, with room to route around its ends.
    g.logodds[17:24,30:33]=3.
    s=SimpleNamespace(t=0.,position=np.array([0.,0.,.3]),yaw=0.)
    t=CalibratedTarget(0,.8,(0,0,20,30),0.,3.,(3.,0.,.7),100,(10,10),.025)
    a=MappedApproach();command=a.command(s,t,g)
    assert command is not None and a.path is not None and a.goals
    rr,cc=g.to_cell(a.path)
    assert np.isfinite(g.costmap(robot_radius=.42)[rr[1:],cc[1:]]).all()
    assert a.command(s,None,g) is None


def test_calibrated_reacquisition_is_not_stalled_by_old_depth_blacklist():
    m=mission();s=SimpleNamespace(t=0.,upright=1.,lin_vel_world=np.zeros(3))
    for t in (0.,.2,.4):s.t=t;m.update(s,t,[target(t,2.)],0,False)
    s.t=2.6;d=m.update(s,2.6,[],0,False)
    assert d.reason=='target_lost_recovery' and d.command==(0.,0.,0.) and not m.rejected
    for t in (2.8,3.,3.2):s.t=t;d=m.update(s,t,[target(t,2.)],0,False)
    assert d.state==State.APPROACH
