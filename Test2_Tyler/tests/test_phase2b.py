"""Phase 2B environment, observability, reward and exploration contracts."""
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
from Test2_Tyler.q3_rl.object_search_env import ObjectSearchEnv,reward_terms,DEFAULTS
from Test2_Tyler.q1_vision.exploration import ExploringMission
from Test2_Tyler.q1_vision.observations import TargetObservation
from Test2_Tyler.q1_vision.mission import State
from Test2_Tyler.adapters.shared import WORKSPACE


def test_gymnasium_contract():
    from gymnasium.utils.env_checker import check_env
    env=ObjectSearchEnv()
    check_env(env,skip_render_check=True)
    obs,_=env.reset(seed=12)
    assert obs.shape==(38,) and env.observation_space.contains(obs)


def test_seeded_episodes_and_transitions():
    a,b=ObjectSearchEnv(),ObjectSearchEnv()
    x,_=a.reset(seed=36);y,_=b.reset(seed=36)
    assert np.array_equal(x,y)
    rng=np.random.default_rng(21)
    for _ in range(20):
        act=rng.uniform(-1,1,2)
        ra,rb=a.step(act),b.step(act)
        assert np.array_equal(ra[0],rb[0]) and ra[1:4]==rb[1:4]
        if ra[2] or ra[3]:break


def test_randomized_resets_and_finite_complete_rollouts():
    env=ObjectSearchEnv();rng=np.random.default_rng(23);starts=[];targets=[]
    for seed in range(4):
        obs,_=env.reset(seed=seed)
        starts.append(tuple(env.xy));targets.append(tuple(env._target_xy))
        assert env.observation_space.contains(obs)
        for _ in range(env.cfg['max_steps']):
            action=rng.uniform(-1,1,2).astype(np.float32)
            assert env.action_space.contains(action)
            obs,reward,terminated,truncated,info=env.step(action)
            assert np.isfinite(obs).all() and env.observation_space.contains(obs)
            assert np.isfinite(reward) and np.isclose(reward,sum(info['reward_terms'].values()))
            if terminated or truncated:break
        assert terminated or truncated
    assert len(set(starts))==4 and len(set(targets))==4
    env.close()


def test_unseen_ground_truth_cannot_change_policy_observation():
    a,b=ObjectSearchEnv({'dropout':0}),ObjectSearchEnv({'dropout':0})
    x,ia=a.reset(seed=4,options={'layout':'open','spawn':[1,1,np.pi],'target':[6,4]})
    y,ib=b.reset(seed=4,options={'layout':'open','spawn':[1,1,np.pi],'target':[6,5]})
    assert a.target is None and b.target is None and np.array_equal(x,y)
    assert np.all(x[24:29]==0)
    assert 'target_xy' not in ia and 'target_bearing' not in ib


def test_occluded_target_is_not_a_detection():
    env=ObjectSearchEnv({'dropout':0})
    o,_=env.reset(seed=5,options={'spawn':[3,2,0],'target':[4.5,2]})
    assert env.target is None and o[24]==0


def test_inspection_confirmation_reward_and_termination():
    env=ObjectSearchEnv({'dropout':0})
    env.reset(seed=5,options={'layout':'open','spawn':[1,1,0],'target':[1.95,1]})
    confirmations=0
    for i in range(5):
        _,reward,term,trunc,info=env.step(np.zeros(2))
        confirmations+=info['reward_terms']['confirm']>0
        if i<4:assert not term
    assert term and not trunc and info['success'] and confirmations==1
    assert info['reward_terms']['inspect']==DEFAULTS['reward']['inspect']
    with pytest.raises(RuntimeError,match='Reset required'):env.step(np.zeros(2))


def test_collision_terminates_as_failure():
    env=ObjectSearchEnv({'dropout':0})
    env.reset(seed=5,options={'layout':'open','spawn':[.251,1,np.pi],'target':[6,4]})
    _,r,term,trunc,info=env.step([1,0])
    assert term and not trunc and info['collision'] and not info['success']
    assert info['reward_terms']['collision']==-DEFAULTS['reward']['collision']


def test_time_limit_truncates_without_false_collision_or_success():
    env=ObjectSearchEnv({'max_steps':2,'dropout':0})
    env.reset(seed=5,options={'layout':'open','spawn':[1,1,np.pi],'target':[6,4]})
    env.step([0,0]);_,_,term,trunc,info=env.step([0,0])
    assert trunc and not term and not info['success'] and not info['collision']


def test_exploration_reward_without_target_detection():
    env=ObjectSearchEnv({'dropout':0})
    env.reset(seed=5,options={'layout':'open','spawn':[1.249,1,0],'target':[6,4]})
    _,_,_,_,info=env.step([1,0])
    assert info['reward_terms']['explore']>0 and not info['target_detected']


def test_reward_terms_and_invalid_actions():
    t=reward_terms(DEFAULTS['reward'],False,False,False,0.,False,False,np.array([1.,1.]))
    assert t['time']<0 and t['smooth']<0 and t['inspect']==0 and t['progress']==0
    env=ObjectSearchEnv();env.reset(seed=1)
    for action in ([np.nan,0],[0,0,0]):
        with pytest.raises(ValueError):env.step(action)


def test_search_wrapper_moves_without_hidden_target_but_stops_for_confirmation():
    class Explorer:
        def command(self,state,blocked):return (.15,0.,0.),'frontier_travel'
    c=yaml.safe_load((WORKSPACE/'config/chair.yaml').read_text())['task']
    m=ExploringMission(c,0.,Explorer())
    state=SimpleNamespace(t=0.,upright=1.,lin_vel_world=np.zeros(3))
    d=m.update(state,0.,[],0,False)
    assert d.state==State.SEARCH and d.command[0]>.1
    state.t=.2
    target=TargetObservation(.2,.8,(0,0,20,30),0.,2.,(1,2,.5),10)
    d=m.update(state,.2,[target],0,False)
    assert d.state==State.CONFIRM and d.command==(0.,0.,0.)


def test_sensor_planner_avoids_obstacles_and_unknown_space():
    from Test2_Tyler.q1_vision.sensor_map import OccupancyGrid,CostField
    g=OccupancyGrid((0,0),(3,3),.1)
    g.logodds[:]=-1.2;g.seen[:]=True
    g.logodds[5:20,14:16]=3.
    cost=g.costmap(robot_radius=.2)
    f=CostField(g,cost,(.5,1.))
    path=f.path_to(g.to_cell((2.5,1.)))
    assert path is not None and (path[:,1].min()<.5 or path[:,1].max()>2.)
    rr,cc=g.to_cell(path)
    assert not g.occupied()[rr,cc].any()
    g.seen[:]=False;g.clear_disk((.5,1.),.3)
    f=CostField(g,g.costmap(robot_radius=.2),(.5,1.))
    assert f.path_to(g.to_cell((2.5,1.))) is None


def test_target_loss_recovery_never_drives_blind():
    class Explorer:
        def command(self,state,blocked):return (.15,0.,0.),'frontier_travel'
    c=yaml.safe_load((WORKSPACE/'config/chair.yaml').read_text())['task']
    m=ExploringMission(c,0.,Explorer())
    s=SimpleNamespace(t=0.,upright=1.,lin_vel_world=np.zeros(3))
    for t in (0.,.2,.4):
        s.t=t
        m.update(s,t,[TargetObservation(t,.8,(0,0,20,30),0.,2.,(1,2,.5),10)],0,False)
    s.t=2.6;d=m.update(s,2.6,[],0,False)
    assert d.reason=='target_lost_recovery' and d.command==(0.,0.,0.) and m.recoveries==1


def test_sb3_contract_when_dependency_available():
    pytest.importorskip('stable_baselines3',reason='PyPI wheel download blocker recorded in Phase2B report')
    from stable_baselines3.common.env_checker import check_env
    check_env(ObjectSearchEnv(),warn=True)


def test_explorer_keeps_reachable_goal_between_route_refreshes():
    from Test2_Tyler.q1_vision.exploration import SensorExplorer
    e=SensorExplorer(np.array([0.,0.]),{})
    e.grid.logodds[:]=-1.2;e.grid.seen[:]=True
    s=SimpleNamespace(t=0.,position=np.array([0.,0.,.3]),yaw=0.)
    e.command(s)
    goal=e.goal.copy();count=len(e.goals)
    # Changed visits should not replace a reachable committed vantage.
    e.visits[:]=100;s.t=2.2
    e.command(s)
    assert np.array_equal(goal,e.goal) and len(e.goals)==count


def test_inspection_window_requires_current_fresh_stable_identity_evidence():
    class Explorer:
        def command(self,state,blocked):return (0.,0.,.3),'scan'
    c=yaml.safe_load((WORKSPACE/'config/chair.yaml').read_text())['task']
    m=ExploringMission(c,0.,Explorer())
    s=SimpleNamespace(t=0.,upright=1.,lin_vel_world=np.zeros(3))
    def tick(t,seen=True,range_m=.99):
        s.t=t
        obs=[TargetObservation(t,.8,(0,0,20,30),0.,range_m,(1,2,.5),10)] if seen else []
        return m.update(s,t,obs,0,False)
    for t in (0.,.2,.4):d=tick(t,range_m=1.5)
    assert d.state==State.APPROACH
    assert tick(.6).state==State.INSPECT
    assert tick(.8,False).command==(0.,0.,0.)
    assert tick(1.).state==State.INSPECT
    assert tick(1.2,False).state!=State.COMPLETE
    assert tick(1.4).state==State.COMPLETE


def test_inspection_window_expires_and_rejects_out_of_band_frame():
    class Explorer:
        def command(self,state,blocked):return (0.,0.,.3),'scan'
    c=yaml.safe_load((WORKSPACE/'config/chair.yaml').read_text())['task']
    m=ExploringMission(c,0.,Explorer())
    s=SimpleNamespace(t=0.,upright=1.,lin_vel_world=np.zeros(3))
    for t,r in ((0.,1.5),(.2,1.5),(.4,1.5),(.6,.99),(.8,1.5),(1.8,.99),(2.,.99)):
        s.t=t
        d=m.update(s,t,[TargetObservation(t,.8,(0,0,20,30),0.,r,(1,2,.5),10)],0,False)
        assert d.state!=State.COMPLETE
    assert len(m.inspection_votes)==2
