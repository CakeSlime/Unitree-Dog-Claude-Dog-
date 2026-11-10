"""Phase 2C runner adapted from Tyler Phase 2B search.py; old evidence unchanged."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageDraw
import yaml

from Test2_Tyler.phase2c.simulation import RegisteredSimulation as SearchSimulation
from Test2_Tyler.phase2c.detector import SearchDetector as ChairDetector
from Test2_Tyler.phase2c.control import ProgressExplorer as SensorExplorer,CalibratedMission as ExploringMission
from Test2_Tyler.q1_vision.mission import State
from Test2_Tyler.q1_vision.observations import depth_obstacles
from Test2_Tyler.phase2c.perception import DepthAssociation
from Test2_Tyler.q2_data_simulation.recording import TrialRecorder,write_json
from Test2_Tyler.probe import annotated


def trial(config,output,wall_limit=110.):
    if Path(output).exists():
        raise FileExistsError(output)
    detector=ChairDetector(config['detector']['confidence'])
    sim=SearchSimulation(config['simulation'])
    from navigation.local import SafetyLayer,SafetyConfig
    safety=SafetyLayer(SafetyConfig(corridor_margin=.125,stop_dist=.22))
    rec=None
    try:
        rec=TrialRecorder(output,config,detector.sha256)
        import hashlib,shutil
        source_files=[*Path(__file__).parent.glob('*.py'),
                      Path('Test2_Tyler/q1_vision/exploration.py'),Path('Test2_Tyler/q1_vision/mission.py'),
                      Path('Test2_Tyler/q1_vision/sensor_map.py'),Path('Test2_Tyler/adapters/search_sim.py')]
        (rec.path/'source_snapshot').mkdir()
        for source in source_files:shutil.copy2(source,rec.path/'source_snapshot'/source.name)
        write_json(rec.path/'source_snapshot/sha256.json',{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in source_files})
        rec.meta['phase']='2C actual YOLO calibrated RGBD exploration'
        rec.meta['depth']='640x480 rendered optical-z, RGB camera/mask; legacy collision depth separately retained'
        (rec.path/'legacy_depth').mkdir()
        association=DepthAssociation();diagnostics=[];references=[]
        rec.meta['controller']='scripted physical trot + Tyler sensor-map search; no PPO'
        st=sim.session.state()
        explorer=SensorExplorer(st.position,config.get('exploration',{}))
        mission=ExploringMission(config['task'],st.t,explorer)
        t0=st.t
        wall=time.monotonic()
        previous=None
        path=[];phase_counts=Counter();confidences=[];images=[]
        first_detection=None;first_confirm=None;initial_detected=None
        search_forward=contact_steps=contact_events=limited=0
        previous_contacts=set();was_blocked=False
        reference=None;min_upright=1.
        while True:
            f,scan,st,contacts=sim.sample()
            det=detector.detect_rgb(f.rgb)
            targets,quality=association.measure(f,det)
            diagnostics.append({"frame":rec.count,"t":st.t,"checks":quality})
            np.savez_compressed(rec.path/"legacy_depth"/f"{rec.count:05d}.npz",depth_m=sim.legacy_frame.depth_m,timestamp=f.t)
            if initial_detected is None:initial_detected=bool(targets)
            if targets and first_detection is None:first_detection=st.t-t0
            pts=depth_obstacles(f)
            explorer.observe(scan,st,pts)
            mission.camera_intrinsics=(f.K[0,2],f.K[0,0])
            decision=mission.update(st,f.t,targets,len(contacts),was_blocked,discoveries=det)
            if first_confirm is None and decision.state==State.APPROACH:first_confirm=st.t-t0
            if time.monotonic()-wall>wall_limit and not mission.visual.terminal:
                decision=mission.visual.fail('wall_clock_budget')
            # Semantic mesh queries occur AFTER the decision, for scoring only.
            refs=[sim.score_surface(f,t) for t in targets]
            references.append({'frame':rec.count,'t':st.t,'references':refs})
            safety.observe(scan)
            requested=decision.command
            near=explorer.grid.occupied_points_near(st.position[:2])
            extra=np.vstack([pts[:,:2],near])
            applied=safety.filter(st,*requested,map_points=extra)
            was_blocked=bool(np.linalg.norm(requested)>.02 and np.linalg.norm(applied)<.005)
            limited+=int(safety.last['limited'])
            now_contacts={(c.robot_geom,c.other_body) for c in contacts}
            contact_events+=len(now_contacts-previous_contacts);previous_contacts=now_contacts
            contact_steps+=int(bool(contacts))
            min_upright=min(min_upright,st.upright)
            search_forward+=int(decision.state==State.SEARCH and applied[0]>.01)
            path.append([st.t,*st.position[:2],st.yaw,decision.state.value])
            phase_counts[decision.state.value]+=1
            confidences.extend(d.conf for d in det)
            rec.record(f,scan,st,det,decision,requested,applied,safety.last,contacts,detector.last_latency_s)
            if rec.count%5==1 or mission.visual.terminal:
                im=annotated(f.rgb,det);im.thumbnail((400,300))
                draw=ImageDraw.Draw(im);draw.rectangle((0,0,400,22),fill='black')
                draw.text((4,5),f"{st.t-t0:.1f}s {decision.state.value}: {decision.reason}",fill='white')
                images.append(im)
            if previous!=(decision.state,decision.reason):
                print(f't={st.t-t0:.1f} {decision.state.value}: {decision.reason}',flush=True)
                previous=(decision.state,decision.reason)
            if decision.state==State.COMPLETE:
                Image.fromarray(f.rgb).save(rec.path/'inspection/rgb.png')
                annotated(f.rgb,det).save(rec.path/'inspection/annotated.png')
                reference=sim.score_surface(f,decision.target)
                write_json(rec.path/'inspection/observation.json',{'target':decision.target,'pose':st.pose2d,
                            'timestamp':st.t,'scoring_only_reference':reference})
            if mission.visual.terminal:break
            sim.step(applied,config['simulation']['dt'])
        sim.session.stop()
        xy=np.array([r[1:3] for r in path])
        result={'controller_complete':decision.state==State.COMPLETE,
                'success':bool(decision.state==State.COMPLETE and reference and reference['ray_hit_chair']
                               and abs(reference['reference_horizontal_range_m']-config['task']['standoff_m'])<=config['task']['distance_tolerance_m'] and contact_steps==0),'reason':decision.reason,'final_state':decision.state.value,
                'initially_detected':initial_detected,'first_detection_s':first_detection,'confirmed_s':first_confirm,
                'duration_sim_s':st.t-t0,'wall_time_s':time.monotonic()-wall,'frames':rec.count,
                'path_m':float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
                'net_displacement_m':float(np.linalg.norm(xy[-1]-xy[0])),
                'search_forward_steps':search_forward,'planned_vantages':explorer.goals,
                'target_loss_recoveries':mission.recoveries,
                'mapping_backend':'NumPy/OpenCV + heapq adaptations of shared mapping/planning',
                'contact_steps':contact_steps,'collision_events':contact_events,'min_upright':min_upright,
                'safety_intervention_steps':limited,'phase_counts':dict(phase_counts),
                'detection_confidence_mean':float(np.mean(confidences)) if confidences else None,
                'detection_confidence_max':max(confidences) if confidences else None,
                'inspection_range_m':decision.target.range_m if decision.target else None,
                'standoff_estimate_error_m':decision.target.range_m-config['task']['standoff_m'] if decision.target else None,
                'scoring_only_reference':reference,'final_applied_command':applied,
                'protocol':'actual YOLO + rendered registered RGBD + sensor-map exploration; not PPO/proxy',
                'range_rejections':sum(not q['accepted'] for d in diagnostics for q in d['checks']),
                'observed_surface_approach_goals':mission.approach_route.goals,
                'standoff_definition':'horizontal camera-optical-centre to measured visible chair surface at recorded sample_pixel; 0.95 +/- 0.10 m, uncertainty interval inside band'}
        np.savez_compressed(rec.path/'observed_map.npz',seen=explorer.grid.seen,occupied=explorer.grid.occupied(),
                            camera_seen=explorer.camera_seen,origin=explorer.grid.origin,res=explorer.grid.res)
        write_json(rec.path/'path.json',path)
        write_json(rec.path/'depth_quality.json',diagnostics)
        write_json(rec.path/'evaluation_references.json',references)
        if images:images[0].save(rec.path/'demo.gif',save_all=True,append_images=images[1:],duration=1000,loop=0)
        rec.finish(result)
        print(json.dumps(result,default=lambda x:x.tolist(),indent=2),flush=True)
        return result
    finally:
        sim.close()
        if rec:
            rec.states.close();rec.events.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--wall-limit',type=float,default=110.)
    a=p.parse_args()
    if not 0<a.wall_limit<=150:p.error('wall limit must be (0,150]')
    r=trial(yaml.safe_load(a.config.read_text()),a.out,a.wall_limit)
    raise SystemExit(0 if r['success'] else 2)


if __name__=='__main__':main()
