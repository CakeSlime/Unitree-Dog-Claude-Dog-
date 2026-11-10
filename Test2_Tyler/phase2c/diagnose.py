"""Read-only historical RGB/depth/LiDAR comparison and evaluation-only mesh rays."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from ..q2_data_simulation.recording import write_json
from ..q1_vision.search_observations import search_targets
from ..q1_vision.observations import targets_from_depth
from .simulation import RegisteredSimulation


def diagnose(out):
    import yaml,mujoco,xml.etree.ElementTree as ET
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    config=yaml.safe_load(Path('Test2_Tyler/config/phase2c/calibration_visible.yaml').read_text())
    sim=RegisteredSimulation(config['simulation']);rows=[]
    try:
        from sim_env.interfaces import CameraFrame
        from go2_stream.perception.detector import Detection
        world=ET.parse(sim.session.scene_path).getroot().find('worldbody')
        write_json(out/'scene_children.json',[(c.tag,c.get('name')) for c in world])
        for name,index in [('occluded_001',3),('occluded_20261110',155),('visible_20261110',49),('near_20261110',68)]:
            p=Path('Test2_Tyler/data/phase2b')/name
            es=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()];e=es[index]
            with (p/'states.csv').open() as f:state=list(csv.DictReader(f))[index]
            s=sim.session
            # Historical camera transform is exact; joint positions were not saved.
            # Reconstruct base pose for diagnostic self-hit identification only.
            s.data.qpos[:3]=[float(state[x]) for x in ('x','y','z')]
            s.data.qpos[3:7]=[float(state['quat_'+x]) for x in ('w','x','y','z')]
            mujoco.mj_forward(s.model,s.data)
            rgb=np.asarray(Image.open(p/'rgb'/f'{index:05d}.png').convert('RGB'))
            with np.load(p/'depth'/f'{index:05d}.npz') as d:
                f=CameraFrame(e['t'],rgb,d['depth_m'],d['K'],d['T_world_cam'])
            ds=[Detection(**d) for d in e['detections']];ts=search_targets(f,ds)
            annotation=Image.fromarray(rgb);draw=ImageDraw.Draw(annotation)
            for target in ts:
                u,v=target.sample_pixel;draw.rectangle(target.bbox,outline='red',width=2);draw.ellipse((u-4,v-4,u+4,v+4),fill='yellow')
                ray=f.T_world_cam[:3,:3]@np.array([(u-f.K[0,2])/f.K[0,0],(v-f.K[1,2])/f.K[1,1],1.]);ray/=np.linalg.norm(ray)
                refs={}
                for label,mask in [('visual',[0,1,0,0,0,0]),('collision',[1,0,0,1,0,1]),
                                   ('centre_visual',[0,1,0,0,0,0]),('centre_collision',[1,0,0,1,0,1])]:
                    if label.startswith('centre'):
                        cu=(target.bbox[0]+target.bbox[2])/2;cv=(target.bbox[1]+target.bbox[3])/2
                        ray=f.T_world_cam[:3,:3]@np.array([(cu-f.K[0,2])/f.K[0,0],(cv-f.K[1,2])/f.K[1,1],1.]);ray/=np.linalg.norm(ray)
                    gid=np.zeros(1,np.int32)
                    distance=mujoco.mj_ray(s.model,s.data,f.T_world_cam[:3,3],ray,np.array(mask,np.uint8),1,s.base_body,gid)
                    refs[label]={'ray_m':float(distance),'horizontal_m':float(distance*np.linalg.norm(ray[:2])),
                      'geom':mujoco.mj_id2name(s.model,mujoco.mjtObj.mjOBJ_GEOM,int(gid[0])) if distance>0 else None,
                      'body':mujoco.mj_id2name(s.model,mujoco.mjtObj.mjOBJ_BODY,int(s.model.geom_bodyid[gid[0]])) if distance>0 else None,
                      'robot_self_hit':bool(int(gid[0]) in s.robot_geoms) if distance>0 else False}
                with np.load(p/'lidar'/f'{index:05d}.npz') as l:
                    angles=float(l['angle_min'])+np.arange(len(l['ranges']))*float(l['angle_increment'])
                    # Bearing in the measured LiDAR sensor frame, using observed-depth point.
                    point=np.linalg.inv(l['T_world_lidar'])@np.r_[target.world_xyz,1.]
                    bearing=np.arctan2(point[1],point[0]);k=np.argmin(np.abs((angles-bearing+np.pi)%(2*np.pi)-np.pi))
                    lidar=float(l['ranges'][k])
                rows.append({'trial':name,'frame':index,'legacy_upper_range_m':target.range_m,'sample_pixel':[u,v],
                             'legacy_central_ranges_m':[t.range_m for t in targets_from_depth(f,ds)],
                             'lidar_planar_range_near_observed_bearing_m':lidar,'references':refs})
            annotation.save(out/f'{name}_annotated.png')
            # Save measured legacy depth for a comparison figure, not replacing old data.
            np.savez_compressed(out/f'{name}_depth.npz',depth_m=f.depth_m)
        write_json(out/'depth_comparisons.json',rows)
        print(json.dumps(rows,indent=2))
    finally:sim.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();diagnose(a.out)
