"""Evaluation-only fixed-base panorama. Does not provide waypoints to a controller.

Kinematic camera/base heading sweeps are counterfactual visibility tests, not robot
motion trials. Physics time is frozen; every qpos change is restored afterward.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
from .simulation import RegisteredSimulation
from ..adapters.yolo import ChairDetector
from ..probe import annotated
from ..q2_data_simulation.recording import write_json


def panorama(config,out):
    import mujoco
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    sim=RegisteredSimulation(config['simulation']);detector=ChairDetector(.3);rows=[]
    try:
        s=sim.session;qpos=s.data.qpos.copy()
        chair_geoms=[]
        for g in range(s.model.ngeom):
            b=int(s.model.geom_bodyid[g]);names=[]
            while b>0:
                names.append(mujoco.mj_id2name(s.model,mujoco.mjtObj.mjOBJ_BODY,b) or '');b=int(s.model.body_parentid[b])
            if any('living_room_task_chair_0' in n for n in names):chair_geoms.append(g)
        for k,yaw in enumerate(np.linspace(-np.pi,np.pi,24,endpoint=False)):
            s.data.qpos[:]=qpos;s.data.qpos[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)];mujoco.mj_forward(s.model,s.data)
            f,_,_,_=sim.sample();ds=detector.detect_rgb(f.rgb)
            annotated(f.rgb,ds).save(out/f'heading_{k:02d}.png')
            r=s.sensors.renderers[s.camera_key]
            try:
                r.enable_segmentation_rendering();r.update_scene(s.data,camera=s.sensors.cameras[s.camera_key]['camera'],scene_option=s.sensors.render_option)
                seg=r.render();pixels=int((np.isin(seg[:,:,0],chair_geoms)&(seg[:,:,1]==mujoco.mjtObj.mjOBJ_GEOM)).sum())
            finally:r.disable_segmentation_rendering()
            rows.append({'yaw':float(yaw),'accepted_yolo_chairs':len(ds),'chair_visual_pixels':pixels})
        s.data.qpos[:]=qpos;mujoco.mj_forward(s.model,s.data)
        result={'protocol':'evaluation-only frozen-base sweep, 24 headings; no navigation',
                'settled_base_xy':qpos[:2].tolist(),'max_chair_visual_pixels':max(r['chair_visual_pixels'] for r in rows),
                'total_yolo_detections':sum(r['accepted_yolo_chairs'] for r in rows),'headings':rows,'config':config}
        write_json(out/'panorama.json',result);print(json.dumps(result,indent=2))
    finally:sim.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--out',required=True);a=p.parse_args()
    panorama(yaml.safe_load(a.config.read_text()),a.out)
