"""Phase 2C evidence checks: checksums, actual YOLO, source and inspection guard."""
import argparse,json,csv,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from .detector import SearchDetector
from .perception import CalibratedTarget,inspection_valid
from ..q2_data_simulation.recording import write_json


def verify(path,repeat_yolo=False):
    p=Path(path);hashes=json.loads((p/'sha256.json').read_text())
    for n,h in hashes.items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
    meta=json.loads((p/'metadata.json').read_text());summary=json.loads((p/'summary.json').read_text())
    es=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()]
    with (p/'states.csv').open() as f:states=list(csv.DictReader(f))
    quality=json.loads((p/'depth_quality.json').read_text())
    assert len(es)==len(states)==len(quality)==meta['frames']==summary['frames']
    detector=SearchDetector(meta['config']['detector']['confidence']) if repeat_yolo else None
    if detector:assert detector.sha256==meta['model_sha256']
    full_source_contract=any('depth_source' in f.read_text() for f in (p/'source_snapshot').glob('simulation.py'))
    times=[];xy=[]
    for i,(e,s,q) in enumerate(zip(es,states,quality)):
        t=float(s['t']);times.append(t);xy.append([float(s['x']),float(s['y'])])
        assert np.isfinite([float(s[k]) for k in ('x','y','z','yaw','upright','command_vx','command_vy','command_wz')]).all()
        assert e['frame']==q['frame']==i and abs(e['t']-t)<1e-5
        assert all(abs(float(s[k])-t)<1e-5 for k in ('camera_t','lidar_t','imu_t'))
        for folder in ('depth','legacy_depth','lidar'):
            with np.load(p/folder/f'{i:05d}.npz') as d:assert abs(float(d['timestamp'])-t)<1e-5
        if detector:
            ds=detector.detect_rgb(np.asarray(Image.open(p/'rgb'/f'{i:05d}.png').convert('RGB')))
            # The first calibration runner used the preserved full-box decoder.
            if not full_source_contract:ds=[d for d in ds if not d.edge_clipped]
            assert len(ds)==len(e['detections']),i
            for a,b in zip(ds,e['detections']):
                assert abs(a.conf-b['conf'])<1e-5 and np.allclose(a.bbox,b['bbox'],atol=1e-3),i
        if e['decision']['state']=='complete':
            target=CalibratedTarget(**e['decision']['target'])
            assert inspection_valid(target,meta['config']['task'])
            assert abs(target.timestamp-t)<1e-5
            assert float(s['upright'])>=.92 and int(s['contacts'])==0
            speed=np.hypot(float(s['velocity_world_x']),float(s['velocity_world_y']))
            assert speed<.06
            source=(p/'source_snapshot/control.py').read_text()
            strict='        self.inspection_votes=[]' in source.splitlines()
            previous=range(max(0,i-2),i+1) if strict else [j for j in range(max(0,i-5),i+1) if t-float(states[j]['t'])<=1.+1e-8]
            good=0
            for j in previous:
                obj=es[j]['decision']['target']
                if obj is not None and inspection_valid(CalibratedTarget(**obj),meta['config']['task']):
                    vel=np.hypot(float(states[j]['velocity_world_x']),float(states[j]['velocity_world_y']))
                    good+=int(vel<.06 and int(states[j]['contacts'])==0)
            assert good>=meta['config']['task']['inspect_frames']
    assert np.allclose(np.diff(times),meta['config']['simulation']['dt'])
    assert np.allclose(es[-1]['command_applied'],0.)
    assert np.isclose(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum(),summary['path_m'])
    if summary['success']:
        ref=summary['scoring_only_reference']
        assert ref['ray_hit_chair'] and abs(ref['reference_horizontal_range_m']-.95)<=.10
    return {'trial':str(p),'frames':len(es),'files_verified':len(hashes),'sensor_timestamps_valid':True,
            'path_reaggregated':True,'inspection_conditions_checked':True,'success':summary['success'],
            'yolo_reinference_matches':True if repeat_yolo else None,'final_command_zero':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trial',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--repeat-yolo',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    r=verify(a.trial,a.repeat_yolo);write_json(a.out,r);print(json.dumps(r,indent=2))
