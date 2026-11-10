"""Read-only Phase 2B recording integrity and independent numeric aggregation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .recording import write_json


def verify(path,repeat_yolo=False):
    p=Path(path)
    hashes=json.loads((p/'sha256.json').read_text())
    for name,digest in hashes.items():
        if hashlib.sha256((p/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Checksum: {name}')
    with (p/'states.csv').open() as f:rows=list(csv.DictReader(f))
    events=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()]
    summary=json.loads((p/'summary.json').read_text());meta=json.loads((p/'metadata.json').read_text())
    detector=None
    if repeat_yolo:
        from ..adapters.yolo import ChairDetector
        detector=ChairDetector(meta['config']['detector']['confidence'])
        assert detector.sha256==meta['model_sha256']
    assert len(rows)==len(events)==meta['frames']==summary['frames']
    times=[];xy=[];confidence=[]
    for i,(r,e) in enumerate(zip(rows,events)):
        t=float(r['t']);times.append(t);xy.append([float(r['x']),float(r['y'])])
        assert int(r['frame'])==e['frame']==i and abs(e['t']-t)<1e-5
        assert all(abs(float(r[k])-t)<1e-5 for k in ('camera_t','lidar_t','imu_t'))
        with np.load(p/'depth'/f'{i:05d}.npz',allow_pickle=False) as d:
            assert abs(float(d['timestamp'])-t)<1e-5
        with np.load(p/'lidar'/f'{i:05d}.npz',allow_pickle=False) as l:
            assert abs(float(l['timestamp'])-t)<1e-5 and l['ranges'].shape==(720,)
        for d in e['detections']:
            assert d['source']=='yolo';confidence.append(d['conf'])
        if detector:
            from PIL import Image
            actual=detector.detect_rgb(np.asarray(Image.open(p/'rgb'/f'{i:05d}.png').convert('RGB')))
            assert len(actual)==len(e['detections']),f'YOLO count mismatch at frame {i}'
            for a,b in zip(actual,e['detections']):
                assert abs(a.conf-b['conf'])<1e-5 and np.allclose(a.bbox,b['bbox'],atol=1e-3),f'YOLO mismatch at frame {i}'
    assert np.allclose(np.diff(times),meta['config']['simulation']['dt'])
    path_m=float(np.linalg.norm(np.diff(np.array(xy),axis=0),axis=1).sum())
    assert np.isclose(path_m,summary['path_m'])
    assert np.allclose(events[-1]['command_applied'],0.)
    return {'trial':str(p),'files_verified':len(hashes),'frames':len(rows),'modalities_synchronized':True,
            'path_m_reaggregated':path_m,'mean_confidence_reaggregated':float(np.mean(confidence)) if confidence else None,
            'initially_detected':summary['initially_detected'],'search_forward_steps':summary['search_forward_steps'],
            'success':summary['success'],'reason':summary['reason'],'final_command_zero':True,
            'yolo_reinference_matches':True if detector else None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repeat-yolo',action='store_true')
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    result=verify(a.trial,a.repeat_yolo);write_json(a.out,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
