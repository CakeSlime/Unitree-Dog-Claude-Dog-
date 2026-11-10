"""Single-frame actual-YOLO / registered-depth gate, without navigation."""
import argparse
from pathlib import Path
import yaml,numpy as np
from PIL import Image
from .simulation import RegisteredSimulation
from .detector import SearchDetector
from .perception import DepthAssociation
from ..probe import annotated
from ..q2_data_simulation.recording import write_json


def probe(config,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);sim=RegisteredSimulation(config['simulation'])
    try:
        f,scan,state,contacts=sim.sample();d=SearchDetector();ds=d.detect_rgb(f.rgb);ts,q=DepthAssociation().measure(f,ds)
        Image.fromarray(f.rgb).save(out/'rgb.png');annotated(f.rgb,ds).save(out/'annotated.png')
        np.savez_compressed(out/'sensors.npz',registered_depth=f.depth_m,legacy_depth=sim.legacy_frame.depth_m,
                            lidar=scan.ranges,K=f.K,T_world_cam=f.T_world_cam)
        result={'config':config,'detections':ds,'quality':q,'model_sha256':d.sha256,'contacts':len(contacts),
                'scoring_only_references':[sim.score_surface(f,t) for t in ts],
                'passed':bool(ts and not contacts)}
        write_json(out/'gate.json',result);return result
    finally:sim.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',required=True)
    p.add_argument('--spawn',nargs=3,type=float);a=p.parse_args();c=yaml.safe_load(a.config.read_text())
    if a.spawn:c['simulation']['spawn']=a.spawn
    r=probe(c,a.out);print('gate passed:',r['passed']);raise SystemExit(0 if r['passed'] else 2)
