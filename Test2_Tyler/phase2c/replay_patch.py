"""Same-frame offline comparison of recorded box sampling and new components.

No navigation is executed. Static scene mesh queries score both measured points
AFTER estimation. The original RGB/depth/boxes and recording remain unchanged.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .simulation import RegisteredSimulation
from .perception import DepthAssociation,CalibratedTarget
from .detector import SearchDetection
from ..q2_data_simulation.recording import write_json


def compare(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    p=Path('Test2_Tyler/data/phase2c/explore_context_001');meta=json.loads((p/'metadata.json').read_text())
    e=json.loads((p/'events.jsonl').read_text().splitlines()[202])
    sim=RegisteredSimulation(meta['config']['simulation'])
    try:
        from sim_env.interfaces import CameraFrame
        rgb=np.asarray(Image.open(p/'rgb/00202.png').convert('RGB'))
        with np.load(p/'depth/00202.npz') as d:frame=CameraFrame(e['t'],rgb,d['depth_m'],d['K'],d['T_world_cam'])
        frame.depth_source='registered_RGBD'
        targets,quality=DepthAssociation().measure(frame,[SearchDetection(**x) for x in e['detections']])
        old=CalibratedTarget(**e['decision']['target'])
        result={'protocol':'offline same-frame comparison; no navigation','input_trial':str(p),'frame':202,
                'recorded_old_target':old,'old_scoring_only':sim.score_surface(frame,old),
                'new_targets':targets,'quality':quality,'new_scoring_only':[sim.score_surface(frame,t) for t in targets]}
        write_json(out/'comparison.json',result)
        im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);draw.rectangle(old.bbox,outline='yellow',width=2)
        for label,t,color in [('recorded box sample',old,'red')]+[('component',t,'lime') for t in targets]:
            u,v=t.sample_pixel;draw.ellipse((u-4,v-4,u+4,v+4),fill=color);draw.text((u+7,v),f'{label}: {t.range_m:.3f} m',fill=color)
        im.save(out/'comparison.png');print(json.dumps({'old_hit_chair':result['old_scoring_only']['ray_hit_chair'],
            'new_hit_chair':[r['ray_hit_chair'] for r in result['new_scoring_only']]},indent=2))
    finally:sim.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();compare(a.out)
