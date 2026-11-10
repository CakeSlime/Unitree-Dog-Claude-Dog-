"""Bounded baseline feasibility and budget ablation. Proxy evidence ONLY."""
import argparse,csv,json,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
from .object_search_env import ObjectSearchEnv
from .feasible_env import FeasibleSearchEnv
from .search_baseline import ScanVantageController
from .evaluate_proxy import classical


def evaluate(out,episodes=8,seed=902031):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);rows=[];started=time.monotonic()
    settings=[('legacy_reactive_180',ObjectSearchEnv,180,False),('legacy_scan_180',ObjectSearchEnv,180,True),
              ('legacy_scan_1200',ObjectSearchEnv,1200,True),('calibrated_scan_1200',FeasibleSearchEnv,1200,True)]
    for name,cls,budget,use_scan in settings:
        for k in range(episodes):
            env=cls({'max_steps':budget},render_mode='rgb_array');obs,_=env.reset(seed=seed+k)
            controller=ScanVantageController();initial_xy=env.xy.copy();initial_yaw=env.yaw;target_xy=env._target_xy.copy();frames=[]
            ret=0.;seen=0;moving=0;commands=[]
            for step in range(budget):
                if time.monotonic()-started>90:raise TimeoutError('90-second evaluation budget')
                if step%20==0:frames.append(Image.fromarray(env.render()))
                action=controller.predict(obs) if use_scan else classical(obs)
                assert env.action_space.contains(action)
                obs,r,term,trunc,info=env.step(action)
                assert env.observation_space.contains(obs) and np.isfinite(r)
                ret+=r;seen+=int(info['target_detected']);moving+=int(action[0]>.05)
                commands.append(action.tolist())
                if term or trunc:break
            row={'setting':name,'seed':seed+k,'success':info['success'],'collision':info['collision'],
                 'termination':'success' if info['success'] else 'collision' if term else 'time_limit',
                 'steps':step+1,'duration_s':info['time_s'],'return':ret,'confirmed':info['confirmed'],
                 'detection_steps':seen,'moving_command_steps':moving,'coverage_cells':info['coverage_cells'],
                 'path_m':float(np.linalg.norm(np.diff(env.path,axis=0),axis=1).sum()),
                 'initial_x':float(initial_xy[0]),'initial_y':float(initial_xy[1]),
                 'initial_yaw':initial_yaw,
                 'eval_only_initial_target_distance_m':float(np.linalg.norm(initial_xy-target_xy)),
                 'eval_only_translation_lower_bound_s':max(0.,float(np.linalg.norm(initial_xy-target_xy))-.95)/.18}
            rows.append(row)
            # Save both first-seed and every success; no cherry-picked success count.
            if k==0 or info['success']:
                if frames:frames[0].save(out/f'{name}_seed{seed+k}.gif',save_all=True,append_images=frames[1:],duration=150,loop=0)
                (out/f'{name}_seed{seed+k}_trace.json').write_text(json.dumps({'path':[p.tolist() for p in env.path],'commands':commands}))
            print(json.dumps(row),flush=True);env.close()
    with (out/'episodes.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'protocol':'detector PROXY; new seeds, paired starts, no PPO training','seed':seed,'episodes_per_setting':episodes,
             'wall_s':time.monotonic()-started,'settings':{}}
    for name,_,_,_ in settings:
        rs=[r for r in rows if r['setting']==name]
        summary['settings'][name]={'successes':sum(r['success'] for r in rs),'collisions':sum(r['collision'] for r in rs),
            'mean_path_m':float(np.mean([r['path_m'] for r in rs])),
            'mean_return':float(np.mean([r['return'] for r in rs])),
            'mean_coverage_cells':float(np.mean([r['coverage_cells'] for r in rs]))}
    summary['paired_starts_verified']=all(len({(r['initial_x'],r['initial_y'],r['initial_yaw']) for r in rows if r['seed']==seed+k})==1 for k in range(episodes))
    (out/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    # Preserve the exact individual implementation used in this evaluation.
    import shutil
    (out/'source_snapshot').mkdir()
    for n in ('object_search_env.py','feasible_env.py','search_baseline.py','feasibility.py','evaluate_proxy.py','observations.py'):
        shutil.copy2(Path(__file__).parent/n,out/'source_snapshot'/n)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--episodes',type=int,default=8);p.add_argument('--seed',type=int,default=902031);a=p.parse_args()
    if not 1<=a.episodes<=12:p.error('bounded evaluation: 1..12 seeds')
    evaluate(a.out,a.episodes,a.seed)
