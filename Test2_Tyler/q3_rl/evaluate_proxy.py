"""Bounded proxy-only comparison and visualization. Never labels proxy results as YOLO."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .object_search_env import ObjectSearchEnv


def classical(obs):
    # Uses only observed target features and lidar, not an env/target handle.
    if obs[24]>.5:
        bearing=float(obs[26])*np.pi;distance=float(obs[27])*5
        if distance<1.08 and abs(bearing)<.1:return np.array([0.,0.],np.float32)
        return np.array([1. if abs(bearing)<.2 else 0.,np.clip(4*bearing,-1,1)],np.float32)
    front=float(min(obs[0],obs[1],obs[23]))*3
    if front<.7:return np.array([0.,1.],np.float32)
    return np.array([.8,.15],np.float32)


def evaluate(model,out,episodes=8,seed=801022):
    from stable_baselines3 import PPO
    policy=PPO.load(model,device='cpu')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for name in ('random','classical','ppo'):
        for k in range(episodes):
            env=ObjectSearchEnv(render_mode='rgb_array')
            obs,_=env.reset(seed=seed+k)
            spawn=[float(env.xy[0]),float(env.xy[1]),env.yaw]
            rng=np.random.default_rng(seed+k)
            frames=[];ret=0.
            for step in range(env.cfg['max_steps']):
                if k==0 and step%10==0:frames.append(Image.fromarray(env.render()))
                a=rng.uniform(-1,1,2).astype(np.float32) if name=='random' else classical(obs) if name=='classical' else policy.predict(obs,deterministic=True)[0]
                obs,r,term,trunc,info=env.step(a);ret+=r
                if term or trunc:break
            row={'policy':name,'episode':k,'seed':seed+k,'success':info['success'],'collision':info['collision'],
                 'confirmed':info['confirmed'],'time_s':info['time_s'],'coverage_cells':info['coverage_cells'],
                 'return':ret,'detector':'proxy'}
            row.update(spawn_x=spawn[0],spawn_y=spawn[1],spawn_yaw=spawn[2])
            rows.append(row)
            if frames and k==0:frames[0].save(out/f'{name}_proxy.gif',save_all=True,append_images=frames[1:],duration=150,loop=0)
            env.close()
    with (out/'episodes.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'protocol':'proxy only; independent evaluation seeds; same initial seeds per method',
            'model':str(model),'model_sha256':hashlib.sha256(Path(model).read_bytes()).hexdigest(),
            'episodes_per_policy':episodes,'seed':seed,'policies':{}}
    starts={}
    for row in rows:
        pose=(row['spawn_x'],row['spawn_y'],row['spawn_yaw'])
        if row['seed'] in starts:assert pose==starts[row['seed']]
        starts[row['seed']]=pose
    result['identical_initial_poses_verified']=True
    for name in ('random','classical','ppo'):
        r=[x for x in rows if x['policy']==name]
        result['policies'][name]={'successes':sum(x['success'] for x in r),
            'success_rate':float(np.mean([x['success'] for x in r])),
            'collisions':sum(x['collision'] for x in r),'mean_return':float(np.mean([x['return'] for x in r])),
            'mean_coverage_cells':float(np.mean([x['coverage_cells'] for x in r]))}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,3));ax.bar(list(result['policies']),[v['success_rate'] for v in result['policies'].values()])
    ax.set_ylim(0,1);ax.set_ylabel('Success rate');ax.set_title('Short smoke evaluation: DETECTOR PROXY, not YOLO')
    fig.tight_layout();fig.savefig(out/'proxy_comparison.png');plt.close(fig)
    print(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--episodes',type=int,default=8);a=p.parse_args()
    if not 1<=a.episodes<=30:p.error('Bounded evaluation: 1..30 episodes per policy')
    evaluate(a.model,a.out,a.episodes)


if __name__=='__main__':main()
