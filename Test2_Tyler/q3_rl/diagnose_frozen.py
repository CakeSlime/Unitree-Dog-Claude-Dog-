"""Inference-only diagnostics of the preserved 2,048-step policy; no learning."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from .object_search_env import ObjectSearchEnv


def diagnose(model,out,seed=902031):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    digest=hashlib.sha256(Path(model).read_bytes()).hexdigest();policy=PPO.load(model,device='cpu');rows=[]
    for k in range(8):
        env=ObjectSearchEnv();obs,_=env.reset(seed=seed+k);commands=[];seen=0
        for step in range(env.cfg['max_steps']):
            action,_=policy.predict(obs,deterministic=True);commands.append(action.tolist())
            obs,r,term,trunc,info=env.step(action);seen+=int(info['target_detected'])
            if term or trunc:break
        a=np.array(commands)
        rows.append({'seed':seed+k,'success':info['success'],'collision':info['collision'],
                     'steps':step+1,'path_m':float(np.linalg.norm(np.diff(env.path,axis=0),axis=1).sum()),
                     'forward_command_fraction':float((a[:,0]>.05).mean()),'mean_normalized_forward':float(a[:,0].mean()),
                     'mean_abs_normalized_yaw':float(np.abs(a[:,1]).mean()),'detected_frames':seen})
        env.close()
    assert hashlib.sha256(Path(model).read_bytes()).hexdigest()==digest
    result={'protocol':'frozen old checkpoint / old task / deterministic inference ONLY','model_sha256':digest,'episodes':rows}
    (out/'diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',required=True);a=p.parse_args();diagnose(a.model,a.out)
