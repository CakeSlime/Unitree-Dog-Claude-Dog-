"""Bounded individual PPO training smoke test. Detector-proxy environment ONLY.

Adapted SB3 Monitor/EvalCallback/checkpoint workflow from Test2/q3_rl/train_ppo.py.
No Connor weights are loaded. Output directories must be new; maximum 8192 steps.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from .object_search_env import ObjectSearchEnv


def train(out,steps=2048,seed=601022):
    import torch
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback,EvalCallback,CallbackList
    from stable_baselines3.common.logger import configure
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    env=Monitor(ObjectSearchEnv(),str(out/'training'),info_keywords=('success','coverage_cells','confirmed'))
    ev=Monitor(ObjectSearchEnv())
    ev.reset(seed=seed+100000)
    model=PPO('MlpPolicy',env,n_steps=128,batch_size=64,n_epochs=2,learning_rate=3e-4,
              gamma=.99,gae_lambda=.95,ent_coef=.01,seed=seed,device='cpu',
              policy_kwargs={'net_arch':[64,64]},verbose=0)
    model.set_logger(configure(str(out),['csv']))
    before=torch.nn.utils.parameters_to_vector(model.policy.parameters()).detach().clone()
    start=time.monotonic()
    class Deadline(BaseCallback):
        def _on_step(self):return time.monotonic()-start<90.
    callbacks=CallbackList([CheckpointCallback(save_freq=512,save_path=str(out/'checkpoints'),name_prefix='tyler_ppo'),
                  EvalCallback(ev,eval_freq=512,n_eval_episodes=3,best_model_save_path=str(out/'best'),
                               log_path=str(out/'validation'),deterministic=True,verbose=0),Deadline()])
    try:
        model.learn(total_timesteps=steps,callback=callbacks)
        model.save(out/'final_model')
        after=torch.nn.utils.parameters_to_vector(model.policy.parameters()).detach()
        gradients=[p.grad for p in model.policy.parameters() if p.grad is not None]
        delta=float(torch.norm(after-before))
        finite=bool(torch.isfinite(after).all())
        gradient_finite=bool(gradients and all(torch.isfinite(g).all() for g in gradients))
        gradient_norm=float(sum(float(torch.norm(g))**2 for g in gradients)**.5)
        # Verify load and deterministic inference, without overwriting any artifact.
        loaded=PPO.load(out/'final_model.zip',device='cpu')
        test_env=ObjectSearchEnv();obs,_=test_env.reset(seed=seed+200000)
        action,_=loaded.predict(obs,deterministic=True)
        assert test_env.action_space.contains(action) and np.isfinite(action).all()
        test_env.close()
        with (out/'progress.csv').open() as f:progress=list(csv.DictReader(f))
        losses=[float(r['train/loss']) for r in progress if r.get('train/loss')]
        result={'detector':'FOV/occlusion/dropout proxy; not YOLO','seed':seed,'requested_steps':steps,
                'actual_steps':model.num_timesteps,'ppo_updates':model._n_updates,
                'wall_s':time.monotonic()-start,'torch':torch.__version__,'sb3':sb3.__version__,
                'weights_changed_l2':delta,'weights_finite':finite,'gradients_finite':gradient_finite,
                'last_gradient_norm':gradient_norm,'finite_logged_losses':bool(losses and np.isfinite(losses).all()),
                'checkpoint_count':len(list((out/'checkpoints').glob('*.zip'))),
                'reload_inference_ok':True,'final_model_sha256':hashlib.sha256((out/'final_model.zip').read_bytes()).hexdigest(),
                'useful_learning_claim':False,'environment_config':env.unwrapped.cfg,
                'hyperparameters':{'n_steps':128,'batch_size':64,'n_epochs':2,'lr':.0003,'net_arch':[64,64],
                                   'gamma':.99,'gae_lambda':.95,'entropy':.01,'device':'cpu'}}
        result['smoke_passed']=bool(delta>0 and finite and gradient_finite and gradient_norm>0
                                    and model.num_timesteps==steps and model._n_updates>0
                                    and result['finite_logged_losses'] and result['checkpoint_count']>0
                                    and result['reload_inference_ok'])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(9,3))
        axes[0].plot([float(r['time/total_timesteps']) for r in progress if r.get('train/loss')],losses)
        axes[0].set(xlabel='steps',ylabel='PPO loss',title='Short smoke run; detector proxy')
        with (out/'training.monitor.csv').open() as f:
            episodes=list(csv.DictReader(line for line in f if not line.startswith('#')))
        axes[1].plot([float(r['r']) for r in episodes],marker='.')
        axes[1].set(xlabel='training episode',ylabel='return',title='Not evidence of useful learning yet')
        fig.tight_layout();fig.savefig(out/'smoke_learning_curves.png');plt.close(fig)
        (out/'smoke_summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
        return result
    finally:
        env.close();ev.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--steps',type=int,default=2048)
    p.add_argument('--seed',type=int,default=601022);a=p.parse_args()
    if not 512<=a.steps<=8192 or a.steps%128:p.error('Smoke test must be 512..8192 steps, divisible by 128')
    r=train(a.out,a.steps,a.seed)
    raise SystemExit(0 if r['smoke_passed'] else 2)


if __name__=='__main__':main()
