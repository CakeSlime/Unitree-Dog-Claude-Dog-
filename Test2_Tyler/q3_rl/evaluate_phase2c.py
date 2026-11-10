"""Evaluate the calibrated proxy PPO and observable baselines on paired new seeds."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .feasible_env import FeasibleSearchEnv
from .search_baseline import ScanVantageController


def evaluate(model, out, episodes=8, seed=811110):
    from stable_baselines3 import PPO
    import torch
    torch.set_num_threads(1)
    policy = PPO.load(model, device='cpu')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    starts = {}
    started = time.monotonic()
    for name in ('random', 'scan_vantage', 'ppo'):
        for k in range(episodes):
            env = FeasibleSearchEnv()
            obs, _ = env.reset(seed=seed+k)
            pose = [*env.xy.tolist(), env.yaw]
            if k in starts:
                assert pose == starts[k]
            starts[k] = pose
            controller = ScanVantageController()
            rng = np.random.default_rng(seed+k)
            total = 0.
            for step in range(env.cfg['max_steps']):
                if time.monotonic()-started > 90:
                    raise TimeoutError('Evaluation wall budget exceeded')
                action = (rng.uniform(-1, 1, 2).astype(np.float32) if name == 'random'
                          else controller.predict(obs) if name == 'scan_vantage'
                          else policy.predict(obs, deterministic=True)[0])
                assert env.action_space.contains(action)
                obs, reward, terminated, truncated, info = env.step(action)
                assert env.observation_space.contains(obs) and np.isfinite(reward)
                total += reward
                if terminated or truncated:
                    break
            rows.append(dict(policy=name, seed=seed+k, success=info['success'],
                             collision=info['collision'], steps=step+1,
                             time_s=info['time_s'], coverage_cells=info['coverage_cells'],
                             episode_return=total, spawn_x=pose[0], spawn_y=pose[1], spawn_yaw=pose[2]))
            env.close()
    with (out/'episodes.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(protocol='Phase 2C detector PROXY only; final checkpoint; new paired seeds',
                  seed=seed, episodes_per_policy=episodes, identical_initial_poses_verified=True,
                  model=str(model), model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),
                  task_version='phase2c-camera-surface-1200', useful_learning_claim=False,
                  wall_s=time.monotonic()-started, policies={})
    for name in ('random', 'scan_vantage', 'ppo'):
        rs = [r for r in rows if r['policy'] == name]
        result['policies'][name] = dict(successes=sum(r['success'] for r in rs),
            collisions=sum(r['collision'] for r in rs),
            mean_return=float(np.mean([r['episode_return'] for r in rs])),
            mean_coverage_cells=float(np.mean([r['coverage_cells'] for r in rs])))
    (out/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=811110)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 12:
        parser.error('Use 1..12 evaluation episodes per method')
    evaluate(args.model, args.out, args.episodes, args.seed)
