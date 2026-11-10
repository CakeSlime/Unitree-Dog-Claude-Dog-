"""Regenerate Phase 2C comparison figures from actual saved evidence only."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def figures(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);root=Path('Test2_Tyler')
    trials={p.parent.name:json.loads(p.read_text()) for p in sorted((root/'data/phase2c').glob('*/summary.json'))}
    n=len(trials);fig,axs=plt.subplots((n+2)//3,3,figsize=(12,3.7*((n+2)//3)),squeeze=False)
    for ax,(name,s) in zip(axs.flat,trials.items()):
        p=root/'data/phase2c'/name;meta=json.loads((p/'metadata.json').read_text());cfg=meta['config']['simulation']
        path=json.loads((p/'path.json').read_text());xy=np.array([x[1:3] for x in path]);ax.plot(xy[:,0],xy[:,1],label='actual path')
        ax.scatter(*xy[0],marker='o');ax.scatter(*xy[-1],marker='x')
        for b in cfg.get('room_obstacles',[]):ax.add_patch(Rectangle((b[0]-b[3],b[1]-b[4]),2*b[3],2*b[4],color='gray',alpha=.4))
        if not cfg.get('hide_chair'):ax.scatter(*cfg['chair_pose'][:2],marker='*',color='red',label='target: evaluation only')
        ax.set(title=f"{name}\n{s['reason']}; success={s['success']}",xlabel='x (m)',ylabel='y (m)');ax.axis('equal');ax.grid(alpha=.25);ax.legend(fontsize=6)
    for ax in list(axs.flat)[n:]:ax.set_visible(False)
    fig.suptitle('Actual YOLO / physical MuJoCo development trials — all completed attempts');fig.tight_layout();fig.savefig(out/'actual_paths.png',dpi=150);plt.close(fig)
    # Range errors are evaluated at the recorded sample pixel, not an object-centre oracle.
    errors={}
    for name in ('visible_20261110','near_20261110'):
        s=json.loads((root/'data/phase2b'/name/'summary.json').read_text());errors['2B '+name]=[s['scoring_only_reference']['range_error_m']]
    for name in trials:
        rs=json.loads((root/'data/phase2c'/name/'evaluation_references.json').read_text())
        e=[r['range_error_m'] for f in rs for r in f['references'] if r['ray_hit_chair'] and r['range_error_m'] is not None]
        if e:errors['2C '+name]=e
    fig,ax=plt.subplots(figsize=(10,4))
    for i,(name,e) in enumerate(errors.items()):ax.scatter(np.full(len(e),i),np.array(e)*1000,s=8,alpha=.35);ax.plot(i,np.median(e)*1000,'rx')
    ax.axhline(0,color='black',lw=.5);ax.set_xticks(range(len(errors)),list(errors),rotation=25,ha='right');ax.set_ylabel('estimated minus visual reference (mm)');ax.set_title('Semantic chair-hit rays only; ideal registered simulated sensor, not hardware accuracy')
    fig.tight_layout();fig.savefig(out/'range_error_comparison.png',dpi=150);plt.close(fig)
    proxy={}
    for name in ('proxy_feasibility_001','proxy_feasibility_002'):
        proxy[name]=json.loads((root/'results/phase2c'/name/'metrics.json').read_text())
    fig,ax=plt.subplots(figsize=(9,4));settings=list(next(iter(proxy.values()))['settings'])
    for j,(name,metric) in enumerate(proxy.items()):
        vals=[metric['settings'][s]['successes'] for s in settings];ax.bar(np.arange(4)+(j-.5)*.35,vals,width=.35,label=f"seeds {metric['seed']}..{metric['seed']+7}")
    ax.set_xticks(range(4),settings,rotation=15,ha='right');ax.set_ylim(0,8);ax.set_ylabel('successes / 8 proxy episodes');ax.set_title('Baseline/budget feasibility, not PPO training or actual YOLO');ax.legend();fig.tight_layout();fig.savefig(out/'proxy_feasibility.png',dpi=150);plt.close(fig)
    # Historical overlays and measured collision-depth maps, untouched originals.
    diag=root/'results/phase2c/depth_diagnostics_002'
    if diag.exists():
        names=['occluded_001','occluded_20261110','visible_20261110','near_20261110']
        fig,axs=plt.subplots(4,2,figsize=(10,12))
        for i,name in enumerate(names):
            axs[i,0].imshow(Image.open(diag/f'{name}_annotated.png'));axs[i,0].set_title(name+' RGB / old upper sample')
            with np.load(diag/f'{name}_depth.npz') as d:z=d['depth_m']
            im=axs[i,1].imshow(np.ma.masked_invalid(z),vmin=0,vmax=5,cmap='viridis');axs[i,1].set_title('legacy collision optical-z (m)')
            for a in axs[i]:a.axis('off')
        fig.colorbar(im,ax=axs[:,1].tolist(),shrink=.6);fig.subplots_adjust(hspace=.25);fig.savefig(out/'historical_depth_diagnostics.png',dpi=140);plt.close(fig)
    gate=root/'results/phase2c/component_gate_001'
    fig,axs=plt.subplots(1,3,figsize=(12,3.5));axs[0].imshow(Image.open(gate/'rgb.png'));axs[0].set_title('Actual rendered RGB')
    with np.load(gate/'sensors.npz') as d:
        for ax,key in zip(axs[1:],('legacy_depth','registered_depth')):
            ax.imshow(np.ma.masked_invalid(d[key]),vmin=0,vmax=4,cmap='viridis');ax.set_title(key+' optical-z (m)')
    for ax in axs:ax.axis('off')
    fig.tight_layout();fig.savefig(out/'registered_vs_legacy_depth.png',dpi=150);plt.close(fig)
    statistics={name:{'samples':len(e),'median_m':float(np.median(e)),'p95_absolute_m':float(np.quantile(np.abs(e),.95))} for name,e in errors.items()}
    (out/'summary.json').write_text(json.dumps({'trials':trials,'range_error_statistics':statistics,'proxy':proxy},indent=2))
    print(json.dumps({'trials':len(trials),'range_statistics':statistics,'figures':str(out)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();figures(a.out)
