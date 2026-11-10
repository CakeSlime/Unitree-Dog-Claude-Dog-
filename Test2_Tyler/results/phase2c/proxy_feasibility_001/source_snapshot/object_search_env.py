"""Tyler's independent Gymnasium object-search foundation (detector PROXY).

Reference: Test2/q3_rl/crime_scene_env.py for Gymnasium/SB3 episode organization.
This task/geometry/reward code is new: no oracle semantic goal direction or route
hint is observed. A target is detected only inside a noisy, occlusion-tested FOV.
Actual YOLO is evaluated separately in Test2_Tyler.search, not simulated here.
"""
from types import SimpleNamespace
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .observations import observable_vector


DEFAULTS={'dt':.2,'max_steps':180,'lidar_rays':24,'sensor_range':3.,'detect_range':2.7,
          'half_fov':.61,'dropout':.08,'radius':.25,'standoff':.95,'tolerance':.15,
          'confirm_frames':3,'inspect_frames':3,
          'reward':{'explore':.25,'discover':1.,'confirm':3.,'progress':1.,'inspect':15.,
                    'collision':5.,'time':.02,'smooth':.015}}


def reward_terms(weights,new_cell,new_detection,new_confirmation,visible_progress,success,collision,action_delta):
    return {'explore':weights['explore']*int(new_cell),
            'discover':weights['discover']*int(new_detection),
            'confirm':weights['confirm']*int(new_confirmation),
            'progress':weights['progress']*float(np.clip(visible_progress,-.2,.2)),
            'inspect':weights['inspect']*int(success),
            'collision':-weights['collision']*int(collision),
            'time':-weights['time'],
            'smooth':-weights['smooth']*float(np.dot(action_delta,action_delta))}


class ObjectSearchEnv(gym.Env):
    metadata={'render_modes':['rgb_array'],'render_fps':5}

    def __init__(self,config=None,render_mode=None):
        import copy
        self.cfg=copy.deepcopy(DEFAULTS)
        if config:
            self.cfg.update({k:v for k,v in config.items() if k!='reward'})
            self.cfg['reward'].update(config.get('reward',{}))
        if self.cfg['max_steps']<1 or not 0<=self.cfg['dropout']<1:
            raise ValueError('Invalid episode/sensor configuration')
        if self.cfg['lidar_rays']!=24:
            raise ValueError('This encoder requires exactly 24 lidar rays')
        self.observation_space=spaces.Box(-1.,1.,shape=(38,),dtype=np.float32)
        self.action_space=spaces.Box(-1.,1.,shape=(2,),dtype=np.float32)
        self.render_mode=render_mode
        self.width,self.height=8.,6.
        self.finished=False

    def _blocked(self,points,margin=0.):
        p=np.asarray(points)
        out=(p[...,0]<margin)|(p[...,0]>self.width-margin)|(p[...,1]<margin)|(p[...,1]>self.height-margin)
        for x0,y0,x1,y1 in self.obstacles:
            out|=(p[...,0]>=x0-margin)&(p[...,0]<=x1+margin)&(p[...,1]>=y0-margin)&(p[...,1]<=y1+margin)
        return out

    def _sample_free(self):
        for _ in range(2000):
            p=self.np_random.uniform([.6,.6],[self.width-.6,self.height-.6])
            if not self._blocked(p,.4):return p
        raise RuntimeError('No valid randomized free pose')

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        opts=options or {}
        jitter=float(self.np_random.uniform(-.25,.25))
        self.obstacles=[(3.7+jitter,0.,4.05+jitter,3.4),(1.4,3.7,2.3,4.5),(5.6,2.0,6.5,2.7)]
        if opts.get('layout')=='open':self.obstacles=[]
        self._target_xy=np.asarray(opts['target'],float) if 'target' in opts else self._sample_free()
        self.xy=np.asarray(opts['spawn'][:2],float) if 'spawn' in opts else self._sample_free()
        if 'spawn' not in opts:
            for _ in range(200):
                if np.linalg.norm(self.xy-self._target_xy)>2.:break
                self.xy=self._sample_free()
        self.yaw=float(opts['spawn'][2]) if 'spawn' in opts else float(self.np_random.uniform(-np.pi,np.pi))
        if self._blocked(self.xy,self.cfg['radius']) or self._blocked(self._target_xy,.1):
            raise ValueError('Spawn or scoring target lies in an obstacle')
        self.velocity=np.zeros(3);self.previous=np.zeros(2)
        self.t=0;self.finished=False;self.hits=0;self.hold=0
        self.confirmed=False;self.ever_seen=False
        self.visited=np.zeros((24,32),bool)
        self._visit()
        self.target=None
        self._sense()
        self.last_range=self.target.range_m if self.target else None
        self.path=[self.xy.copy()]
        self.total_collisions=0
        return self._observation(),self._info()

    def _visit(self):
        c=min(31,max(0,int(self.xy[0]/.25)));r=min(23,max(0,int(self.xy[1]/.25)))
        new=not self.visited[r,c];self.visited[r,c]=True
        return new

    def _sense(self):
        angles=self.yaw+np.arange(24)*2*np.pi/24
        distances=np.arange(.05,self.cfg['sensor_range']+.001,.05)
        pts=self.xy+distances[None,:,None]*np.stack([np.cos(angles),np.sin(angles)],axis=1)[:,None,:]
        hit=self._blocked(pts)
        self.ranges=np.where(hit.any(axis=1),distances[hit.argmax(axis=1)],self.cfg['sensor_range'])
        d=self._target_xy-self.xy
        distance=float(np.linalg.norm(d))
        bearing=float((math.atan2(d[1],d[0])-self.yaw+np.pi)%(2*np.pi)-np.pi)
        line=self.xy+np.linspace(0,1,max(2,int(distance/.05)))[:,None]*d
        visible=distance<=self.cfg['detect_range'] and abs(bearing)<=self.cfg['half_fov'] and not self._blocked(line).any()
        self.target=None
        if visible and self.np_random.random()>=self.cfg['dropout']:
            # Clearly identified proxy, not an actual detector or unseen-target hint.
            self.target=SimpleNamespace(confidence=.85,bearing=bearing,range_m=distance)

    def _observation(self):
        previous_physical=[.18*max(0.,self.previous[0]),.35*self.previous[1]]
        base=observable_vector(self.ranges,self.target,self.velocity,previous_physical)
        local=[]
        for dx,dy in ((.5,0),(0,.5),(-.5,0),(0,-.5)):
            c=int(np.clip((self.xy[0]+dx)/.25,0,31));r=int(np.clip((self.xy[1]+dy)/.25,0,23))
            local.append(float(self.visited[r,c]))
        return np.concatenate([base,np.asarray(local,np.float32)]).astype(np.float32)

    def _info(self,success=False,collision=False,terms=None):
        return {'success':bool(success),'is_success':bool(success),'collision':bool(collision),
                'time_s':self.t*self.cfg['dt'],'confirmed':bool(self.confirmed),
                'coverage_cells':int(self.visited.sum()),'detector_source':'noisy-FOV-occlusion PROXY',
                'reward_terms':terms or {},'target_detected':self.target is not None}

    def step(self,action):
        if self.finished:raise RuntimeError('Reset required after terminated/truncated episode')
        a=np.asarray(action,float)
        if a.shape!=(2,) or not np.isfinite(a).all():raise ValueError('Expected finite 2-vector action')
        a=np.clip(a,-1,1)
        requested=np.array([.18*max(0.,a[0]),0.,.35*a[1]])
        collided=False
        h=self.cfg['dt']/4
        for _ in range(4):
            self.velocity+=(requested-self.velocity)*(1-np.exp(-h/.15))
            yaw=(self.yaw+h*self.velocity[2]+np.pi)%(2*np.pi)-np.pi
            xy=self.xy+h*self.velocity[0]*np.array([np.cos(yaw),np.sin(yaw)])
            if self._blocked(xy,self.cfg['radius']):
                collided=True;self.velocity[:]=0;break
            self.xy,self.yaw=xy,float(yaw)
        self.t+=1
        new_cell=self._visit()
        self._sense()
        new_detection=self.target is not None and not self.ever_seen
        self.ever_seen|=self.target is not None
        self.hits=self.hits+1 if self.target else 0
        new_confirm=not self.confirmed and self.hits>=self.cfg['confirm_frames']
        self.confirmed|=new_confirm
        in_band=(self.target is not None and abs(self.target.range_m-self.cfg['standoff'])<=self.cfg['tolerance']
                 and abs(self.target.bearing)<.1 and abs(self.velocity[0])<.06 and self.confirmed)
        self.hold=self.hold+1 if in_band else 0
        success=self.hold>=self.cfg['inspect_frames'] and not collided
        progress=(self.last_range-self.target.range_m) if self.last_range is not None and self.target else 0.
        terms=reward_terms(self.cfg['reward'],new_cell,new_detection,new_confirm,progress,success,collided,a-self.previous)
        self.previous=a.copy()
        self.last_range=self.target.range_m if self.target else None
        self.total_collisions+=int(collided)
        terminated=bool(success or collided)
        truncated=bool(not terminated and self.t>=self.cfg['max_steps'])
        self.finished=terminated or truncated
        self.path.append(self.xy.copy())
        return self._observation(),float(sum(terms.values())),terminated,truncated,self._info(success,collided,terms)

    def render(self):
        if self.render_mode!='rgb_array':return None
        from PIL import Image,ImageDraw
        im=Image.new('RGB',(480,360),'white');d=ImageDraw.Draw(im)
        for x0,y0,x1,y1 in self.obstacles:d.rectangle((x0*60,y0*60,x1*60,y1*60),fill='gray')
        p=self._target_xy*60;d.ellipse((p[0]-6,p[1]-6,p[0]+6,p[1]+6),fill='green')
        path=[tuple(p*60) for p in self.path]
        if len(path)>1:d.line(path,fill='blue',width=2)
        p=self.xy*60;d.ellipse((p[0]-7,p[1]-7,p[0]+7,p[1]+7),fill='orange')
        return np.asarray(im)
