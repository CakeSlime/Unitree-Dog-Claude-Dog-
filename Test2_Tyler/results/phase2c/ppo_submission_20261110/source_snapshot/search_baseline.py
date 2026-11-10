"""Stateful scan/vantage baseline. Accepts observations ONLY, no environment handle.

Odometry integrates sensed body velocities/gyro. Vantages use measured LiDAR and
visit memory; discovered targets use only current proxy detection bearing/range.
"""
import numpy as np


def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi


class ScanVantageController:
    def __init__(self,dt=.2):
        self.dt=dt;self.xy=np.zeros(2);self.yaw=0.;self.t=0.
        self.scan_remaining=2*np.pi;self.goal=None;self.visits={};self.last_seen=-np.inf

    def predict(self,obs):
        obs=np.asarray(obs)
        self.yaw=wrap(self.yaw+float(obs[31])*.5*self.dt)
        self.xy+=float(obs[29])*.4*self.dt*np.array([np.cos(self.yaw),np.sin(self.yaw)])
        self.t+=self.dt
        cell=tuple(np.floor(self.xy/.6).astype(int));self.visits[cell]=self.visits.get(cell,0)+1
        ranges=np.asarray(obs[:24])*3
        if obs[24]>.5:
            self.last_seen=self.t
            bearing=float(obs[26])*np.pi;distance=float(obs[27])*5
            if abs(distance-.95)<=.065 and abs(bearing)<=.08:
                return np.zeros(2,np.float32)
            yaw=np.clip(4*bearing,-1,1)
            vx=np.clip(.7*(distance-.95)/.18,0,1) if abs(bearing)<.12 else 0.
            if min(ranges[[0,1,23]])<.55:vx=0.
            return np.array([vx,yaw],np.float32)
        if self.t-self.last_seen<1.:
            return np.zeros(2,np.float32)
        if self.scan_remaining>0:
            self.scan_remaining-=max(0.,float(obs[31])*.5*self.dt)
            return np.array([0.,1.],np.float32)
        if self.goal is None:
            candidates=[]
            for i in range(24):
                reach=min(1.5,float(min(ranges[(i-1)%24],ranges[i],ranges[(i+1)%24]))-.65)
                if reach<.65:continue
                angle=self.yaw+i*2*np.pi/24
                point=self.xy+reach*np.array([np.cos(angle),np.sin(angle)])
                visit=self.visits.get(tuple(np.floor(point/.6).astype(int)),0)
                candidates.append((visit+.1*abs(wrap(angle-self.yaw))-reach,point))
            if not candidates:
                self.scan_remaining=np.pi;return np.array([0.,1.],np.float32)
            self.goal=min(candidates,key=lambda p:p[0])[1]
        delta=self.goal-self.xy;error=wrap(np.arctan2(delta[1],delta[0])-self.yaw)
        if np.linalg.norm(delta)<.20 or (abs(error)<.3 and min(ranges[[0,1,23]])<.65):
            self.goal=None;self.scan_remaining=2*np.pi
            return np.zeros(2,np.float32)
        return np.array([.9 if abs(error)<.2 else 0.,np.clip(4*error,-1,1)],np.float32)
