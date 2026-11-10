"""Versioned Phase 2C task: enough exploration time, camera-surface standoff.

Preserves Phase 2B ObjectSearchEnv/checkpoints. Detector remains an explicit proxy.
No privileged target data is added to the 38-dimensional observation or info.
"""
from types import SimpleNamespace
import numpy as np
from .object_search_env import ObjectSearchEnv


class FeasibleSearchEnv(ObjectSearchEnv):
    def __init__(self,config=None,render_mode=None):
        c={'max_steps':1200,'tolerance':.075,'camera_offset_m':.30,'range_uncertainty_m':.025,
           'nominal_tolerance_m':.10}
        c.update(config or {})
        super().__init__(c,render_mode)

    def _sense(self):
        angles=self.yaw+np.arange(24)*2*np.pi/24
        distances=np.arange(.05,self.cfg['sensor_range']+.001,.05)
        pts=self.xy+distances[None,:,None]*np.stack([np.cos(angles),np.sin(angles)],axis=1)[:,None,:]
        hit=self._blocked(pts)
        self.ranges=np.where(hit.any(axis=1),distances[hit.argmax(axis=1)],self.cfg['sensor_range'])
        camera=self.xy+self.cfg['camera_offset_m']*np.array([np.cos(self.yaw),np.sin(self.yaw)])
        delta=self._target_xy-camera;distance=float(np.linalg.norm(delta))
        bearing=float((np.arctan2(delta[1],delta[0])-self.yaw+np.pi)%(2*np.pi)-np.pi)
        line=camera+np.linspace(0,1,max(2,int(distance/.05)))[:,None]*delta
        visible=(distance<=self.cfg['detect_range'] and abs(bearing)<=self.cfg['half_fov']
                 and not self._blocked(line).any())
        self.target=None
        if visible and self.np_random.random()>=self.cfg['dropout']:
            self.target=SimpleNamespace(confidence=.85,bearing=bearing,range_m=distance)
