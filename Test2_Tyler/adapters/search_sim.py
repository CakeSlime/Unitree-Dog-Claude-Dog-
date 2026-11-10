"""Phase 2B scene variants and scoring, without modifying the Phase 2A adapter.

Hidden geometry is available only for explicit post-decision evaluation. The
exploration controller receives samples, never this adapter/model or scene config.
"""
import numpy as np
from .shared import Go2Simulation


class SearchSimulation(Go2Simulation):
    def _lights(self,path):
        super()._lights(path)
        import xml.etree.ElementTree as ET
        tree=ET.parse(path)
        world=tree.getroot().find('worldbody')
        if self.config.get('hide_chair'):
            chair=world.find(".//body[@name='living_room_task_chair_0']")
            chair.set('pos','50 50 0')
        for i,b in enumerate(self.config.get('occluders',[])):
            ET.SubElement(world,'geom',name=f'tyler_occluder_{i}',type='box',
                          pos=' '.join(map(str,b[:3])),size=' '.join(map(str,b[3:])),
                          rgba='.4 .45 .5 1',group='3',contype='1',conaffinity='1')
            ET.SubElement(world,'geom',name=f'tyler_occluder_visual_{i}',type='box',
                          pos=' '.join(map(str,b[:3])),size=' '.join(map(str,b[3:])),
                          rgba='.4 .45 .5 1',group='1',contype='0',conaffinity='0')
        tree.write(path)

    def score_surface(self,frame,target):
        """Independent mesh-ray reference AFTER the decision; not a control input."""
        if target is None:
            return None
        import mujoco
        u=(target.bbox[0]+target.bbox[2])/2
        v=(target.bbox[1]+target.bbox[3])/2
        if hasattr(target,'sample_pixel'):u,v=target.sample_pixel
        ray_cam=np.array([(u-frame.K[0,2])/frame.K[0,0],(v-frame.K[1,2])/frame.K[1,1],1.])
        ray=frame.T_world_cam[:3,:3]@ray_cam
        ray/=np.linalg.norm(ray)
        gid=np.zeros(1,np.int32)
        # Visual group 1 only; semantic correctness is scored from the hit body.
        dist=mujoco.mj_ray(self.session.model,self.session.data,frame.T_world_cam[:3,3],ray,
                           np.array([0,1,0,0,0,0],np.uint8),1,
                           int(self.session.model.cam_bodyid[self.session.camera_mj_id]),gid)
        body=int(self.session.model.geom_bodyid[gid[0]]) if dist>0 else -1
        names=[]
        while body>0:
            names.append(mujoco.mj_id2name(self.session.model,mujoco.mjtObj.mjOBJ_BODY,body) or '')
            body=int(self.session.model.body_parentid[body])
        return {'reference_ray_range_m':float(dist),'ray_hit_chair':any('task_chair' in n for n in names),
                'range_error_m':float(target.range_m-dist*np.linalg.norm(ray[:2])) if dist>0 else None,
                'reference_horizontal_range_m':float(dist*np.linalg.norm(ray[:2])) if dist>0 else None,
                'reference_definition':'single visual-mesh ray at depth sample_pixel (box centre if absent); scoring only'}
