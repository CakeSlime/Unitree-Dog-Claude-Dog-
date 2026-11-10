"""RGB-registered simulated depth sensor, independent from scoring mesh rays.

Reuses the shared camera renderer with the SAME camera and visibility mask as RGB.
No semantic geom/body ID or target location enters the depth image. The legacy
collision-ray depth remains separately available for comparison and recording.
"""
import numpy as np
from ..adapters.search_sim import SearchSimulation


class RegisteredSimulation(SearchSimulation):
    def _lights(self,path):
        super()._lights(path)
        if not self.config.get('calibration_room'):return
        import xml.etree.ElementTree as ET
        tree=ET.parse(path);world=tree.getroot().find('worldbody')
        # Outer-scene contact exclusions refer to removed articulated furniture.
        # Robot exclusions live in its unchanged include and remain in effect.
        for block in list(tree.getroot()):
            if block.tag in ('contact','equality','actuator','tendon','sensor'):
                tree.getroot().remove(block)
        # Only this generated overlay is edited. The robot is a separate include;
        # retain the shared textured chair asset and create Tyler's test room.
        for child in list(world):
            name=child.get('name','')
            context=(self.config.get('context_backdrop') and child.tag=='body' and
                     (name.startswith('room_geometry_') or name=='living_room_writing_desk_0'
                      or any(k in name for k in ('print','whiteboard','clock'))))
            if name=='living_room_bed_0' and self.config.get('context_bed_pose'):
                x,y=self.config['context_bed_pose'];child.set('pos',f'{x} {y} -.00199256');context=True
            if child.tag in ('body','geom') and name!='living_room_task_chair_0' and not context:
                world.remove(child)
        if not self.config.get('context_backdrop'):
            ET.SubElement(world,'geom',name='tyler_floor',type='plane',size='10 10 .1',
                          rgba='.55 .43 .30 1',friction='1 .01 .001',group='0')
        boxes=[[3.,-.1,1.,3.1,.1,1.],[3.,5.1,1.,3.1,.1,1.],
               [-.1,2.5,1.,.1,2.6,1.],[6.1,2.5,1.,.1,2.6,1.]]
        boxes+=self.config.get('room_obstacles',[])
        for i,b in enumerate(boxes):
            for group,physical in ((3,True),(1,False)):
                ET.SubElement(world,'geom',name=f'tyler_room_{i}_{group}',type='box',
                    pos=' '.join(map(str,b[:3])),size=' '.join(map(str,b[3:])),
                    rgba='.84 .86 .82 1',group=str(group),contype='1' if physical else '0',
                    conaffinity='1' if physical else '0')
        tree.write(path)

    def sample(self):
        from sim_env.interfaces import CameraFrame
        f,scan,state,contacts=super().sample()
        self.legacy_frame=f
        renderer=self.session.sensors.renderers[self.session.camera_key]
        try:
            renderer.enable_depth_rendering()
            renderer.update_scene(self.session.data,
                camera=self.session.sensors.cameras[self.session.camera_key]['camera'],
                scene_option=self.session.sensors.render_option)
            z=renderer.render().copy()
        finally:
            renderer.disable_depth_rendering()
        z[(z<.15)|(z>8.)|~np.isfinite(z)]=np.inf
        registered=CameraFrame(f.t,f.rgb,z,f.K,f.T_world_cam,f.frame_id)
        registered.depth_source='registered_RGBD'
        return registered,scan,state,contacts
