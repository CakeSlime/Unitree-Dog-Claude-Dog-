"""Dependency-light adaptations of shared occupancy/frontier/Dijkstra algorithms.

Sources: Phase_2_Investigator/navigation/{occupancy,frontier,planner}. The sensor
contracts/log-odds conventions are retained. OpenCV EDT/components + heapq replace
SciPy/scikit-image so network/package failures need not block the actual robot demo.
No full-scene geometry or target location is used.
"""
import heapq
import cv2
import numpy as np


class OccupancyGrid:
    def __init__(self,origin,size,res=.1):
        self.origin=np.asarray(origin,float);self.res=res
        self.shape=(round(size[1]/res),round(size[0]/res))
        self.logodds=np.zeros(self.shape,np.float32);self.seen=np.zeros(self.shape,bool)
        self.low_hits=np.zeros(self.shape,np.int16)
    def to_cell(self,xy):
        ij=np.floor((np.asarray(xy)-self.origin)/self.res).astype(int)
        return ij[...,1],ij[...,0]
    def to_xy(self,r,c):return np.stack([self.origin[0]+(np.asarray(c)+.5)*self.res,self.origin[1]+(np.asarray(r)+.5)*self.res],axis=-1)
    def inside(self,r,c):return (r>=0)&(r<self.shape[0])&(c>=0)&(c<self.shape[1])
    def _add(self,points,delta):
        if not len(points):return
        r,c=self.to_cell(points);cells=np.unique(np.stack([r,c],axis=1),axis=0)
        cells=cells[self.inside(cells[:,0],cells[:,1])]
        r,c=cells.T;self.logodds[r,c]=np.clip(self.logodds[r,c]+delta,-2,3.5);self.seen[r,c]=True
    def clear_disk(self,xy,radius):
        r,c=self.to_cell(xy);rr,cc=np.indices(self.shape)
        mask=(rr-r)**2+(cc-c)**2<=(radius/self.res)**2
        self.logodds[mask]=-1.2;self.seen[mask]=True
    def update(self,scan,max_range=2.8):
        T=scan.T_world_lidar;origin=T[:3,3]
        rays=np.stack([np.cos(scan.angles),np.sin(scan.angles),np.zeros_like(scan.angles)],axis=1)@T[:3,:3].T
        rays=rays[:,:2]/np.maximum(np.linalg.norm(rays[:,:2],axis=1,keepdims=True),1e-9)
        ranges=np.asarray(scan.ranges,float);hit=np.isfinite(ranges)&(ranges<=max_range)
        reach=np.where(hit,ranges,max_range)
        distances=np.arange(0,max_range,self.res*.5)
        points=origin[:2]+rays[:,None,:]*distances[None,:,None]
        self._add(points[distances[None,:]<reach[:,None]-self.res],-.4)
        self._add(origin[:2]+rays[hit]*ranges[hit,None],.85)
        self.add_low_points(scan.points_world(),origin[:2],max_range)
    def add_low_points(self,points,origin_xy,max_range=3.):
        p=np.asarray(points)
        if not len(p):return
        valid=(p[:,2]>.07)&(p[:,2]<.7)&(np.linalg.norm(p[:,:2]-origin_xy,axis=1)<max_range)
        r,c=self.to_cell(p[valid,:2]);cells=np.unique(np.stack([r,c],axis=1),axis=0)
        cells=cells[self.inside(cells[:,0],cells[:,1])]
        if len(cells):self.low_hits[cells[:,0],cells[:,1]]+=1
    def occupied(self):return (self.logodds>.6)|(self.low_hits>=2)
    def free(self):return self.seen&(self.logodds<-.3)&~self.occupied()
    def unknown(self):return ~self.seen
    def obstacle_distance(self):
        return cv2.distanceTransform((~self.occupied()).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)*self.res
    def costmap(self,robot_radius=.32):
        distance=self.obstacle_distance()
        cost=1+4*np.clip((.65-distance)/(.65-robot_radius),0,1)**2
        cost[(distance<=robot_radius)|~self.free()]=np.inf
        return cost
    def occupied_points_near(self,xy,radius=1.5):
        r,c=np.nonzero(self.occupied());p=self.to_xy(r,c)
        return p[np.linalg.norm(p-xy,axis=1)<=radius]


def frontier_clusters(grid,min_cells=3):
    neighbors=cv2.dilate(grid.unknown().astype(np.uint8),np.ones((3,3),np.uint8))>0
    n,labels=cv2.connectedComponents((grid.free()&neighbors).astype(np.uint8),connectivity=8)
    return [p for k in range(1,n) if len(p:=np.argwhere(labels==k))>=min_cells]


class CostField:
    def __init__(self,grid,cost,start_xy):
        self.grid=grid;self.start=tuple(map(int,grid.to_cell(start_xy)))
        self.cum=np.full(grid.shape,np.inf);self.cum[self.start]=0.
        self.parents={};queue=[(0.,*self.start)]
        while queue:
            d,r,c=heapq.heappop(queue)
            if d>self.cum[r,c]:continue
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
                nr,nc=r+dr,c+dc
                if not grid.inside(nr,nc) or not np.isfinite(cost[nr,nc]):continue
                if dr and dc and (not np.isfinite(cost[r,nc]) or not np.isfinite(cost[nr,c])):continue
                nd=d+float(cost[nr,nc])*np.hypot(dr,dc)
                if nd<self.cum[nr,nc]:
                    self.cum[nr,nc]=nd;self.parents[nr,nc]=(r,c);heapq.heappush(queue,(nd,nr,nc))
    def path_to(self,rc):
        rc=tuple(map(int,rc))
        if not np.isfinite(self.cum[rc]):return None
        cells=[rc]
        while cells[-1]!=self.start:cells.append(self.parents[cells[-1]])
        cells=np.array(cells[::-1]);return self.grid.to_xy(cells[:,0],cells[:,1])
