"""Actual YOLO adapter retaining edge-clipped discoveries for centering ONLY.

Preprocessing/weights/CPU inference adapted unchanged from adapters/yolo.py.
Full-frame detections remain necessary for calibrated approach/inspection.
"""
from dataclasses import dataclass
import time
import cv2
import numpy as np
from ..adapters.yolo import ChairDetector
from go2_stream.perception.detector import Detection


@dataclass
class SearchDetection(Detection):
    edge_clipped:bool=False


def decode_discovery(rows,width,height,confidence=.3,scale=1.):
    out=[]
    for x1,y1,x2,y2,score,cls in np.asarray(rows).reshape(-1,6):
        if not np.isfinite([x1,y1,x2,y2,score,cls]).all() or score<confidence or int(cls)!=56:continue
        box=(max(0.,float(x1)/scale),max(0.,float(y1)/scale),min(width-1.,float(x2)/scale),min(height-1.,float(y2)/scale))
        if box[2]-box[0]<20 or box[3]-box[1]<30:continue
        out.append(SearchDetection(56,'chair',float(score),box,edge_clipped=bool(box[0]<=2 or box[2]>=width-3)))
    return out


class SearchDetector(ChairDetector):
    def detect_rgb(self,rgb):
        h,w=rgb.shape[:2];scale=min(640/w,640/h)
        resized=cv2.resize(rgb,(round(w*scale),round(h*scale)))
        padded=np.zeros((640,640,3),np.uint8);padded[:resized.shape[0],:resized.shape[1]]=resized
        x=np.ascontiguousarray(padded.transpose(2,0,1)[None].astype(np.float32)/255.)
        start=time.perf_counter();out=self.session.run(None,{self.input:x})[0]
        self.last_latency_s=time.perf_counter()-start
        if out.shape[-1]!=6:raise ValueError('Unexpected YOLO output shape')
        return decode_discovery(out,w,h,self.confidence,scale)
