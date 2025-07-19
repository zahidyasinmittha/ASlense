#!/usr/bin/env python
# -*- coding: utf-8 -*-
# live_integration.py – TF32 (#2) + async GCN (#5) + ThreadPool + double-buffer

import os, math, time, threading, queue, csv, cv2, numpy as np         # +csv
from concurrent.futures import ThreadPoolExecutor, Future
import torch, torch.nn.functional as F
from yacs.config import CfgNode as CN

from pose_hrnet import get_pose_net
from config      import cfg
from utils       import pose_process


# ───────────────── label map ──────────────────
LABEL_CSV = "label_map.csv"            # expects 2 columns: word,label
label2word = {}
with open(LABEL_CSV, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        label2word[int(row["label"])] = row["word"]


# ─────────────── global flags ────────────────
torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

DEVICE   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"
stream_gcn = torch.cuda.Stream() if USE_CUDA else None  # for async GCN


# ─────────────── frame grabber ───────────────
class VideoStream:
    def __init__(self, src=0, qsize=64):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video source")
        self.q, self.stopped = queue.Queue(maxsize=qsize), False
        threading.Thread(target=self._reader, daemon=True).start()
    def _reader(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok: self.stop(); break
            try: self.q.put_nowait(f)
            except queue.Full: time.sleep(0.002)
    def read(self):  return self.q.get()
    def more(self):  return not self.q.empty() or not self.stopped
    def stop(self):  self.stopped = True; self.cap.release()


# ─────────────── preprocessing helpers ────────────────
mean = np.array((0.485,0.456,0.406),np.float32)[None,None,None,:]
std  = np.array((0.229,0.224,0.225),np.float32)[None,None,None,:]
def norm_numpy_totensor(a):
    t=((a.astype(np.float32)/255.)-mean)/std
    t=torch.from_numpy(t).permute(0,3,1,2).contiguous()
    return t.pin_memory() if USE_CUDA else t
def stack_flip(img): return np.stack([img,cv2.flip(img,1)],0)
index_mirror=(np.concatenate([[1,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16],
                              [21,22,23,18,19,20],np.arange(40,23,-1),
                              np.arange(50,40,-1),np.arange(51,55),
                              np.arange(59,54,-1),[69,68,67,66,71,70],
                              [63,62,61,60,65,64],np.arange(78,71,-1),
                              np.arange(83,78,-1),[88,87,86,85,84,91,90,89],
                              np.arange(113,134),np.arange(92,113)])-1).astype(np.int64)
def merge_hm(hlist):
    for hm in hlist: hm[1]=torch.flip(hm[1,index_mirror],dims=[2])
    return torch.stack(hlist,0).mean((0,1))
def resize_4(img):
    h,w=img.shape[:2]; return cv2.resize(img,((w+3)//4*4,(h+3)//4*4))
multi_scales=[512,640]


# ─────────────── skeleton helpers ────────────────
sel_j={'27':np.concatenate(([0,5,6,7,8,9,10],[91,95,96,99,100,103,104,107,108,111],
                            [112,116,117,120,121,124,125,128,129,132]))}
pairs=[(5,6),(5,7),(6,8),(8,10),(7,9),(9,11),(12,13),(12,14),(12,16),(12,18),
       (12,20),(14,15),(16,17),(18,19),(20,21),(22,23),(22,24),(22,26),(22,28),
       (22,30),(24,25),(26,27),(28,29),(30,31),(10,12),(11,22)]
idx_from,idx_to=np.array([(a-5,b-5) for a,b in pairs]).T
max_frame=240
def proc_skel(inp):
    sk=inp[:,sel_j['27']]; L=sk.shape[0]
    fp=np.zeros((1,max_frame,len(sel_j['27']),3,1),np.float32)
    if L<max_frame:
        fp[0,:L,:,:,0]=sk
        fp[0,L:,:,:,0]=np.tile(sk,(math.ceil((max_frame-L)/L)+1,1,1))[:max_frame-L]
    else: fp[0]=sk[:max_frame,None]
    fp=np.transpose(fp,(0,3,1,2,4))
    bone=fp.copy(); bone[:,:,:,idx_to,:]-=fp[:,:,:,idx_from,:]
    return fp,bone
def motion(d):
    m=np.zeros_like(d); m[:,:,:-1]=d[:,:,1:]-d[:,:,:-1]; return m


# ─────────────── GCN ensemble ───────────────
import os
def _import(p):
    mod=__import__(p.split('.')[0])
    for s in p.split('.')[1:]: mod=getattr(mod,s)
    return mod
_models_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_CFG=[dict(weight=os.path.join(_models_dir, r"weights\bone weights\bone_wights-checkpoint-epoch245.pt"),data="bone_data"),
            dict(weight=os.path.join(_models_dir, r"weights\joint\joints_weights-checkpoint-epoch293.pt"),data="joint_data"),
            dict(weight=os.path.join(_models_dir, r"weights\bone motion weights\bone_motion_wights-checkpoint-epoch291.pt"),data="bone_motion"),
            dict(weight=os.path.join(_models_dir, r"weights\joints_motion\joint_motion_wights-checkpoint-epoch267.pt"),data="joint_motion")]
SOFT_W_T=torch.tensor([0.30,0.40,0.10,0.20],device=DEVICE).view(-1,1,1)
def _load_gcn(p):
    Model=_import('model.decouple_gcn_attn.Model')
    net=Model(num_class=2207,num_point=27,num_person=1,
              graph='graph.sign_27.Graph',groups=16,block_size=41,
              graph_args={'labeling_mode':'spatial'}).to(DEVICE)
    st=torch.load(p,map_location=DEVICE)
    st=st.get('model_state_dict',st)
    net.load_state_dict({k.replace('module.',''):v for k,v in st.items()},strict=False)
    net.eval(); return net
def load_ensemble(): return [_load_gcn(c['weight']) for c in MODELS_CFG]
@torch.no_grad()
def fused_logits(models,bank):
    probs=[]
    for net,cfg,w in zip(models,MODELS_CFG,SOFT_W_T):
        x=bank[cfg['data']].to(DEVICE,non_blocking=True)
        l=net(x); l=l[0] if isinstance(l,tuple) else l
        probs.append(w*F.softmax(l,1))
    return torch.stack(probs,0).sum(0)     # shape [1,C]


# ─────────────── HRNet worker ───────────────
def hrnet_worker(frame_bgr,hr):
    rgb=resize_4(cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB))
    h,w,_=rgb.shape; h4,w4=h//4,w//4
    hms=[]
    with torch.no_grad():
        for s in multi_scales:
            img=cv2.resize(rgb,(s,s)) if s!=512 else rgb
            inp=norm_numpy_totensor(stack_flip(img)).to(DEVICE,non_blocking=True)
            hm=hr(inp)
            if s!=512: hm=F.interpolate(hm,(h4,w4),mode='bilinear')
            hms.append(hm)
    merged=merge_hm(hms)
    idx=merged.reshape(133,-1).argmax(1).cpu().numpy()
    y,x=idx//w4,idx%w4
    kp=np.stack((x,y,np.ones_like(x)),1).astype(np.float32)
    kp=pose_process(kp,merged.cpu().numpy().reshape(133,h4,w4)); kp[:,:2]*=4
    vis=frame_bgr.copy()
    for px,py,c in kp:
        if c>0.3: cv2.circle(vis,(int(px),int(py)),3,(0,255,0),-1)
    return kp,vis


# ─────────────── main pipeline loop ───────────────
def main():
    WINDOW,SKIP = 20,1
    vs=VideoStream(r"E:\final_dataset_videos\he_video_5.mp4")
    gcn=load_ensemble()
    cfg.defrost(); cfg.merge_from_file("wholebody_w48_384x288.yaml"); cfg.freeze()
    hr=get_pose_net(cfg,is_train=False).to(DEVICE).eval()
    hr.load_state_dict({(k[9:] if k.startswith('backbone.')
                         else k[14:] if k.startswith('keypoint_head.') else k):
                        v for k,v in torch.load("hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth",
                                                map_location="cuda")['state_dict'].items()})

    pool=ThreadPoolExecutor(max_workers=2)

    # Double-buffer: 2 frames in flight
    f0=vs.read(); future=pool.submit(hrnet_worker,f0,hr)
    next_frame=vs.read() if vs.more() else None

    buf=[]; fid=0; gcn_event=gcn_logits=None
    print("[Run] press q / ESC to quit")
    while future:
        kp,vis=future.result(); fid+=1   # frame k

        # Launch HRNet on frame k+1
        if next_frame is not None:
            future=pool.submit(hrnet_worker,next_frame,hr)
            next_frame=vs.read() if vs.more() else None
        else:
            future=None

        # Poll async GCN
        if gcn_event and gcn_event.query():
            top_ids  = torch.topk(gcn_logits,4).indices.tolist()   # [4]
            top_words = [label2word.get(i,f"<{i}>") for i in top_ids]

            cv2.putText(vis,f"Pred: {top_words[0]}",(15,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,255,0),2)
            print(f"[{fid:06d}] Top-1 = {top_words[0]:<20} Top-4 = {top_words}")

            gcn_event=gcn_logits=None

        buf.append(kp)

        # Maybe launch next GCN run
        if len(buf)==WINDOW and USE_CUDA and gcn_event is None:
            joint,bone=proc_skel(np.asarray(buf))
            bank=dict(joint_data=torch.from_numpy(joint).float().pin_memory(),
                      bone_data=torch.from_numpy(bone ).float().pin_memory(),
                      joint_motion=torch.from_numpy(motion(joint)).float().pin_memory(),
                      bone_motion=torch.from_numpy(motion(bone )).float().pin_memory())
            with torch.cuda.stream(stream_gcn):
                gcn_logits=fused_logits(gcn,bank).squeeze(0)  # 1-D ✔
            gcn_event=torch.cuda.Event(); gcn_event.record(stream_gcn)
            buf.clear()

        # Display
        cv2.imshow("Live",vis)
        if cv2.waitKey(1)&0xFF in (27,ord('q')): break

    vs.stop(); pool.shutdown(); cv2.destroyAllWindows(); print("[Done] Bye!")


if __name__=="__main__":
    main()
