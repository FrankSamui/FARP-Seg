# FARP-Seg segmentation kernels.

from __future__ import annotations
import gc,hashlib
import numpy as np
from scipy import ndimage as ndi

def flood(mask):
    seed=np.zeros(mask.shape,bool)
    for axis in range(3):
        for index in (0,-1):
            sl=[slice(None)]*3;sl[axis]=index;sl=tuple(sl);seed[sl]=mask[sl]
    result=ndi.binary_propagation(seed,mask=mask);del seed;return result

def exterior_atoms(atoms,reached):
    """Exact face adjacency to voxel-level outside flood, plus array-frame labels."""
    flags=np.zeros(atoms.max()+1,bool);prev=None
    for z in range(atoms.shape[0]):
        a=atoms.frame(z);r=reached[z]
        flags[a[0]]=True;flags[a[-1]]=True;flags[a[:,0]]=True;flags[a[:,-1]]=True
        if z in (0,atoms.shape[0]-1):flags[a]=True
        for aa,bb in ((a[:-1],r[1:]),(a[1:],r[:-1]),(a[:,:-1],r[:,1:]),(a[:,1:],r[:,:-1])):flags[aa[bb]]=True
        if prev is not None:
            flags[prev[r]]=True;flags[a[reached[z-1]]]=True
        prev=a
    flags[0]=False;return flags

def exterior_for(atoms,mapping,cache=None):
    key=hashlib.sha256(np.packbits(mapping==0).tobytes()).hexdigest()
    if cache is not None and key in cache:return cache[key]
    mask=np.empty(atoms.shape,bool)
    for z in range(atoms.shape[0]):mask[z]=mapping[atoms.frame(z)]==0
    reached=flood(mask);del mask;result=exterior_atoms(atoms,reached);del reached;gc.collect()
    if cache is not None:cache[key]=result
    return result

def contacts(mapping,edges,faces):
    a,b=mapping[edges].T;keep=(a>0)&(b>0)&(a!=b)&(faces>0)
    key=(np.minimum(a[keep],b[keep]).astype(np.int64)<<32)|np.maximum(a[keep],b[keep]);keys,inv=np.unique(key,return_inverse=True)
    count=np.bincount(inv,weights=faces[keep]);return np.column_stack([keys>>32,keys&0xffffffff]).astype(int),count

def current_state(mapping,mass,na):
    n=int(mapping.max())+1;volume=np.bincount(mapping,weights=mass,minlength=n);volume[0]=0
    acount=np.bincount(mapping[1:na+1],weights=mass[1:na+1],minlength=n);acount[0]=0
    pure=(volume>0)&(acount==0);pure[0]=False
    return volume,acount,pure

def pa(before,mass,edges,faces,ext,na):
    volume,acount,pure=current_state(before,mass,na);ce,area=contacts(before,edges,faces);extcells=np.zeros(len(volume),bool);extcells[before[ext]]=True;extcells[0]=False
    touchesf=np.zeros(len(volume),bool);neighbors={}
    for (a,b),w in zip(ce,area):
        neighbors.setdefault(int(a),{})[int(b)]=int(w);neighbors.setdefault(int(b),{})[int(a)]=int(w)
        if pure[a]:touchesf[b]=True
        if pure[b]:touchesf[a]=True
    interior=(acount>0)&~extcells&~touchesf
    if not interior.any():raise RuntimeError('No clean interior A-containing cells')
    median=float(np.median(volume[interior]));candidates=set(np.flatnonzero((acount>0)&(volume<.20*median)&(extcells|touchesf)))
    lut=np.arange(len(volume),dtype=np.int32);actions=[]
    for c in sorted(candidates):
        options=[(w,-b,b) for b,w in neighbors.get(c,{}).items() if b not in candidates]
        target=max(options)[2] if options else 0;lut[c]=target
        actions.append(dict(cell=c,target=target,volume=int(volume[c]),ratio=float(volume[c]/median),exterior=bool(extcells[c]),touches_F=bool(touchesf[c])))
    return lut[before],dict(pa_candidates=len(candidates),pa_merged=sum(r['target']>0 for r in actions),pa_deleted=sum(r['target']==0 for r in actions),interior_median=median),actions
