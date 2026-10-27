# FARP-Seg segmentation kernels.

from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from . import proposers as alg
from . import geometry as g

def normalize(a):return np.unique(a,return_inverse=True)[1].astype(np.int32)+1

def connected(atoms,pairs,keep):
    ij=np.searchsorted(atoms,pairs[keep]);n=len(atoms)
    a=sparse.csr_matrix((np.ones(2*len(ij)),(np.r_[ij[:,0],ij[:,1]],np.r_[ij[:,1],ij[:,0]])),shape=(n,n))
    return connected_components(a,directed=False,return_labels=True)[1].astype(np.int32)+1

def select_regions(rows,scores,budget):
    order=np.argsort(-np.asarray(scores),kind='stable');selected=[];used=set()
    for i in order:
        cells=set(rows[i]['clusters'])
        if cells&used:continue
        selected.append(int(i));used.update(cells)
        if len(selected)>=budget:break
    return selected

def proposal_bank(atoms,eids,a):
    edges=a['edges'][eids];prob=a['prob'][eids];current=normalize(a['mapping'][atoms]);maps=[current];families=['current'];seen={current.tobytes()}
    def add(m,name):
        if m is None:return
        m=normalize(m)
        if m.tobytes() not in seen:seen.add(m.tobytes());maps.append(m);families.append(name)
    for t in sorted(set([.10,.25,.40,.55,.70,.80,.90,.95,.98,.995]+[round(float(x),5) for x in np.quantile(prob,[.1,.25,.5,.75,.9])])):add(connected(atoms,edges,prob>=t),'hierarchy')
    for bias in (-1.,-.5,0.,.5,1.):add(alg.gaec_partition(atoms,edges,prob,bias=bias),'gaec')
    for k in (1,2,3,4):add(alg.mst_cut_partition(atoms,edges,prob,k),'mst')
    same=a['mapping'][edges[:,0]]==a['mapping'][edges[:,1]];priority=(1-prob)*(1+np.log1p(a['faces'][eids]));priority[~same]=-np.inf
    for j in np.argsort(priority)[::-1][:4]:
        if not same[j]:continue
        split=alg.seeded_geodesic_split(atoms,a['mapping'],edges,prob,tuple(edges[j]));add(split,'geodesic')
        if split is not None:
            add(alg.gaec_partition(atoms,edges,prob,bias=0.,initial_labels=split),'geodesic_gaec')
            add(alg.gaec_partition(atoms,edges,prob,bias=.5,initial_labels=split),'geodesic_gaec')
    return np.asarray(maps,np.int32),families

def component_summary(labels,slab,slab_atom,z,nz,contacts,pairs):
    nc=int(labels.max());parent=labels[slab_atom];moment=np.zeros((nc+1,10));np.add.at(moment,parent,slab[:,:10]);surface=np.bincount(parent,weights=slab[:,10]+2*slab[:,0]-2*slab[:,11],minlength=nc+1)
    same=labels[pairs[:,0]]==labels[pairs[:,1]];np.add.at(surface,labels[pairs[same,0]],-2*contacts[same])
    assert np.all(surface[1:]>0)
    desc=g.descriptors(moment[1:],surface[1:]);weight=moment[1:,0]/moment[1:,0].sum()
    geo=np.r_[np.average(desc,axis=0,weights=weight),np.min(desc,axis=0),np.max(desc,axis=0)]
    # Whole proposed cells, recomputed for every partition (not pair summaries).
    profile=np.zeros(((nc+1)*nz,3));index=parent*nz+z
    np.add.at(profile,index,slab[:,[0,2,3]]);profile=profile.reshape(nc+1,nz,3)[1:]
    area=profile[:,:,0];present=area>0;cent=profile[:,:,1:]/np.maximum(area[:,:,None],1)
    vals=[]
    def summarize(value,mask):
        selected=value[mask];vals.extend([float(selected.mean()) if len(selected) else 0.,float(np.quantile(selected,.9)) if len(selected) else 0.])
    both=present[:,1:]&present[:,:-1]
    summarize(np.abs(np.diff(area,axis=1))/np.maximum((area[:,1:]+area[:,:-1])*.5,1),both)
    summarize(np.linalg.norm(np.diff(cent,axis=1),axis=2)/np.maximum(np.sqrt((area[:,1:]+area[:,:-1])*.5),1),both)
    for radius in (1,3,6):
        if nz<=2*radius:vals.extend([0.]*4);continue
        mid=area[:,radius:-radius];lo=area[:,:-2*radius];hi=area[:,2*radius:];valid=present[:,radius:-radius]&present[:,:-2*radius]&present[:,2*radius:]
        summarize(np.abs(mid-.5*(lo+hi))/np.maximum(.5*(lo+hi),1),valid)
        summarize(np.linalg.norm(cent[:,radius:-radius]-.5*(cent[:,:-2*radius]+cent[:,2*radius:]),axis=2)/np.maximum(np.sqrt(mid),1),valid)
    spans=np.maximum(np.max(np.where(present,np.arange(nz)+1,0),axis=1)-np.min(np.where(present,np.arange(nz),nz),axis=1),1)
    vals.extend([float(np.average(1-present.sum(axis=1)/spans,weights=weight)),float(np.average(np.log1p(spans),weights=weight)),float(np.std(np.sum(present,axis=0))),float(np.mean(np.abs(np.diff(np.sum(present,axis=0)))))])
    return geo,np.asarray(vals)
