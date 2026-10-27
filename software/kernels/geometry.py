# FARP-Seg segmentation kernels.

from __future__ import annotations
import numpy as np

def covariance(moment,voxel=False):
    count=np.maximum(moment[:,0],1);center=moment[:,1:4]/count[:,None]
    cov=np.zeros((len(moment),3,3),np.float64)
    for j,k,index in ((0,0,4),(1,1,5),(2,2,6),(0,1,7),(0,2,8),(1,2,9)):
        cov[:,j,k]=moment[:,index]/count-center[:,j]*center[:,k];cov[:,k,j]=cov[:,j,k]
    if voxel:cov[:,np.arange(3),np.arange(3)]+=1/12
    return cov,center

def descriptors(moment,surface):
    v=np.maximum(moment[:,0],1);surface=np.maximum(surface,1)
    cov,_=covariance(moment,True);ev=np.maximum(np.linalg.eigvalsh(cov),1e-9)
    xy=np.maximum(np.linalg.eigvalsh(cov[:,1:,1:]),1e-9)
    return np.column_stack([np.log1p(v),np.log1p(surface),np.cbrt(np.pi)*(6*v)**(2/3)/surface,
                            ev[:,0]/ev[:,2],ev[:,1]/ev[:,2],
                            v/((4*np.pi/3)*(5**1.5)*np.sqrt(np.prod(ev,axis=1))),
                            np.sqrt(np.maximum(cov[:,0,0],1e-9)/np.maximum(cov[:,1,1]+cov[:,2,2],1e-9)),
                            np.sqrt(xy[:,0]/xy[:,1])])
