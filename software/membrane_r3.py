"""Non-learning membrane input restoration. No segmentation, Reference, or scores."""
import numpy as np
from scipy import ndimage as ndi
from skimage.restoration import denoise_nl_means,richardson_lucy,wiener
from skimage.filters import sato,threshold_sauvola,apply_hysteresis_threshold
from skimage.feature import canny

METHODS=('nlm','rl','wiener','local_gate','sauvola','ridge_gate','coherent','hysteresis','fusion','canny_control')
def ramp(v,low,high):
 t=np.clip((v-low)/max(float(high-low),1e-6),0,1);return t*t*(3-2*t)
def normalize(p,low=.10,high=.995):
 lo,hi=np.quantile(p,[low,high]);return np.clip((p-lo)/max(float(hi-lo),.005),0,1).astype(np.float32)
def gaussian_psf(sigma=1.,radius=4):
 x=np.arange(-radius,radius+1);p=np.exp(-(x[:,None]**2+x[None,:]**2)/(2*sigma**2));return p/p.sum()

def balance_ridges(coherent,ridge_gate):
 """Brighten supported membranes without lifting arbitrarily weak residual texture."""
 return np.power(np.maximum(coherent,.3*ridge_gate),.6)*ramp(ridge_gate,.025,.10)

def noise_sigma(p):
 """MAD of normalized diagonal Haar coefficients in input-derived nonempty support."""
 p=np.asarray(p,np.float32);h=p.shape[0]//2*2;w=p.shape[1]//2*2;a=p[:h,:w];detail=(a[::2,::2]-a[1::2,::2]-a[::2,1::2]+a[1::2,1::2])/2
 smooth=ndi.gaussian_filter(a,1.);lo,hi=np.quantile(a,[.1,.99]);mask=(smooth>lo+.08*(hi-lo))[::2,::2];values=detail[mask] if np.count_nonzero(mask)>32 else detail.ravel();return float(np.median(np.abs(values-np.median(values)))/.67449)

def restore(stack, audit_only=False):
 """Return all candidate grayscale fields and input-only configuration metadata."""
 stack=np.asarray(stack,np.float32);p=stack[len(stack)//2];sigma=float(np.clip(noise_sigma(p),1/1024,.1))
 den=denoise_nl_means(stack,patch_size=3,patch_distance=3,h=1.0*sigma,sigma=sigma,fast_mode=True,preserve_range=True,channel_axis=None)[len(stack)//2].astype(np.float32)
 residual_sigma=float(np.clip(noise_sigma(den),1/2048,sigma))
 bg=ndi.grey_opening(den,size=(17,17),mode='reflect');positive=np.maximum(den-bg,0);upper=ndi.gaussian_filter(ndi.maximum_filter(positive,size=17,mode='reflect'),2.)
 local=np.clip(positive/np.maximum(upper,max(3*residual_sigma,.01)),0,1)
 significance=ramp(positive,1.5*residual_sigma,4.*residual_sigma)
 gx=ndi.gaussian_filter(den,1.,order=(0,1));gy=ndi.gaussian_filter(den,1.,order=(1,0));jxx=ndi.gaussian_filter(gx*gx,2.);jyy=ndi.gaussian_filter(gy*gy,2.);jxy=ndi.gaussian_filter(gx*gy,2.);coherence=np.sqrt((jxx-jyy)**2+4*jxy*jxy)/(jxx+jyy+1e-12)
 ridge=sato(den,sigmas=(.8,1.2,1.8,2.4),black_ridges=False,mode='reflect').astype(np.float32);ridge=np.clip(ridge/max(float(np.quantile(ridge,.995)),1e-5),0,1)
 ridge_gate=ridge*significance
 local_gate=local*significance*ramp(local,.10,.38)
 coherent=ridge_gate*ramp(coherence,.12,.55)
 # Coherence suppresses round speckles but junctions can have low coherence.
 # The fusion candidate retains an intensity-supported junction contribution.
 fusion=np.maximum(coherent,.70*local_gate)*np.maximum(ramp(ridge,.05,.22),ramp(local,.55,.88))
 fusion=np.clip(fusion,0,1)**.7
 balanced=balance_ridges(coherent,ridge_gate)
 if audit_only:
  return dict(nlm=normalize(den),ridge_gate=ridge_gate,balanced=balanced,coherent=coherent),dict(noise_sigma_input=sigma,noise_sigma_after_nlm=residual_sigma,reference_read=False,segmentation_run=False,metrics_computed=False)
 signal=normalize(den);padded=np.pad(signal,12,mode='reflect');rl=richardson_lucy(padded,gaussian_psf(1.0),num_iter=10,clip=False,filter_epsilon=1e-6)[12:-12,12:-12];rl=np.clip(rl,0,1)
 wr=wiener(padded,gaussian_psf(1.0),balance=.015,clip=False)[12:-12,12:-12];wr=np.clip(wr,0,1)
 inverted=1-signal;t=threshold_sauvola(inverted,window_size=25,k=.15,r=.5);scan=np.clip((t-inverted)/.12,0,1);scan=ndi.gaussian_filter(scan,.4)
 connected=apply_hysteresis_threshold(np.maximum(coherent,.5*local_gate),.13,.42);labels,n=ndi.label(connected);sizes=np.bincount(labels.ravel());keep=sizes>=10;keep[0]=False;hysteresis=ndi.gaussian_filter(keep[labels].astype(np.float32),.45)
 edge=canny(signal,sigma=1.0,low_threshold=.08,high_threshold=.20)
 outputs=dict(nlm=signal,rl=rl,wiener=wr,local_gate=local_gate,sauvola=scan,ridge_gate=ridge_gate,coherent=coherent,hysteresis=hysteresis,fusion=fusion,canny_control=edge.astype(np.float32))
 outputs={k:np.clip(v,0,1).astype(np.float32) for k,v in outputs.items()}
 return outputs,dict(noise_sigma_input=sigma,noise_sigma_after_nlm=residual_sigma,nlm=dict(patch_size=3,patch_distance=3,h_sigma=1.,neighbor_z_slices=len(stack)),local_background_square=17,gate_sigma=[1.5,4.],ridge_sigmas=[.8,1.2,1.8,2.4],rl=dict(assumed_gaussian_psf_sigma=1.,iterations=10,measured_psf=False),reference_read=False,segmentation_run=False,metrics_computed=False)
