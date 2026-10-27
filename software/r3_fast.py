"""R3 balanced only: omit unused audit candidates; preserve every selected operation."""
import numpy as np
from scipy import ndimage as ndi
from skimage.restoration import denoise_nl_means
from skimage.filters import sato
from membrane_r3 import noise_sigma, ramp, balance_ridges


def balanced(stack):
    stack=np.asarray(stack,np.float32)
    p=stack[len(stack)//2]
    sigma=float(np.clip(noise_sigma(p),1/1024,.1))
    den=denoise_nl_means(stack,patch_size=3,patch_distance=3,h=1.0*sigma,
        sigma=sigma,fast_mode=True,preserve_range=True,channel_axis=None)[len(stack)//2].astype(np.float32)
    residual_sigma=float(np.clip(noise_sigma(den),1/2048,sigma))
    bg=ndi.grey_opening(den,size=(17,17),mode='reflect')
    positive=np.maximum(den-bg,0)
    significance=ramp(positive,1.5*residual_sigma,4.*residual_sigma)
    gx=ndi.gaussian_filter(den,1.,order=(0,1));gy=ndi.gaussian_filter(den,1.,order=(1,0))
    jxx=ndi.gaussian_filter(gx*gx,2.);jyy=ndi.gaussian_filter(gy*gy,2.);jxy=ndi.gaussian_filter(gx*gy,2.)
    coherence=np.sqrt((jxx-jyy)**2+4*jxy*jxy)/(jxx+jyy+1e-12)
    ridge=sato(den,sigmas=(.8,1.2,1.8,2.4),black_ridges=False,mode='reflect').astype(np.float32)
    ridge=np.clip(ridge/max(float(np.quantile(ridge,.995)),1e-5),0,1)
    ridge_gate=ridge*significance
    coherent=ridge_gate*ramp(coherence,.12,.55)
    return balance_ridges(coherent,ridge_gate)
