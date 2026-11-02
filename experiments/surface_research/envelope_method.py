"""Experimental local-envelope marker generator, arbitrary ND arrays.

Not a surface-accuracy guarantee. Local min/max levels are heuristic and can
confuse reconstruction streaks with thin walls. Inputs and output share crop.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed


def remove_tiny(mask, n=3):
    lab, _=ndi.label(mask)
    areas=np.bincount(lab.ravel());keep=areas>=n;keep[0]=False
    return keep[lab]


def local_envelope_markers(smooth, envelope, sizes=(11,21,41), low=.20, high=.80,
                           min_contrast=.05, ridge_sigma=3., ridge_floor=.003,
                           support_padding=24):
    """Return markers {0: unknown, 1: air, 2: material} and diagnostics.

    envelope is only a broad object locator. It does not clip final material:
    support allows extension by support_padding voxels. Local extrema propose
    two sides; contradictory proposals remain unknown. No global gray threshold
    is used to seed material or internal air.
    """
    s=np.asarray(smooth,dtype=np.float32)
    support=ndi.binary_dilation(envelope,iterations=support_padding)
    ridge=s-ndi.gaussian_filter(s,ridge_sigma)
    material_votes=np.zeros(s.shape,np.uint8)
    air_votes=np.zeros(s.shape,np.uint8)
    valid_votes=np.zeros(s.shape,np.uint8)
    qmax=np.zeros(s.shape,np.float32)
    for size in sizes:
        lo=ndi.minimum_filter(s,size=size)
        hi=ndi.maximum_filter(s,size=size)
        rng=hi-lo
        good=rng>=min_contrast
        q=(s-lo)/np.maximum(rng,min_contrast)
        valid_votes+=good
        material_votes+=(q>high)&good
        air_votes+=(q<low)&good
        qmax=np.maximum(qmax,np.where(good,q,0))
    material=(material_votes>=1)&(ridge>ridge_floor)&support
    # Two independent windows must see a local valley. Unlike a global low
    # threshold, this permits bright cavities. Air/material disagreement stays
    # unknown rather than creating an irreversible wrong seed.
    air=(air_votes>=2)&(material_votes==0)&(ridge<-ridge_floor)&support
    conflict=material&air
    material&=~conflict;air&=~conflict
    marks=np.zeros(s.shape,np.uint8)
    marks[~support]=1
    marks[remove_tiny(air)]=1
    marks[remove_tiny(material)]=2
    return marks,dict(support=support,material_votes=material_votes,air_votes=air_votes,
                      conflict=conflict,valid_votes=valid_votes)


def local_envelope_watershed(smooth,envelope,**kwargs):
    marks,diag=local_envelope_markers(smooth,envelope,**kwargs)
    grad=np.sqrt(sum(ndi.gaussian_filter(smooth,.6,order=tuple(int(k==j) for k in range(smooth.ndim)))**2
                     for j in range(smooth.ndim)))
    result=watershed(grad,marks)==2
    return result,marks,diag
