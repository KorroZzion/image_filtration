"""Adaptive isolated CT edge fitting along supplied physical 3D normals.

This is a research model, not a certified surface measurement. Windows are
6,12,24 reference voxels, where the reference spacing is min(spacing_zyx).
Noise comes from original-volume voxel differences, never oversampled-profile
MAD. Rejected points keep their seed; no surfaces are connected or fabricated.
An asymmetric PSF or reconstruction artefact can bias even an accepted fit.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from scipy.special import ndtr
from scipy.signal import find_peaks

FLAGS = {1:'profile_outside_original_volume',2:'insufficient_step_amplitude_snr',
         4:'opposing_edge_or_nonmonotone_transition',8:'shift_or_sigma_at_bound',
         16:'large_model_residual',32:'unobserved_background_or_material_plateau',
         64:'large_background_slope',128:'ill_conditioned_or_uncertain_location',
         256:'optimizer_nonfinite_or_failed',512:'inconsistent_supported_windows',1024:'multiple_separated_rising_edges'}
PARAMETERS=['background','background_slope_per_reference_voxel','amplitude','shift_vox','sigma_vox']


def estimate_volume_noise(raw, max_samples=250000):
    """Robust voxel-noise estimate under a locally smooth background model.

    Uses original-grid first and second differences with white-noise variance
    normalisation. Taking the larger robust estimate avoids using the tiny MAD
    of trilinearly oversampled profiles. Correlated reconstruction noise and
    low-frequency artefacts are not fully described by this scalar; a supplied
    calibrated noise_sigma is preferable. Edges can inflate this estimate.
    """
    raw=np.asarray(raw,np.float32)
    estimates=[]
    for axis in range(3):
        if raw.shape[axis]<3:continue
        step=max(1,int(np.ceil(raw.size/max_samples)))
        first=np.diff(raw,axis=axis).ravel()[::step]
        second=np.diff(raw,n=2,axis=axis).ravel()[::step]
        for values,normalizer in ((first,np.sqrt(2)),(second,np.sqrt(6))):
            med=np.median(values)
            estimates.append(float(np.median(abs(values-med))/.67448975/normalizer))
    if not estimates:raise ValueError('volume too small for voxel-noise estimation')
    # A gradient of a noiseless plane is not called random noise: remove the
    # median difference in each direction before its MAD is evaluated.
    dynamic=float(np.percentile(raw.ravel()[::max(1,raw.size//100000)],99)-
                  np.percentile(raw.ravel()[::max(1,raw.size//100000)],1))
    return max(float(np.median(estimates)),dynamic*1e-5,1e-9)


def _fit_batch(t,y,noise,radius,max_shift):
    n=len(y);dt=float(t[1]-t[0]);sigma_max=radius/3
    endpoint=(abs(t)>=radius*.75)
    left=t<=-radius*.75;right=t>=radius*.75
    air=np.median(y[:,left],axis=1);material=np.median(y[:,right],axis=1)
    slopes=[]
    for which in (left,right):
        te=t[which];a=y[:,which]
        slopes.append((a-a.mean(axis=1)[:,None])@(te-te.mean())/np.sum((te-te.mean())**2))
    b0=(slopes[0]+slopes[1])/2
    amp0=np.maximum(material-air-b0*(t[right].mean()-t[left].mean()),noise*.1)
    filtered=ndi.gaussian_filter1d(y,.65/dt,axis=1,mode='nearest')
    derivative=np.gradient(filtered,dt,axis=1)
    search=np.flatnonzero(abs(t)<=max_shift)
    rows=np.arange(n);k=search[np.argmax(derivative[:,search],axis=1)]
    peak=derivative[rows,k]
    p=np.column_stack([air-b0*t[left].mean(),b0,amp0,t[k],
                       np.clip(amp0/(np.sqrt(2*np.pi)*np.maximum(peak-b0,noise*.05)),.55,sigma_max*.9)])
    eye=np.eye(5)[None]
    failed=np.zeros(n,bool)
    for iteration in range(24):
        u=(t[None]-p[:,3,None])/p[:,4,None]
        cdf=ndtr(u);density=np.exp(-.5*u*u)/np.sqrt(2*np.pi)
        model=p[:,0,None]+p[:,1,None]*t+p[:,2,None]*cdf
        residual=y-model
        scale=np.maximum(np.median(abs(residual),axis=1)*1.4826,noise)
        weights=np.minimum(1,1.5*scale[:,None]/np.maximum(abs(residual),1e-12))
        jac=np.stack([np.ones_like(cdf),np.broadcast_to(t,cdf.shape),cdf,
                      -p[:,2,None]*density/p[:,4,None],
                      -p[:,2,None]*u*density/p[:,4,None]],axis=2)
        lhs=np.einsum('nki,nk,nkj->nij',jac,weights,jac)
        rhs=np.einsum('nki,nk,nk->ni',jac,weights,residual)
        diagonal=np.maximum(np.diagonal(lhs,axis1=1,axis2=2),1e-9)
        lhs+=eye*(diagonal*.001+1e-10)[:,None,:]
        try:delta=np.linalg.solve(lhs,rhs[...,None])[...,0]
        except np.linalg.LinAlgError:
            failed[:]=True;break
        delta[:,3]=np.clip(delta[:,3],-.6,.6)
        delta[:,4]=np.clip(delta[:,4],-.5,.5)
        p+=delta
        p[:,2]=np.clip(p[:,2],noise*.01,np.maximum(amp0*5,noise*2))
        p[:,3]=np.clip(p[:,3],-max_shift,max_shift)
        p[:,4]=np.clip(p[:,4],.4,sigma_max)
    if 'delta' in locals():
        failed|=(abs(delta[:,3])>.025)|(abs(delta[:,4])>.04)
    u=(t[None]-p[:,3,None])/p[:,4,None]
    cdf=ndtr(u);density=np.exp(-.5*u*u)/np.sqrt(2*np.pi)
    model=p[:,0,None]+p[:,1,None]*t+p[:,2,None]*cdf
    residual=y-model;rms=np.sqrt(np.mean(residual**2,axis=1))
    relative_rms=rms/np.maximum(p[:,2],noise*1e-5)
    jac=np.stack([np.ones_like(cdf),np.broadcast_to(t,cdf.shape),cdf,
                  -p[:,2,None]*density/p[:,4,None],
                  -p[:,2,None]*u*density/p[:,4,None]],axis=2)
    lhs=np.einsum('nki,nkj->nij',jac,jac)
    scale=np.sqrt(np.maximum(np.diagonal(lhs,axis1=1,axis2=2),1e-20))
    condition=np.linalg.cond(lhs/(scale[:,:,None]*scale[:,None,:]))
    try:
        covariance=np.linalg.inv(lhs+eye*1e-10)
        # Dense interpolation is not independent repeated measurement. Use
        # the volume-noise floor and ~one independent sample/reference voxel.
        covariance*=np.maximum(rms**2,noise**2)[:,None,None]/dt
        se=np.sqrt(np.maximum(np.diagonal(covariance,axis1=1,axis2=2),0))
    except np.linalg.LinAlgError:
        failed[:]=True;se=np.full((n,5),np.inf)
    flags=np.zeros(n,np.uint16)
    snr=p[:,2]/noise
    flags[(snr<12)|(material-air<=0)]|=2
    negative=np.max(-derivative[:,abs(t)<=radius-.75],axis=1)
    # Opposing structures above measured voxel-noise level are not bridged by
    # a large window. A small window may still measure an isolated edge.
    flags[negative>np.maximum(peak*.30,noise*3)]|=4
    # A broad erf must not bridge two distinct rising transitions. Require
    # a single significant derivative mode at measured voxel-noise scale.
    for row in range(n):
        height=max((peak[row]-p[row,1])*.30,noise[row]*3)
        modes,_=find_peaks(derivative[row],height=p[row,1]+height,
                           prominence=max(height*.8,noise[row]*3),
                           distance=max(1,int(np.ceil(2/dt))))
        modes=modes[abs(t[modes])<radius-.75]
        if len(modes)>1:flags[row]|=1024
    flags[(abs(p[:,3])>=max_shift-.035)|(p[:,4]<=.415)|(p[:,4]>=sigma_max-.04)]|=8
    flags[rms>np.maximum(p[:,2]*.065,noise*1.5)]|=16
    flags[(p[:,3]-3*p[:,4]<-radius+.75)|(p[:,3]+3*p[:,4]>radius-.75)]|=32
    # Both sides must have observed low-slope plateaus after the independently
    # fitted linear background is removed. Endpoint means alone do not suffice.
    allowed=np.maximum(p[:,2]*.035/np.maximum(p[:,4],1),noise*2/np.sqrt(max(radius/4,1)))
    flags[(abs(slopes[0]-p[:,1])>allowed)|(abs(slopes[1]-p[:,1])>allowed)]|=32
    flags[(abs(p[:,1])*2*radius>p[:,2]*.65)|
          (abs(p[:,1])*p[:,4]>p[:,2]*.12)]|=64
    flags[(condition>1e7)|(se[:,3]>.35)|(se[:,4]>.5*p[:,4])|
          (se[:,2]>.4*p[:,2])|~np.isfinite(se).all(axis=1)]|=128
    flags[failed|~np.isfinite(p).all(axis=1)]|=256
    return p,flags,{'rms':rms,'relative_rms':relative_rms,'snr':snr,
                    'position_se_model_only_vox':se[:,3],
                    'sigma_se_model_only_vox':se[:,4],'condition_scaled':condition}


def adaptive_refine(raw,pts,normals,noise_sigma=None,spacing_zyx=(1,1,1),batch_size=512):
    """Fit broad material-air profiles using the smallest supported window.

    raw is original float ZYX data, pts are voxel-centre ZYX coordinates and
    normals are finite nonzero physical ZYX vectors pointing dark->bright.
    ``noise_sigma`` is a supplied scalar measured on the original voxel grid;
    when omitted estimate_volume_noise is used. No extrapolation is allowed.

    ``all_flags`` (Nx3) and ``all_parameters`` (Nx3x5) preserve each radius's
    rejected candidates. Accepted coordinates are ``points_zyx``. Every
    rejected coordinate equals its input seed. ``sigma_vox`` and ``shift_vox``
    use min(spacing_zyx) as a length unit. Coverage and uncertainty diagnostics
    do not certify accuracy or identify the surface's physical meaning.
    """
    raw=np.asarray(raw,np.float32);pts=np.asarray(pts,float);normals=np.asarray(normals,float)
    spacing=np.asarray(spacing_zyx,float)
    if raw.ndim!=3 or pts.ndim!=2 or pts.shape[1]!=3 or normals.shape!=pts.shape:
        raise ValueError('raw must be ZYX, pts and normals matching Nx3')
    if spacing.shape!=(3,) or np.any(spacing<=0) or not np.isfinite(spacing).all():
        raise ValueError('spacing must be three positive finite numbers')
    if not np.isfinite(raw).all() or not np.isfinite(pts).all() or not np.isfinite(normals).all():
        raise ValueError('finite input required')
    lengths=np.linalg.norm(normals,axis=1)
    if np.any(lengths<1e-8):raise ValueError('nonzero physical normals required')
    normals=normals/lengths[:,None]
    direction=normals*spacing.min()/spacing
    noise=estimate_volume_noise(raw) if noise_sigma is None else float(noise_sigma)
    if not np.isfinite(noise) or noise<=0:raise ValueError('noise_sigma must be positive finite')
    n=len(pts);radii=np.array([6.,12.,24.])
    out={'seed_zyx':pts.copy(),'points_zyx':pts.copy(),'accepted':np.zeros(n,bool),
         'chosen_radius_vox':np.zeros(n),'sigma_vox':np.full(n,np.nan),
         'shift_vox':np.zeros(n),'parameters':np.full((n,5),np.nan),
         'all_parameters':np.full((n,3,5),np.nan),'all_flags':np.ones((n,3),np.uint16),
         'noise_sigma_volume':noise,'noise_source':'supplied_original_grid' if noise_sigma is not None else 'robust_original_voxel_differences',
         'radii_vox':radii,'flag_definitions':FLAGS.copy(),'parameter_names':PARAMETERS.copy(),
         'normal_physical_zyx':normals,'reference_spacing':float(spacing.min())}
    for field in ('rms','relative_rms','snr','position_se_model_only_vox','sigma_se_model_only_vox','condition_scaled'):
        out['all_'+field]=np.full((n,3),np.nan)
        out[field]=np.full(n,np.nan)
    for ri,radius in enumerate(radii):
        t=np.arange(-radius,radius+.01,.25)
        for first in range(0,n,batch_size):
            last=min(n,first+batch_size)
            coords=pts[first:last,None]+direction[first:last,None]*t[None,:,None]
            valid=(coords>=0).all(axis=(1,2))&(coords<=np.asarray(raw.shape)[None,None]-1).all(axis=(1,2))
            if not valid.any():continue
            chosen=np.flatnonzero(valid);indices=chosen+first
            profiles=ndi.map_coordinates(raw,coords[chosen].reshape(-1,3).T,
                                         order=1,mode='nearest').reshape(len(chosen),len(t))
            p,flags,metrics=_fit_batch(t,profiles,np.full(len(chosen),noise),radius,min(3,radius/4))
            out['all_parameters'][indices,ri]=p;out['all_flags'][indices,ri]=flags
            for key,values in metrics.items():out['all_'+key][indices,ri]=values
    supported=out['all_flags']==0
    # Multiple independently supported windows must agree on position. This
    # does not turn a rejected large window into a veto of an isolated small
    # edge merely because a second, distant interface exists.
    for i in range(n):
        good=np.flatnonzero(supported[i])
        if len(good)>1:
            shifts=out['all_parameters'][i,good,3]
            uncertainty=out['all_position_se_model_only_vox'][i,good]
            tolerance=max(.35,3*float(np.sqrt(np.max(uncertainty)**2+np.min(uncertainty)**2)))
            if shifts.max()-shifts.min()>tolerance:
                out['all_flags'][i,good]|=512;supported[i,good]=False
        good=np.flatnonzero(supported[i])
        if not len(good):continue
        ri=good[0];p=out['all_parameters'][i,ri]
        out['accepted'][i]=True;out['chosen_radius_vox'][i]=radii[ri]
        out['parameters'][i]=p;out['sigma_vox'][i]=p[4];out['shift_vox'][i]=p[3]
        out['points_zyx'][i]=pts[i]+direction[i]*p[3]
        for key in ('rms','relative_rms','snr','position_se_model_only_vox','sigma_se_model_only_vox','condition_scaled'):
            out[key][i]=out['all_'+key][i,ri]
    out['flags_any_window']=np.bitwise_or.reduce(out['all_flags'],axis=1)
    return out
