"""Research-only joint fit of two neighbouring CT interfaces.

A wall (+1) or gap (-1) is modelled as two opposite blurred steps with a
shared Gaussian width and a linear background. Fits are independent profiles;
this module does not modify meshes or connect surfaces automatically.

Coordinates, widths and sigma have the units of t (normally reference voxels).
Widths below the PSF resolution, unknown/asymmetric PSFs, beam hardening and
unmodelled neighbours can make the two positions unidentifiable. Acceptance is
only a model-quality diagnostic and does not certify dimensional accuracy.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.special import ndtr

PARAMETER_NAMES = ['background', 'background_slope', 'amplitude', 'center', 'width', 'sigma']
FLAGS = {1:'insufficient_signal_or_no_opposing_edges',
         2:'optimizer_failed_or_nonfinite',
         4:'parameter_at_bound',
         8:'large_residual',
         16:'ill_conditioned_or_unidentifiable_positions',
         32:'edge_outside_observed_profile_or_no_background_plateau',
         64:'large_background_slope'}


def model_double(t, p, polarity=1):
    a, b, amplitude, centre, width, sigma = np.asarray(p)
    left, right = centre-width/2, centre+width/2
    return a+b*t+polarity*amplitude*(ndtr((t-left)/sigma)-ndtr((t-right)/sigma))


def select_opposing_profiles(t, profiles, polarity=1, min_snr=6,
                             width_range=(.7,8), max_count=5000):
    """Return deterministic candidate indices from pre-sampled CT profiles.

    Detects an ordered pair of opposite derivative peaks, estimates endpoint
    noise and compares the interior extremum with a linear endpoint baseline.
    Supply profiles sampled along a physically meaningful line/normal; this
    filter does not discover a mesh or establish that the feature is a wall.
    """
    t=np.asarray(t,float); y=np.asarray(profiles,float)
    if y.ndim==1:y=y[None]
    if y.shape[1]!=len(t) or len(t)<17 or np.any(np.diff(t)<=0):
        raise ValueError('profiles must be NxM, t increasing with M>=17')
    dt=float(np.median(np.diff(t)))
    smooth=ndi.gaussian_filter1d(y,.45/dt,axis=1,mode='nearest')
    derivative=np.gradient(polarity*smooth,t,axis=1)
    endpoint=max(3,int(len(t)*.12))
    a=np.median(y[:,:endpoint],axis=1)
    z=np.median(y[:,-endpoint:],axis=1)
    baseline=a[:,None]+(z-a)[:,None]*(t-t[:endpoint].mean())/(t[-endpoint:].mean()-t[:endpoint].mean())
    detrend=polarity*(y-baseline)
    centre=np.argmax(ndi.gaussian_filter1d(detrend,.4/dt,axis=1),axis=1)
    noise=np.maximum(np.median(abs(np.concatenate([y[:,:endpoint]-baseline[:,:endpoint],
                                                  y[:,-endpoint:]-baseline[:,-endpoint:]],axis=1)),axis=1)*1.4826,
                     np.maximum(np.ptp(y,axis=1)*1e-4,1e-10))
    candidates=[]
    for i in range(len(y)):
        k=centre[i]
        if k<3 or k>len(t)-4:continue
        l=np.argmax(derivative[i,:k]);r=k+np.argmin(derivative[i,k:])
        width=t[r]-t[l]
        height=float(np.max(detrend[i]))
        if (width_range[0]<=width<=width_range[1] and height/noise[i]>=min_snr and
            derivative[i,l]>0 and derivative[i,r]<0):
            candidates.append(i)
    candidates=np.asarray(candidates,int)
    if len(candidates)>max_count:
        candidates=candidates[np.linspace(0,len(candidates)-1,max_count,dtype=int)]
    return candidates


def fit_double_profiles(t, profiles, polarity=1, max_profiles=5000,
                        width_bounds=(.7,8), sigma_bounds=(.35,2),
                        max_relative_rms=.08, max_boundary_se=.2, min_signal_snr=12):
    """Fit shared-PSF wall/gap profiles; return candidates and explicit rejects.

    Fits all supplied rows (N<=max_profiles). ``parameters`` are in
    PARAMETER_NAMES order. ``boundaries`` holds mu_left, mu_right and contains
    candidate values even when rejected: always check ``accepted`` first.
    ``boundary_standard_errors_model_only`` assumes independent residuals,
    correct PSF and the fitted model. Correlated CT noise/model error/calibration
    are outside that diagnostic. ``flags`` is a bit mask documented in FLAGS.
    """
    if polarity not in (-1,1):raise ValueError('polarity must be +1 wall or -1 gap')
    t=np.asarray(t,float); profiles=np.asarray(profiles,float)
    if profiles.ndim==1:profiles=profiles[None]
    if profiles.ndim!=2 or profiles.shape[1]!=len(t) or len(t)<17:
        raise ValueError('profiles must be NxM matching t with M>=17')
    if len(profiles)>max_profiles:
        raise ValueError(f'N={len(profiles)} exceeds max_profiles={max_profiles}; select profiles first')
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(profiles)) or np.any(np.diff(t)<=0):
        raise ValueError('finite profiles and strictly increasing t required')
    if width_bounds[0]<=0 or width_bounds[1]<=width_bounds[0] or sigma_bounds[0]<=0:
        raise ValueError('invalid width or sigma bounds')
    n=len(profiles);m=len(t)
    out={'parameters':np.full((n,6),np.nan),'boundaries':np.full((n,2),np.nan),
         'accepted':np.zeros(n,bool),'flags':np.zeros(n,np.uint16),
         'rms':np.full(n,np.nan),'relative_rms':np.full(n,np.nan),
         'snr':np.full(n,np.nan),'jacobian_condition_scaled':np.full(n,np.nan),
         'boundary_standard_errors_model_only':np.full((n,2),np.nan),
         'width_standard_error_model_only':np.full(n,np.nan),
         'noise':np.full(n,np.nan),'parameter_names':PARAMETER_NAMES.copy(),
         'flag_definitions':FLAGS.copy(),'polarity':int(polarity)}
    endpoint=max(4,int(m*.12));dt=float(np.median(np.diff(t)))
    eligible=set(select_opposing_profiles(t,profiles,polarity,min_snr=6,
                                           width_range=(width_bounds[0],width_bounds[1]+1.5),
                                           max_count=max_profiles))
    for i,y in enumerate(profiles):
        flags=0
        if i not in eligible:flags|=1
        a0=float(np.median(y[:endpoint]));ar=float(np.median(y[-endpoint:]))
        b0=(ar-a0)/(t[-endpoint:].mean()-t[:endpoint].mean())
        a0-=b0*t[:endpoint].mean()
        baseline=a0+b0*t
        detrend=polarity*(y-baseline)
        scale=max(float(np.percentile(detrend,95)-np.percentile(detrend,5)),
                  float(np.ptp(y))*1e-3,1e-9)
        noise=max(float(np.median(abs(np.r_[y[:endpoint]-baseline[:endpoint],
                                               y[-endpoint:]-baseline[-endpoint:]]))*1.4826),scale*1e-4)
        out['noise'][i]=noise
        out['snr'][i]=float(np.max(detrend))/noise
        if np.ptp(y)<1e-9:
            out['flags'][i]=flags|1;continue
        smooth=ndi.gaussian_filter1d(detrend,.5/dt,mode='nearest')
        peak=int(np.argmax(smooth));peak=max(2,min(m-3,peak))
        deriv=np.gradient(smooth,t)
        left=int(np.argmax(deriv[:peak]));right=peak+int(np.argmin(deriv[peak:]))
        w0=float(np.clip(t[right]-t[left],width_bounds[0]+.05,width_bounds[1]-.05))
        c0=float(np.clip((t[left]+t[right])/2,t[0]+.25,t[-1]-.25))
        # Normalize intensities only, preserving geometric t units. This
        # keeps conditioning diagnostics independent of uint16 versus [0,1].
        yn=(y-a0)/scale
        p0=[0,b0/scale,max(float(np.max(smooth))/scale,1e-3),c0,w0,1.0]
        lower=np.array([-3,-2,1e-4,t[0]+.1,width_bounds[0],sigma_bounds[0]])
        upper=np.array([3,2,8,t[-1]-.1,width_bounds[1],sigma_bounds[1]])
        p0=np.clip(p0,lower+1e-5,upper-1e-5)
        try:
            fit=least_squares(lambda p:model_double(t,p,polarity)-yn,p0,
                               bounds=(lower,upper),loss='soft_l1',
                               f_scale=max(noise/scale, .004),
                               x_scale='jac',max_nfev=220,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError):
            out['flags'][i]=flags|2;continue
        p=fit.x
        residual=model_double(t,p,polarity)-yn
        rms=float(np.sqrt(np.mean(residual**2)))
        out['rms'][i]=rms*scale
        effective_noise=max(noise,rms*scale)
        out['snr'][i]=float(np.max(detrend))/effective_noise
        if out['snr'][i]<min_signal_snr:flags|=1
        out['relative_rms'][i]=rms/max(p[2],1e-9)
        original=p.copy();original[0]=a0+scale*p[0];original[1:3]*=scale
        out['parameters'][i]=original
        out['boundaries'][i]=[p[3]-p[4]/2,p[3]+p[4]/2]
        if not fit.success or not np.all(np.isfinite(p)):flags|=2
        # Flag geometric bounds and amplitude ceiling. The background's
        # intentionally generous numerical bounds are not physical evidence.
        if (p[4]<=lower[4]+.025 or p[4]>=upper[4]-.025 or
            p[5]<=lower[5]+.015 or p[5]>=upper[5]-.015 or p[2]>=upper[2]-.02):flags|=4
        if out['relative_rms'][i]>max_relative_rms:flags|=8
        # Compute the unweighted model Jacobian, not robust-loss transformed
        # optimizer Jacobian, for transparent covariance diagnostics.
        eps=1e-5
        jac=np.stack([(model_double(t,p+np.eye(6)[j]*eps,polarity)-
                       model_double(t,p-np.eye(6)[j]*eps,polarity))/(2*eps)
                      for j in range(6)],axis=1)
        scales=np.maximum(np.linalg.norm(jac,axis=0),1e-12)
        cond=float(np.linalg.cond(jac/scales))
        out['jacobian_condition_scaled'][i]=cond
        try:
            covariance=np.linalg.inv(jac.T@jac)*max(float(residual@residual)/(m-6),
                                                   (noise/scale)**2)
            transforms=np.array([[0,0,0,1,-.5,0],[0,0,0,1,.5,0]])
            variance=np.einsum('ij,jk,ik->i',transforms,covariance,transforms)
            se=np.sqrt(np.maximum(variance,0))
            width_se=np.sqrt(max(covariance[4,4],0))
            out['boundary_standard_errors_model_only'][i]=se
            out['width_standard_error_model_only'][i]=width_se
            if (not np.isfinite(se).all() or cond>1e5 or
                se.max()>max_boundary_se or width_se>max(.3,p[4]*.25)):
                flags|=16
        except np.linalg.LinAlgError:flags|=16
        left,right=out['boundaries'][i]
        if left-2*p[5]<t[0]+dt or right+2*p[5]>t[-1]-dt:flags|=32
        if abs(p[1])*(t[-1]-t[0])>p[2]*.8:flags|=64
        out['flags'][i]=flags
        out['accepted'][i]=(flags==0)
    return out
