"""Local region fitting and PSF-aware surface localization for CT volumes.

An experimental 3D extension/adaptation of local binary fitting with a convex
TV inner solve; this is not a reproduction of the original level-set solver.
All coordinates are voxel coordinates, without a claim of calibrated accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
from scipy.special import ndtr
from skimage import filters, measure
from boundary_study import load, plt, contours


def farid_gradients(image):
    """Separable 7-tap Farid--Simoncelli first derivatives in all three axes.

    Coefficients: Farid & Simoncelli, IEEE TIP 13(4), 2004; also documented
    at raymondngiam/subpixel-edge-contour-in-opencv. Correlation preserves the
    sign of a positive intensity ramp; smoothing and ramp gains are normalized.
    """
    p=np.array([.004711,.069321,.245410,.361117,.245410,.069321,.004711])
    d=np.array([-.018708,-.125376,-.193091,0,.193091,.125376,.018708])
    p/=p.sum();d/=np.dot(d,np.arange(-3,4))
    def separable(kernels):
        value=image
        for axis,kernel in enumerate(kernels):
            value=ndi.correlate1d(value,kernel,axis=axis,mode='nearest')
        return value
    smooth=separable([p,p,p])
    gradients=[separable([d if axis==j else p for j in range(3)]) for axis in range(3)]
    return smooth,gradients,np.sqrt(sum(g*g for g in gradients))


def local_fit(image, initial, scale=8., rounds=6, tv=.12, inner=25):
    """Region-scalable squared-error energy with normalized local contrast.

    The label field can change topology everywhere. A TV regularizer couples
    all three spatial axes; no 2D hole-filling or fixed cavity count is used.
    Gaussian kernels are separable and use the real available Z context.
    """
    image = ndi.gaussian_filter(image, .65)
    u = initial.astype(np.float32)
    history = []
    sigma = (min(scale, 3.), scale, scale)
    blur = lambda a: ndi.gaussian_filter(a, sigma, mode='reflect')
    for iteration in range(rounds):
        density = blur(u)
        m1 = blur(image*u) / np.maximum(density, .005)
        m0 = blur(image*(1-u)) / np.maximum(1-density, .005)
        # Absent local classes borrow their previous global class mean.
        global1 = float((image*u).sum()/max(u.sum(), 1))
        global0 = float((image*(1-u)).sum()/max((1-u).sum(), 1))
        m1 = np.where(density > .005, m1, global1)
        m0 = np.where(density < .995, m0, global0)
        # Material is brighter than air. Do not invert the physical labels.
        midpoint = (m1+m0)/2
        contrast = np.maximum(m1-m0, .035)
        m1, m0 = midpoint+contrast/2, midpoint-contrast/2
        e1 = -2*image*blur(m1)+blur(m1*m1)
        e0 = -2*image*blur(m0)+blur(m0*m0)
        data = np.clip((e1-e0)/(blur(contrast*contrast)+.0025), -3, 3)
        previous = u.copy()
        dual = np.zeros((3, *u.shape), np.float32)
        extrapolated = u.copy()
        for _ in range(inner):
            for axis in range(3):
                a, b = [slice(None)]*3, [slice(None)]*3
                a[axis], b[axis] = slice(None,-1), slice(1,None)
                a,b = tuple(a),tuple(b)
                dual[axis][a] += .25*(extrapolated[b]-extrapolated[a])
                end=[slice(None)]*3;end[axis]=-1;dual[axis][tuple(end)]=0
            dual /= np.maximum(1, np.sqrt(np.sum(dual*dual, axis=0))/tv)
            divergence = np.zeros_like(u)
            for axis in range(3):
                a,b=[slice(None)]*3,[slice(None)]*3
                a[axis],b[axis]=slice(None,-1),slice(1,None);a,b=tuple(a),tuple(b)
                divergence[a] += dual[axis][a]
                divergence[b] -= dual[axis][a]
            new = np.clip(u+.25*divergence-.25*data, 0, 1)
            extrapolated = 2*new-u
            u = new
        history.append({'round':iteration+1,'mean_change':float(abs(u-previous).mean()),
                        'material_voxels':int((u>.5).sum())})
        print('  local fit',history[-1],flush=True)
    return u>.5, u, history


def profile_model(theta, t, slab=False):
    """Blurred step/slab plus linear background, with analytic Jacobian."""
    a,b,c,d,logs = theta[:,:5].T
    sigma=np.exp(logs)[:,None]
    q=(t[None,:]-d[:,None])/sigma
    phi=np.exp(-q*q/2)/np.sqrt(2*np.pi)
    f=ndtr(q)
    dd=-c[:,None]*phi/sigma
    ds=-c[:,None]*q*phi
    if slab:
        v=(t[None,:]-d[:,None]-theta[:,5,None])/sigma
        phiv=np.exp(-v*v/2)/np.sqrt(2*np.pi)
        f-=ndtr(v)
        dd+=c[:,None]*phiv/sigma
        ds+=c[:,None]*v*phiv
        dw=c[:,None]*phiv/sigma
    value=a[:,None]+b[:,None]*(t[None,:]/6)+c[:,None]*f
    columns=[np.ones_like(f),np.broadcast_to(t[None,:]/6,f.shape),f,dd,ds]
    if slab:columns.append(dw)
    return value,np.stack(columns,axis=2)


def fit_profiles(profiles, t, width=None, iterations=18, centered=False):
    """Batched damped least squares; residual and conditioning remain visible."""
    slab=width is not None
    n=len(profiles);p=6 if slab else 5
    theta=np.zeros((n,p),np.float64)
    theta[:,0]=np.median(profiles[:,t < -3],axis=1)
    theta[:,2]=np.maximum(np.percentile(profiles,90,axis=1)-theta[:,0],.03)
    theta[:,4]=np.log(1.)
    if slab:
        theta[:,5]=np.clip(width,.8,10)
        if centered:theta[:,3]=-theta[:,5]/2
    lower=np.array([-.1,-.25,.015,-6 if centered else -2.5,np.log(.4)]+([.7] if slab else []))
    upper=np.array([1.2,.25,1.3,2.5,np.log(2.5)]+([12] if slab else []))
    damping=np.full(n,1e-3)
    for _ in range(iterations):
        predicted,j=profile_model(theta,t,slab)
        residual=profiles-predicted
        # Huber influence limits isolated reconstruction streaks.
        robust=np.minimum(1,.035/np.maximum(abs(residual),1e-8))
        h=np.einsum('nki,nk,nkj->nij',j,robust,j)
        g=np.einsum('nki,nk,nk->ni',j,robust,residual)
        system=h+np.eye(p)[None]*damping[:,None,None]
        delta=np.linalg.solve(system,g[...,None])[...,0]
        proposal=np.clip(theta+delta,lower,upper)
        proposed,_=profile_model(proposal,t,slab)
        loss=lambda r:np.sum(np.where(abs(r)<.035,.5*r*r,.035*(abs(r)-.0175)),axis=1)
        accepted=loss(profiles-proposed)<loss(residual)
        theta[accepted]=proposal[accepted]
        damping=np.clip(damping*np.where(accepted,.5,4),1e-7,1e4)
    predicted,j=profile_model(theta,t,slab)
    residual=profiles-predicted
    rmse=np.sqrt(np.mean(residual**2,axis=1))
    h=np.einsum('nki,nkj->nij',j,j)+np.eye(p)[None]*1e-8
    covariance=np.linalg.inv(h)*np.sum(residual**2,axis=1)[:,None,None]/max(1,len(t)-p)
    return theta,rmse,np.sqrt(np.maximum(covariance[:,3,3],0))


def recover_thin_walls(raw, baseline, max_distance=12.):
    """Hessian sheet normals + two-interface PSF fit propose missing walls.

    This is a custom adaptation of second-derivative ridge localization, not
    an exact Steger implementation. Candidate material must have measured
    contrast, an identifiable pair of interfaces and a low profile residual.
    No closing operation, filled cavity or guessed wall thickness is used.
    """
    smooth=ndi.gaussian_filter(raw,.65)
    hi=ndi.maximum_filter(smooth,size=(3,9,9));lo=ndi.minimum_filter(smooth,size=(3,9,9))
    potential=(smooth>(hi+lo)/2)&(hi-lo>.035)&~baseline
    potential &= ndi.distance_transform_edt(~baseline)<=max_distance
    points=np.argwhere(potential).astype(np.float32)
    hessian=np.empty((len(points),3,3),np.float32)
    for a in range(3):
        for b in range(a,3):
            order=[0,0,0];order[a]+=1;order[b]+=1
            field=ndi.gaussian_filter(raw,1.,order=order)
            value=field[tuple(points.astype(np.int32).T)]
            hessian[:,a,b]=value;hessian[:,b,a]=value
    values,vectors=np.linalg.eigh(hessian)
    normals=vectors[:,:,0]
    sheet=(values[:,0]<-.006)&(abs(values[:,1])<.6*abs(values[:,0]))
    points,normals=points[sheet],normals[sheet]
    support=np.zeros(raw.shape,bool);quality=[]
    t=np.arange(-6,8.001,.35,dtype=np.float32)
    for first in range(0,len(points),1536):
        p=points[first:first+1536];n=normals[first:first+1536]
        coords=p[:,None,:]+t[None,:,None]*n[:,None,:]
        profiles=ndi.map_coordinates(raw,coords.reshape(-1,3).T,order=1,
                                     mode='nearest').reshape(len(p),-1)
        fit,rmse,se=fit_profiles(profiles,t,np.full(len(p),3.),iterations=24,centered=True)
        sigma=np.exp(fit[:,4]);width=fit[:,5];left=fit[:,3];right=left+width
        good=(fit[:,2]>.06)&(rmse<.018)&(rmse<.08*fit[:,2])&(se<.2)
        good &= (width>2*sigma)&(width<7)&(left<0)&(right>0)&(sigma>.42)&(sigma<2.)
        good &= np.all((coords>=0)&(coords<=np.array(raw.shape)[None,None,:]-1),axis=(1,2))
        quality.append({'candidates':len(p),'accepted':int(good.sum())})
        # Rasterize the measured material interval, excluding edge mixed voxels.
        for fraction in np.linspace(.12,.88,9):
            targets=p[good]+n[good]*(left[good]+fraction*width[good])[:,None]
            integer=np.rint(targets).astype(np.int32)
            support[tuple(integer.T)]=True
    return baseline|support, support, {'potential_voxels':int(potential.sum()),
        'sheet_profile_candidates':len(points),'accepted_profiles':sum(x['accepted'] for x in quality),
        'added_voxels':int((support&~baseline).sum()),'max_distance_vox':max_distance}


def fit_surface(raw, mask, core, method='psf'):
    """Use mask geometry for inward normals, fit the *original* CT intensities.

    Slab profiles handle a second, opposing interface within the fit window.
    Raw residual and single-profile standard error are quality diagnostics,
    not a full CT measurement uncertainty. Rejected vertices remain unchanged.
    """
    from canny3d_study import refine_mesh, gradients, canny_variants
    if method in ['gradient','farid']:
        if method=='farid':smooth,g,mag=farid_gradients(raw)
        else:smooth,g,mag,_=gradients(raw)
        roi=ndi.binary_dilation(mask,iterations=6)
        edge=canny_variants(raw,smooth,g,mag,roi)
        return refine_mesh(mask,smooth,g,mag,edge['coherence'],core,edge['adaptive'])
    vertices,faces,_,_=measure.marching_cubes(mask[core].astype(np.float32),.5,
                                           allow_degenerate=False)
    points=vertices.copy();points[:,0]+=core.start
    signed=ndi.distance_transform_edt(mask)-ndi.distance_transform_edt(~mask)
    normals=[]
    for axis in range(3):
        order=[0,0,0];order[axis]=1
        g=ndi.gaussian_filter(signed,1.1,order=order)
        normals.append(ndi.map_coordinates(g,points.T,order=1,mode='nearest'))
    normal=np.stack(normals,axis=1)
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-8)
    t=np.arange(-6,8.001,.35,dtype=np.float32)
    moved=points.copy();accepted=np.zeros(len(points),bool)
    shifts=np.zeros(len(points),np.float32);errors=np.zeros(len(points),np.float32)
    stderrs=np.zeros(len(points),np.float32);sigmas=np.zeros(len(points),np.float32)
    paired=np.zeros(len(points),bool)
    clipped=np.zeros(len(points),bool)
    binary=mask.astype(np.uint8)
    for start in range(0,len(points),1536):
        stop=min(start+1536,len(points));p=points[start:stop];n=normal[start:stop]
        coords=p[:,None,:]+n[:,None,:]*t[None,:,None]
        profiles=ndi.map_coordinates(raw,coords.reshape(-1,3).T,order=1,
                                     mode='nearest').reshape(len(p),-1)
        sampled=ndi.map_coordinates(binary,coords.reshape(-1,3).T,
                                     order=0,mode='nearest').reshape(len(p),-1)
        search=np.where(t>1)[0]
        falling=(sampled[:,search]==0)
        haspair=falling.any(axis=1)
        widths=t[search[np.argmax(falling,axis=1)]]
        haspair &= widths<6.5
        theta,rmse,se=fit_profiles(profiles,t)
        if haspair.any():
            pair,prmse,pse=fit_profiles(profiles[haspair],t,widths[haspair])
            ids=np.flatnonzero(haspair)
            # A second interface must explain the measured data better,
            # and contain an actual resolvable material interval.
            choose=(prmse<rmse[ids]*.9)&(pair[:,5]>1)&(pair[:,5]>2*np.exp(pair[:,4]))&(pse<.15)
            ids=ids[choose];theta[ids]=pair[choose,:5];rmse[ids]=prmse[choose];se[ids]=pse[choose]
            paired[start+ids]=True
        shift=theta[:,3]
        sigma=np.exp(theta[:,4])
        good=(theta[:,2]>.045)&(rmse<.045)&(rmse<theta[:,2]*.12)&(se<.35)
        good &= (abs(shift)<2.4)&(sigma>.42)&(sigma<2.4)
        good &= np.all((coords>=0)&(coords<=np.array(raw.shape)[None,None,:]-1),axis=(1,2))
        cut=(vertices[start:stop,0]<.01)|(vertices[start:stop,0]>mask[core].shape[0]-1.01)
        good &= ~cut
        accepted[start:stop]=good;shifts[start:stop]=np.where(good,shift,0)
        errors[start:stop]=rmse;stderrs[start:stop]=se;sigmas[start:stop]=sigma
        moved[start:stop]=p+n*np.where(good,shift,0)[:,None]
    cross=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
    for _ in range(40):
        updated=np.cross(moved[faces[:,1]]-moved[faces[:,0]],moved[faces[:,2]]-moved[faces[:,0]])
        bad=(np.sum(cross*updated,axis=1)<=0)|(np.linalg.norm(updated,axis=1)<1e-7)
        if not bad.any():break
        ids=np.unique(faces[bad]);clipped[ids]=True
        moved[ids]=(moved[ids]+points[ids])/2;shifts[ids]*=.5;accepted[ids]=False
    if bad.any():raise ValueError('PSF fit left inverted triangles')
    moved[:,0]-=core.start
    info={'accepted_vertex_fraction':float(accepted.mean()),'paired_profiles':int(paired.sum()),
          'geometry_backtracked_vertices':int(clipped.sum()),
          'abs_shift_p95_vox':float(np.percentile(abs(shifts),95)),
          'residual_rmse_median':float(np.median(errors)),
          'quality_warning':'Fit stderr excludes CT systematic errors and is not metrological uncertainty.'}
    return vertices,moved,faces,accepted,shifts,info,{
        'fit_rmse':errors,'fit_stderr_vox':stderrs,'fit_sigma_vox':sigmas,
        'paired_profile':paired,'geometry_backtracked':clipped}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('CT_Test'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--blocks',default='738:762')
    parser.add_argument('--scales',default='6,12')
    parser.add_argument('--rounds',type=int,default=6)
    parser.add_argument('--mesh',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    for block in args.blocks.split(','):
        first,last=map(int,block.split(':'));name=f'block_{first:04d}_{last:04d}'
        reference=Path('results/canny3d_v2/supported_refinement')/name
        meta=json.loads((reference/'metrics.json').read_text())
        y0,y1,x0,x1=meta['crop_box_y0_y1_x0_x1']
        cf,cl=max(500,first-8),min(1000,last+8)
        raw=np.stack([load(args.input,i)[y0:y1,x0:x1] for i in range(cf,cl+1)])
        core=slice(first-cf,last-cf+1)
        baseline=np.load(reference/'initializer_mask.npz')['mask']
        start=np.pad(baseline,((core.start,len(raw)-core.stop),(0,0),(0,0)),mode='edge')
        # Initial labels in the context come from real intensities; the saved
        # baseline exists only for the exported core. No repeated TIFF input.
        threshold=float(filters.threshold_otsu(raw))
        start[:core.start]=raw[:core.start]>threshold
        start[core.stop:]=raw[core.stop:]>threshold
        out=args.output/name;out.mkdir()
        indices=sorted({first,(first+last)//2,last})
        candidates={};records=[]
        for scale in map(float,args.scales.split(',')):
            began=perf_counter();label=f'local_rsf_{scale:g}'
            print(name,label,'start',raw.shape,flush=True)
            mask,prob,history=local_fit(raw,start,scale=scale,rounds=args.rounds)
            np.savez_compressed(out/f'{label}.npz',mask=mask[core],probability=prob[core],
                                crop_origin_yx=[y0,x0],source_slices=np.arange(first,last+1))
            candidates[label]=mask
            record={'method':label,'history':history,'seconds':perf_counter()-began}
            records.append(record);print(record,flush=True)
        for index in indices:
            rz=index-cf;bz=index-first
            fig,axes=plt.subplots(1,len(candidates)+2,figsize=(4*(len(candidates)+2),12))
            for ax,title in zip(axes,['Original CT','Previous watershed',*candidates]):
                ax.imshow(raw[rz],cmap='gray',vmin=.06,vmax=.8);ax.set_title(title);ax.axis('off')
            for ax,mask in zip(axes[1:],[start,*candidates.values()]):
                for line in contours(mask[rz]):ax.plot(line[:,1],line[:,0],c='#00ff70',lw=.65)
            fig.tight_layout();fig.savefig(out/f'comparison_{index:04d}.png',dpi=145);plt.close(fig)
        metadata={'first':first,'last':last,'context_first':cf,'context_last':cl,
                  'crop_box_y0_y1_x0_x1':[y0,y1,x0,x1],'methods':records}
        (out/'metrics.json').write_text(json.dumps(metadata,indent=2)+'\n')
        if args.mesh:
            from canny3d_study import save_ply,plane_segments
            from matplotlib.collections import LineCollection
            for name,mask in candidates.items():
                print(name,'PSF fit',flush=True)
                base,moved,faces,accepted,shift,info,quality=fit_surface(raw,mask,core)
                np.savez_compressed(out/f'{name}_surface.npz',seed_zyx=base,refined_zyx=moved,
                    faces_zyx=faces,accepted=accepted,normal_shift_vox=shift,**quality)
                save_ply(out/f'{name}_surface.ply',moved,faces,(y0,x0))
                (out/f'{name}_fit_metrics.json').write_text(json.dumps(info,indent=2)+'\n')
                for index in indices:
                    fig,ax=plt.subplots(figsize=(7,12));ax.imshow(raw[index-cf],cmap='gray',vmin=.06,vmax=.8)
                    lines=plane_segments(moved,faces,index-first)
                    ax.add_collection(LineCollection(lines[:,:,[1,0]],colors='#00ff70',linewidths=.65))
                    ax.set_title(f'{name} + raw CT PSF fit: {index:04d}');ax.axis('off')
                    fig.tight_layout();fig.savefig(out/f'{name}_psf_{index:04d}.png',dpi=145);plt.close(fig)


if __name__=='__main__':main()
