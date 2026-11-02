"""Targeted experiment: diffuse right external CT boundary, isotropic grid.

Coarse full-image Otsu guides a broad search band only. No cropped-Otsu ROI
and no fixed gradient/contrast cutoffs are used to accept measurements.
Detected paths are image-supported hypotheses; raw fits carry rejection flags.
"""
import os,sys,json,argparse,time,csv,tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'ct-surface-mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'ct-surface-cache'))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters,restoration,exposure
from edge_seed_study import block
import boundary_study as old
from adaptive_profile import adaptive_refine,FLAGS


def coarse_right(source,ids,box):
    curves=[];x0=box[2]
    for i in ids:
        full=old.load(source,i);sm=ndi.gaussian_filter(full,1)
        threshold=filters.threshold_otsu(sm)
        binary=old.largest(sm>threshold)[box[0]:box[1],box[2]:box[3]]
        right=np.where(binary.any(axis=1),binary.shape[1]-1-np.argmax(binary[:,::-1],axis=1),np.nan)
        good=np.flatnonzero(np.isfinite(right))
        right=np.interp(np.arange(len(right)),good,right[good])
        curves.append(right)
    return ndi.gaussian_filter(np.array(curves),(2,4))


def candidate_path(field,coarse,z,box):
    rows=np.arange(max(32,600-box[0]),min(field.shape[1]-32,1250-box[0]),2)
    offsets=np.arange(-40,41)
    x=coarse[z,rows,None]+offsets
    x=np.clip(x,1,field.shape[2]-2)
    coords=np.stack([np.full_like(x,z),np.broadcast_to(rows[:,None],x.shape),x])
    # Coarse surface directions keep high-frequency grain from determining
    # the normal. They do not determine the accepted position.
    dz,dy=np.gradient(coarse)
    normal=np.stack([dz[z,rows],dy[z,rows],-np.ones(len(rows))],axis=1)
    normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    responses=[]
    for scale in [1.,2.,4.,8.]:
        dot=np.zeros_like(x)
        for ax in range(3):
            order=[0,0,0];order[ax]=1
            derivative=ndi.gaussian_filter(field,(1,scale,scale),order=order)
            dot+=ndi.map_coordinates(derivative,coords.reshape(3,-1),order=1).reshape(x.shape)*normal[:,ax,None]
        responses.append(np.maximum(dot,0)*np.sqrt(scale))
    response=np.max(responses,axis=0)
    scale=np.array([1.,2.,4.,8.])[np.argmax(responses,axis=0)]
    # Normalize within each row to permit locally weak interfaces; this is
    # proposal generation, never an acceptance/accuracy test.
    cost=-response/np.maximum(np.percentile(response,90,axis=1,keepdims=True),1e-5)
    cost+=.002*(offsets[None]/10)**2
    cumulative=cost[0].copy();back=[]
    for j in range(1,len(rows)):
        delta=x[j,None,:]-x[j-1,:,None]
        transitions=cumulative[:,None]+.08*np.minimum(delta**2,100)
        k=np.argmin(transitions,axis=0);back.append(k)
        cumulative=cost[j]+transitions[k,np.arange(len(offsets))]
    choice=np.empty(len(rows),int);choice[-1]=np.argmin(cumulative)
    for j in range(len(rows)-2,-1,-1):choice[j]=back[j][choice[j+1]]
    xx=x[np.arange(len(rows)),choice]
    pts=np.stack([np.full(len(rows),z),rows,xx],axis=1)
    return pts,normal,scale[np.arange(len(rows)),choice],(choice==0)|(choice==len(offsets)-1),{
        'search_source_y':[int(rows[0]+box[0]),int(rows[-1]+box[0])],
        'search_offset_from_coarse_x':[-40,40],
        'path_at_band_limit_count':int(np.count_nonzero((choice==0)|(choice==len(offsets)-1)))}


def main(a):
    a.out.mkdir(parents=True,exist_ok=True);records=[]
    for center in map(int,a.centers.split(',')):
        start=time.monotonic();raw,ids,box=block(a.source,center,8);z=ids.index(center)
        coarse=coarse_right(a.source,ids,box)
        available=sorted(int(p.stem.split('_')[-1]) for p in a.source.glob('img_*.tif'))
        fit_ids=list(range(max(min(available),center-28),min(max(available),center+28)+1))
        fit_raw=np.stack([old.load(a.source,i)[box[0]:box[1],box[2]:box[3]] for i in fit_ids])
        # Noise from original voxel-neighbour differences, avoiding artificial
        # precision from quarter-voxel interpolation samples.
        noise=float(np.median(np.abs(np.diff(raw[::3],axis=2)))/.67448975/np.sqrt(2))
        fields=[('multiscale',raw)]
        if not a.skip_nlm:
            den=restoration.denoise_nl_means(raw,h=.9*noise,sigma=noise,
                patch_size=3,patch_distance=3,fast_mode=True,preserve_range=True,channel_axis=None)
            fields.append(('nlm',den.astype(np.float32)))
            corrected=np.stack([exposure.equalize_adapthist(np.clip(im,0,1),kernel_size=64,clip_limit=.01) for im in den])
            fields.append(('nlm_clahe',corrected.astype(np.float32)))
        proposals={};record={'slice':center,'source_slices':ids,'fit_source_slices':fit_ids,'box':box,'noise_sigma_raw_for_nlm':noise,'variants':{}}
        for name,field in fields:
            pts,normals,scales,at_limit,meta=candidate_path(field,coarse,z,box)
            fit_pts=pts.copy();delta=ids[0]-fit_ids[0];fit_pts[:,0]+=delta
            fit=adaptive_refine(fit_raw,fit_pts,normals)
            fit['seed_zyx'][:,0]-=delta;fit['points_zyx'][:,0]-=delta
            fit['model_accepted']=fit['accepted'].copy()
            fit['at_search_limit']=at_limit
            fit['flag_definitions'][2048]='proposal_on_search_limit'
            fit['all_flags'][at_limit]|=2048
            fit['flags_any_window']=np.bitwise_or.reduce(fit['all_flags'],axis=1)
            fit['accepted'][at_limit]=False
            fit['points_zyx'][at_limit]=pts[at_limit]
            fit['shift_vox'][at_limit]=0
            fit['chosen_radius_vox'][at_limit]=0
            fit['sigma_vox'][at_limit]=np.nan
            fit['parameters'][at_limit]=np.nan
            np.savez_compressed(a.out/f'proposals_{name}_{center:04d}.npz',points=pts,normals=normals,scales=scales,source_slices=ids,box=box,noise_sigma=noise,at_search_limit=at_limit)
            np.savez_compressed(a.out/f'fits_{name}_{center:04d}.npz',**{k:v for k,v in fit.items() if isinstance(v,np.ndarray)})
            proposals[name]=(pts,normals,fit)
            acc=fit['accepted'];region=(pts[:,1]+box[0]>=750)&(pts[:,1]+box[0]<950)
            record['variants'][name]={'candidates':len(pts),'raw_fit_accepted':int(acc.sum()),
                'diagnostic_rows_750_949':int(region.sum()),'accepted_rows_750_949':int((acc&region).sum()),
                'accepted_fraction':float(acc.mean()),'raw_fit_noise_sigma':fit['noise_sigma_volume'],
                'window_counts':{str(int(r)):int(np.count_nonzero(acc&(fit['chosen_radius_vox']==r))) for r in [6,12,24]},
                'rejection_any_window':{s:int(np.count_nonzero(np.any((fit['all_flags']&bit)!=0,axis=1))) for bit,s in {**FLAGS,2048:'proposal_on_search_limit'}.items()},**meta}
            with (a.out/f'points_{name}_{center:04d}.csv').open('w',newline='') as f:
                w=csv.writer(f,lineterminator='\n');w.writerow(['source_slice','seed_y','seed_x','fit_source_z','fit_y','fit_x','accepted','at_search_limit','chosen_radius_vox','sigma_vox','flags_R6','flags_R12','flags_R24'])
                for k in range(len(pts)):
                    q=fit['points_zyx'][k]+[ids[0],box[0],box[2]]
                    w.writerow([center,pts[k,1]+box[0],pts[k,2]+box[2],*q,int(acc[k]),int(at_limit[k]),fit['chosen_radius_vox'][k],fit['sigma_vox'][k],*fit['all_flags'][k]])
            print(center,name,'raw_fit accepted',int(acc.sum()),'of',len(pts),flush=True)
        if len(proposals)>1:
            base=proposals['multiscale'][0]
            for name,(pts,_,_) in proposals.items():
                d=np.linalg.norm(pts-base,axis=1)
                record['variants'][name]['proposal_change_from_raw_p95_vox']=float(np.percentile(d,95))
        if a.previous_profiles:
            d=np.load(a.previous_profiles/f'profiles_{center:04d}.npz');oldbox=d['box_y0_y1_x0_x1']
            points=d['erf'][d['erf_accepted']]+[ids[0],oldbox[0],oldbox[2]]
            rows=np.arange(750,950,2);supported=[]
            for row in rows:
                cr=coarse[z,int(row-box[0])]+box[2]
                supported.append(bool(np.any((abs(points[:,1]-row)<1)&(points[:,2]>=cr-40)&(points[:,2]<=cr+40))))
            record['previous_erf_supported_rows_750_949']=int(sum(supported))
        fig,axes=plt.subplots(1,1+len(proposals),figsize=(4*(1+len(proposals)),10))
        for ax in axes:
            ax.imshow(raw[z],cmap='gray',vmin=.06,vmax=.8);ax.axis('off')
        axes[0].set_title('Original raw CT')
        for ax,(name,(p,n,r)) in zip(axes[1:],proposals.items()):
            acc=r['accepted'];q=r['points_zyx']
            limit=r['at_search_limit']
            ax.scatter(p[(~acc)&(~limit),2],p[(~acc)&(~limit),1],s=1,c='orange')
            ax.scatter(p[limit,2],p[limit,1],s=1,c='#999999')
            ax.scatter(q[acc,2],q[acc,1],s=2,c='#00ff70')
            ax.set_title(name+f'\nraw fit {acc.sum()}/{len(acc)}; search bound {limit.sum()}')
        fig.suptitle(f'img_{center:04d}: gray=search bound; orange=unvalidated proposal; green=raw model pass; no certified physical surface')
        fig.tight_layout();fig.savefig(a.out/f'proposals_{center:04d}.png',dpi=110);plt.close(fig)
        record['seconds']=time.monotonic()-start;records.append(record)
        (a.out/'proposal_metrics.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--centers',default='625,750,875,988');p.add_argument('--skip-nlm',action='store_true')
    p.add_argument('--previous-profiles',type=Path)
    main(p.parse_args())
