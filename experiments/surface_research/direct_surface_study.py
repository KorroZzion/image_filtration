#!/usr/bin/env python3
"""Direct local 3D surface candidates: no mandatory global hysteresis seed.

The output is a measured candidate point cloud, not a closed semantic solid.
Profile acceptance is an image diagnostic, not a metrology certificate.
"""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import sys
import time
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'ct-surface-mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'ct-surface-cache'))
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True,exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure
from edge_seed_study import block, envelope_for
from profile_refinement import refine_surface, REJECTION_FLAGS


def direct_candidates(raw,z,sigma=.85,gradient_floor=.006):
    sm=ndi.gaussian_filter(raw,sigma)
    grads=np.stack(np.gradient(sm),axis=0)
    mag=np.linalg.norm(grads,axis=0)
    env=envelope_for(raw)
    y,x=np.nonzero((mag[z]>gradient_floor)&env[z])
    pts=np.stack([np.full_like(y,z),y,x],axis=1).astype(float)
    norm=grads[:,z,y,x].T
    norm/=np.maximum(np.linalg.norm(norm,axis=1,keepdims=True),1e-8)
    left=ndi.map_coordinates(mag,(pts-norm).T,order=1,mode='nearest')
    right=ndi.map_coordinates(mag,(pts+norm).T,order=1,mode='nearest')
    keep=(mag[z,y,x]>=left)&(mag[z,y,x]>=right)
    # Directional consistency suppresses incoherent grain noise. It cannot
    # distinguish a coherent reconstruction artefact from a true interface.
    avg=np.stack([ndi.gaussian_filter(g,1.2) for g in grads])
    coherence=np.linalg.norm(avg,axis=0)/np.maximum(ndi.gaussian_filter(mag,1.2),1e-8)
    return pts[keep],norm[keep],coherence[z,y[keep],x[keep]],{'sigma_vox':sigma,'gradient_floor_normalized_u16_per_vox':gradient_floor}


def write_points(path,ids,box,result):
    keys=['local50','gradient','taylor','erf']
    origin=np.array([ids[0],box[0],box[2]])
    with gzip.open(path,'wt',newline='') as f:
        w=csv.writer(f)
        header=['candidate_id','seed_source_z','seed_y','seed_x']
        for method in keys:
            header += [method+'_'+k for k in ['source_z','y','x','accepted','rejection_bits']]
        header+=['contrast_normalized_u16','snr','erf_sigma_vox','erf_relative_rms']
        w.writerow(header)
        for i in range(len(result['seed'])):
            row=[i,*np.round(result['seed'][i]+origin,6)]
            for method in keys:
                row.extend([*np.round(result[method][i]+origin,6),
                            int(result[method+'_accepted'][i]),int(result[method+'_flags'][i])])
            row.extend(float(result[k][i]) for k in ['contrast','snr','erf_sigma_vox','erf_relative_rms'])
            w.writerow(row)


def stats(result):
    record={}
    for method in ['local50','gradient','taylor','erf']:
        a=result[method+'_accepted']; shifts=result[method+'_shift_vox'][a]
        record[method]={'accepted':int(a.sum()),'candidate_count':len(a),
                        'accepted_fraction':float(a.mean()),
                        'abs_shift_p95_vox':float(np.percentile(abs(shifts),95)) if len(shifts) else None}
        flags=result[method+'_flags']
        record[method]['reject_flag_counts']={text:int(np.count_nonzero(flags&bit))
            for bit,text in {**REJECTION_FLAGS,256:'invalid_local_cubic',
                             512:'below_experimental_contrast_floor',
                             1024:'low_directional_coherence'}.items()}
    common=result['local50_accepted']&result['erf_accepted']
    record['iso50_erf_disagreement_p95_vox']=float(np.percentile(np.linalg.norm(
        result['local50'][common]-result['erf'][common],axis=1),95)) if common.any() else None
    return record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--centers',default='625,750,875,988')
    ap.add_argument('--half',type=int,default=8)
    ap.add_argument('--sigma',type=float,default=.85)
    ap.add_argument('--floor',type=float,default=.006)
    ap.add_argument('--min-contrast',type=float,default=.06)
    ap.add_argument('--min-coherence',type=float,default=.75)
    ap.add_argument('--cpp-root',type=Path,default=None)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    records=[]
    for center in map(int,args.centers.split(',')):
        start=time.monotonic();raw,ids,box=block(args.source,center,args.half)
        z=ids.index(center)
        if z<6 or len(ids)-1-z<6:
            raise ValueError('At least six real neighbouring slices on both sides are required')
        pts,norm,coherence,meta=direct_candidates(raw,z,args.sigma,args.floor)
        result=refine_surface(raw,np.zeros_like(raw,dtype=bool),pts,
                              normals_physical_zyx=norm)
        result['directional_coherence']=coherence
        for method in ['local50','gradient','taylor','erf']:
            result[method+'_model_accepted']=result[method+'_accepted'].copy()
            flags=result[method+'_flags']
            flags[result['contrast']<args.min_contrast]|=512
            flags[coherence<args.min_coherence]|=1024
            result[method+'_accepted']=flags==0
            result[method][flags!=0]=result['seed'][flags!=0]
            result[method+'_shift_vox'][flags!=0]=0
        arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(args.out/f'profiles_{center:04d}.npz',**arrays,
            source_slices=ids,box_y0_y1_x0_x1=box)
        write_points(args.out/f'points_{center:04d}.csv.gz',ids,box,result)
        y0,y1,x0,x1=box
        panels=[('Original CT',None)]
        cpp=args.cpp_root/str(center) if args.cpp_root else args.out/'no_cpp_baseline'
        for method,name in [('standard','Current 3D Canny'),('adaptive','Current adaptive Canny'),('phase','Current phase congruency')]:
            p=cpp/f'{method}_{center:04d}.png'
            panels.append((name,np.asarray(Image.open(p))[y0:y1,x0:x1]>0 if p.exists() else None))
        panels += [('Direct 3D + gradient', 'gradient'),('Direct 3D + erf', 'erf')]
        fig,axs=plt.subplots(1,len(panels),figsize=(18,10))
        for ax,(name,overlay) in zip(axs,panels):
            ax.imshow(raw[z],cmap='gray',vmin=.06,vmax=.80)
            if isinstance(overlay,np.ndarray):
                yy,xx=np.nonzero(overlay);ax.scatter(xx,yy,s=.5,c='#00ff70',linewidths=0)
            elif isinstance(overlay,str):
                accept=result[overlay+'_accepted'];q=result[overlay]
                ax.scatter(q[accept,2],q[accept,1],s=.7,c='#00ff70',linewidths=0)
                # Low-signal texture is retained in CSV, but not painted as
                # thousands of orange points on the review image.
                show_reject=(~accept)&(result['contrast']>=args.min_contrast)
                rejected=result['seed'][show_reject]
                ax.scatter(rejected[:,2],rejected[:,1],s=.4,c='#ffad32',alpha=.6,linewidths=0)
                name+=f'\naccepted {accept.mean():.1%}; orange = rejected'
            elif name!='Original CT':name+='\n(not yet run)'
            ax.axis('off');ax.set_title(name,fontsize=9)
        fig.suptitle(f'img_{center:04d}: green = candidate surface; refined 3D points projected to XY; image-profile support is not ground truth',fontsize=10)
        fig.tight_layout();fig.savefig(args.out/f'comparison_{center:04d}.png',dpi=150);plt.close(fig)
        # Export a separate accepted point cloud without artificial connections.
        accepted=result['erf_accepted'];global_zyx=result['erf'][accepted]+[ids[0],y0,x0]
        with (args.out/f'erf_points_{center:04d}.ply').open('w') as f:
            f.write('ply\nformat ascii 1.0\ncomment candidate surfels; coordinates in source voxel indices; no triangulation\n')
            f.write(f'element vertex {len(global_zyx)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
            np.savetxt(f,global_zyx[:,[2,1,0]],fmt='%.6f')
        record={'slice':center,'source_block':ids,'box':box,**meta,**stats(result),
                'experimental_min_contrast':args.min_contrast,
                'experimental_min_coherence':args.min_coherence,
                'seconds':round(time.monotonic()-start,2)}
        records.append(record);print(json.dumps(record),flush=True)
        (args.out/'metrics.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
