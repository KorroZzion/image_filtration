#!/usr/bin/env python3
"""Research-only double-edge fitting on actual neighbouring CT slice data.

Uses already sampled 3D gradient candidate positions and normals from
profiles_XXXX.npz, but reloads original uint16 TIFF data to sample profiles.
Only single-step profiles flagged for an opposing edge (bit8) are considered.
Fitted pairs remain isolated candidates: no mesh or certified dimensions.
"""
from __future__ import annotations
import argparse,csv,gzip,json,os
import tempfile
from pathlib import Path
from time import perf_counter
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'ct-double-mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'ct-surface-cache'))
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True,exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from double_edge_fit import fit_double_profiles,select_opposing_profiles,model_double,FLAGS


def load_raw(source,ids,box):
    if np.any(np.diff(ids)!=1):
        raise ValueError('Actual consecutive source_slices required; gaps cannot be interpolated')
    y0,y1,x0,x1=map(int,box)
    raw=[]
    for z in ids:
        with Image.open(source/f'img_{int(z):04d}.tif') as im:
            a=np.asarray(im)
            if a.dtype!=np.uint16 or a.ndim!=2:
                raise ValueError('Expected original single-channel uint16 TIFF')
            if y1>a.shape[0] or x1>a.shape[1] or y0<0 or x0<0:
                raise ValueError('Crop outside original image')
            raw.append(a[y0:y1,x0:x1].astype(np.float32)/65535)
    return np.stack(raw)


def stats(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    if not len(values):return None
    return {'min':float(values.min()),'p25':float(np.percentile(values,25)),
            'median':float(np.median(values)),'p75':float(np.percentile(values,75)),
            'p95':float(np.percentile(values,95)),'max':float(values.max())}


def main(args):
    args.out.mkdir(parents=True,exist_ok=True)
    records=[]
    for center in map(int,args.centers.split(',')):
        start=perf_counter();path=args.profiles/f'profiles_{center:04d}.npz'
        with np.load(path) as d:
            seed=d['seed'];normal=d['normal_physical_zyx'];singleflags=d['erf_flags']
            ids=d['source_slices'];box=d['box_y0_y1_x0_x1']
        # These experiments use the isotropic 75.4um grid from the source
        # program. No spacing calibration or micrometre conversion is made.
        normal=normal/np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
        raw=load_raw(args.input,ids,box)
        rejected_opposing=(singleflags & 8)!=0
        ids0=np.flatnonzero(rejected_opposing)
        t=np.arange(-6,6.001,.25)
        coords=seed[ids0,None]+normal[ids0,None]*t[None,:,None]
        valid=(coords>=0).all(axis=(1,2))&(coords<=np.asarray(raw.shape)[None,None]-1).all(axis=(1,2))
        ids0=ids0[valid];coords=coords[valid]
        profiles=ndi.map_coordinates(raw,coords.reshape(-1,3).T,order=1,mode='nearest').reshape(len(ids0),len(t))
        source_origin=np.array([int(ids[0]),int(box[0]),int(box[2])])
        record={'slice':center,'source_slices':ids.tolist(),'box_y0_y1_x0_x1':box.tolist(),
                'total_single_step_candidates':len(seed),
                'single_step_opposing_flag_count':int(rejected_opposing.sum()),
                'valid_opposing_profiles':len(ids0),'profile_t_range_vox':[-6,6],
                'profile_step_vox':.25,'sample_max_per_polarity':args.max_per_polarity,
                'evaluations':[],'note':'Isolated model candidates only; accepted does not establish a physical wall/gap or dimensional accuracy'}
        bundles=[]
        for polarity in (1,-1):
            name='wall' if polarity==1 else 'gap'
            eligible=select_opposing_profiles(t,profiles,polarity,max_count=max(len(profiles),1))
            selected=eligible
            if len(selected)>args.max_per_polarity:
                selected=selected[np.linspace(0,len(selected)-1,args.max_per_polarity,dtype=int)]
            y=profiles[selected];candidate=ids0[selected]
            result=fit_double_profiles(t,y,polarity,max_profiles=args.max_per_polarity)
            ok=result['accepted'];parameters=result['parameters'];boundaries=result['boundaries']
            absolute=seed[candidate,None]+normal[candidate,None]*boundaries[:,:,None]+source_origin
            np.savez_compressed(args.out/f'fits_{center:04d}_{name}.npz',
                t=t,profiles=y,candidate_indices=candidate,seed_zyx_source=seed[candidate]+source_origin,
                normal_physical_zyx=normal[candidate],single_erf_flags=singleflags[candidate],
                parameters=parameters,boundaries_relative_vox=boundaries,
                boundaries_zyx_source=absolute,accepted=ok,flags=result['flags'],
                relative_rms=result['relative_rms'],rms=result['rms'],snr=result['snr'],
                boundary_standard_errors_model_only=result['boundary_standard_errors_model_only'],
                width_standard_error_model_only=result['width_standard_error_model_only'],
                jacobian_condition_scaled=result['jacobian_condition_scaled'],
                source_slices=ids,box_y0_y1_x0_x1=box,polarity=polarity)
            with gzip.open(args.out/f'fits_{center:04d}_{name}.csv.gz','wt',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['candidate_id','seed_source_z','seed_y','seed_x','normal_z','normal_y','normal_x',
                                 'single_erf_flags','pair_kind','model_accepted','pair_flags',
                                 'background','background_slope','amplitude','center_relative_vox','width_vox','sigma_vox',
                                 'left_relative_vox','right_relative_vox',
                                 'left_source_z','left_y','left_x','right_source_z','right_y','right_x',
                                 'relative_rms','rms_u16','snr','left_se_model_only_vox','right_se_model_only_vox'])
                for j,c in enumerate(candidate):
                    writer.writerow([int(c),*(seed[c]+source_origin),*normal[c],int(singleflags[c]),name,int(ok[j]),
                                     int(result['flags'][j]),*parameters[j],*boundaries[j],*absolute[j].ravel(),
                                     result['relative_rms'][j],result['rms'][j]*65535,result['snr'][j],
                                     *result['boundary_standard_errors_model_only'][j]])
            unique,count=np.unique(result['flags'],return_counts=True)
            evaluation={'kind':name,'polarity':polarity,
                        'eligible_opposing_count':len(eligible),'evaluated_count':len(selected),
                        'accepted_count':int(ok.sum()),
                        'rejection_share_evaluated':float(1-ok.mean()) if len(ok) else None,
                        'accepted_width_vox':stats(parameters[ok,4]),
                        'accepted_sigma_vox':stats(parameters[ok,5]),
                        'accepted_amplitude_u16':stats(parameters[ok,2]*65535),
                        'accepted_also_above_diagnostic_min_amplitude_count':int(np.count_nonzero(ok & (parameters[:,2]>=args.min_amplitude))),
                        'diagnostic_min_amplitude_normalized':args.min_amplitude,
                        'accepted_relative_rms':stats(result['relative_rms'][ok]),
                        'accepted_rms_u16':stats(result['rms'][ok]*65535),
                        'accepted_model_only_boundary_se_vox':stats(result['boundary_standard_errors_model_only'][ok].ravel()),
                        'flag_counts_evaluated':{str(int(f)):int(n) for f,n in zip(unique,count)},
                        'flag_reason_counts_evaluated':{label:int(np.count_nonzero(result['flags'] & bit)) for bit,label in FLAGS.items()}}
            record['evaluations'].append(evaluation)
            bundles.append((name,polarity,candidate,y,result))
        # Show accepted and rejected fits from both polarities. Selection is
        # deterministic and cannot serve as a representative accuracy sample.
        shown=[]
        for group,polarity,candidate,y,result in bundles:
            for accepted in (True,False):
                available=np.flatnonzero(result['accepted']==accepted)
                count=min(4,len(available))
                if count:
                    for j in available[np.linspace(0,len(available)-1,count,dtype=int)]:
                        shown.append((group,polarity,int(candidate[j]),y[j],result,int(j)))
        used={(entry[0],entry[2]) for entry in shown}
        for bi,(group,polarity,candidate,y,result) in enumerate(bundles):
            if len(shown)>=16:break
            available=np.array([j for j in range(len(candidate))
                                if not result['accepted'][j] and (group,int(candidate[j])) not in used],int)
            count=min(len(available),int(np.ceil((16-len(shown))/(len(bundles)-bi))))
            if count:
                for j in available[np.linspace(0,len(available)-1,count,dtype=int)]:
                    shown.append((group,polarity,int(candidate[j]),y[j],result,int(j)))
                    used.add((group,int(candidate[j])))
        fig,axes=plt.subplots(4,4,figsize=(15,12),sharex=True)
        for ax in axes.flat:ax.axis('off')
        for ax,(group,polarity,c,y,r,j) in zip(axes.flat,shown):
            ax.axis('on');ok=bool(r['accepted'][j]);color='#087f23' if ok else '#d57800'
            ax.plot(t,y*65535,color='#1769aa',lw=1,label='original samples')
            if np.isfinite(r['parameters'][j]).all():
                ax.plot(t,model_double(t,r['parameters'][j],polarity)*65535,color='#222222',lw=1,label='joint fit')
                for b in r['boundaries'][j]:ax.axvline(b,color=color,ls='--',lw=.9)
            ax.axvline(0,color='#999999',lw=.5)
            zyx=seed[c]+source_origin
            ax.set_title(f'{group} #{c} '+('accepted model' if ok else 'rejected')+
                         f' flags={int(r["flags"][j])}\n'+
                         f'ZYX={zyx[0]:.2f},{zyx[1]:.1f},{zyx[2]:.1f}; w={r["parameters"][j,4]:.2f}',fontsize=8,color=color)
            ax.tick_params(labelsize=7);ax.set_xlabel('normal coordinate, voxels',fontsize=7)
            ax.set_ylabel('original uint16 intensity',fontsize=7)
        fig.suptitle(f'img_{center:04d}: opposing-edge profile fits — isolated research candidates\n'
                     'Green: model gates passed; orange: rejected. Neither certifies a wall/gap or dimensional accuracy.',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(args.out/f'example_profiles_{center:04d}.png',dpi=150);plt.close(fig)
        # Locations on the actual central CT slice. No polylines are inferred.
        z=int(np.flatnonzero(ids==center)[0]);fig,ax=plt.subplots(figsize=(7,12))
        ax.imshow(raw[z],cmap='gray',vmin=.06,vmax=.8)
        for group,polarity,c,y,r,j in shown:
            yx=seed[c,1:]
            ax.scatter(yx[1],yx[0],s=12,color='#00dd66' if r['accepted'][j] else 'orange')
            ax.annotate(str(c),(yx[1],yx[0]),fontsize=5,color='white')
        ax.set_title(f'img_{center:04d}: shown profile seed locations\nGreen=model accepted; orange=rejected; no connected surface',fontsize=10)
        ax.axis('off');fig.tight_layout();fig.savefig(args.out/f'profile_locations_{center:04d}.png',dpi=150);plt.close(fig)
        record['elapsed_seconds']=perf_counter()-start
        records.append(record);print(json.dumps(record),flush=True)
    (args.out/'metrics.json').write_text(json.dumps({'method':'shared-sigma double blurred-step fit',
        'flag_definitions':FLAGS,'limitations':'No ground truth, physical semantic labels, calibrated PSF, or uncertainty budget. Rejected candidates are not exported as measured surfaces.',
        'records':records},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--profiles',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--centers',default='750,875')
    p.add_argument('--max-per-polarity',type=int,default=200)
    p.add_argument('--min-amplitude',type=float,default=.06,help='Additional reported contrast diagnostic, not used to alter model acceptance')
    main(p.parse_args())
