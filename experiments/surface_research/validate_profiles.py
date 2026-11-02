#!/usr/bin/env python3
"""Known analytic 3D boundaries; localization checks, not CT calibration."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.special import ndtr
from skimage.measure import marching_cubes
from profile_refinement import refine_surface


def validate(output):
    rng = np.random.default_rng(20261102)
    shape=(64, 72, 76)
    q=np.moveaxis(np.indices(shape, dtype=np.float64),0,-1)
    centre=np.array([31.35, 35.2, 37.1])
    cases=[]
    n=np.array([.76, -.37, .534]); n/=np.linalg.norm(n)
    signed=(q-centre)@n
    for label, noise, sigma in [('tilted_plane', 0., 1.0),
                                ('tilted_plane_noise', .012, 1.0),
                                ('broad_plane_noise', .01, 1.8)]:
        raw=.12+.65*ndtr(signed/sigma)
        # Linear bias emulates a smoothly varying background, for which a
        # global half threshold is not locally calibrated.
        raw += .04*(q[...,2]-centre[2])/shape[2]
        raw += rng.normal(0,noise,shape)
        cases.append((label,raw,signed>0,lambda v,n=n:abs((v-centre)@n)))
    # Deliberate model mismatch: a non-symmetric PSF with zero first moment.
    # The nominal geometric boundary is still signed=0; an excellent erf
    # residual alone need not imply unbiased boundary localization.
    asymmetric=.12+.65*(.7*ndtr((signed+.3)/.7)+.3*ndtr((signed-.7)/1.5))
    asymmetric+=rng.normal(0,.006,shape)
    cases.append(('asymmetric_psf_model_mismatch',asymmetric,signed>0,
                   lambda v,n=n:abs((v-centre)@n)))
    radius=22.4
    r=np.linalg.norm(q-centre,axis=-1)
    raw=.12+.65*ndtr((radius-r)/1.1)+rng.normal(0,.008,shape)
    cases.append(('sphere_noise',raw,r<radius,
                   lambda v:abs(np.linalg.norm(v-centre,axis=1)-radius)))
    # Two interfaces overlap in a thin shell: a single isolated-edge model
    # should reject their profiles rather than claim accurate localization.
    inner,outer=20.4,23.0
    raw=.12+.65*(ndtr((r-inner)/1.0)-ndtr((r-outer)/1.0))
    raw+=rng.normal(0,.008,shape)
    cases.append(('thin_shell_2p6vox',raw,(r>inner)&(r<outer),
                   lambda v:np.minimum(abs(np.linalg.norm(v-centre,axis=1)-inner),
                                       abs(np.linalg.norm(v-centre,axis=1)-outer))))
    records=[]
    for label,raw,mask,distance in cases:
        vertices=marching_cubes(mask.astype(np.float32),.5,allow_degenerate=False)[0]
        vertices=vertices[(vertices>7).all(axis=1)&(vertices<np.array(shape)-8).all(axis=1)]
        if len(vertices)>4500:
            vertices=vertices[np.linspace(0,len(vertices)-1,4500,dtype=int)]
        started=perf_counter()
        found=refine_surface(raw.astype(np.float32),mask,vertices)
        if label == 'tilted_plane':
            supplied=refine_surface(raw.astype(np.float32),mask,vertices,
                                    normals_physical_zyx=found['normal_physical_zyx'])
            for method in ('local50','gradient','taylor','erf'):
                assert np.array_equal(found[method+'_accepted'],supplied[method+'_accepted'])
                assert np.allclose(found[method],supplied[method],atol=1e-5), 'Provided normal route must agree'
        record={'case':label,'n':len(vertices),'elapsed_seconds':perf_counter()-started,
                'seed_error_p95_vox':float(np.percentile(distance(vertices),95)),
                'methods':{}}
        for method in ['local50','gradient','taylor','erf']:
            valid=found[method+'_accepted']
            error=distance(found[method][valid])
            seed_on_same=distance(vertices[valid])
            flag,count=np.unique(found[method+'_flags'],return_counts=True)
            record['methods'][method]={'accepted_fraction':float(valid.mean()),
                'accepted_count':int(valid.sum()),
                'same_subset_seed_p95_vox':float(np.percentile(seed_on_same,95)) if len(error) else None,
                'error_median_vox':float(np.median(error)) if len(error) else None,
                'error_p95_vox':float(np.percentile(error,95)) if len(error) else None,
                'flag_counts':{str(int(f)):int(c) for f,c in zip(flag,count)}}
            assert np.array_equal(found[method][~valid],vertices[~valid]), 'Rejected points must retain seeds'
            assert np.max(abs(found[method+'_shift_vox']))<=2.0001
            assert np.isfinite(found[method]).all()
        # Normal-profile fitting should improve known isolated interfaces.
        if 'thin' not in label and 'mismatch' not in label:
            m=record['methods']['erf']
            assert m['accepted_fraction']>.9, record
            assert m['error_p95_vox'] < .18, record
            assert m['error_p95_vox']<m['same_subset_seed_p95_vox'], record
        elif 'thin' in label:
            assert record['methods']['erf']['accepted_fraction']<.1, record
        records.append(record)
        print(json.dumps(record),flush=True)
    # Anisotropic test uses the same physical plane sampled at wider Z pitch.
    spacing=np.array([1.7,1.0,1.0])
    physical=q*spacing
    phys_centre=centre*spacing
    n=np.array([.7,.3,.648]); n/=np.linalg.norm(n)
    signed=(physical-phys_centre)@n
    raw=.12+.65*ndtr(signed/1.2)
    mask=signed>0
    vertices=marching_cubes(mask.astype(np.float32),.5)[0]
    vertices=vertices[(vertices>7).all(axis=1)&(vertices<np.array(shape)-8).all(axis=1)][::3]
    found=refine_surface(raw.astype(np.float32),mask,vertices,spacing)
    valid=found['erf_accepted']
    error=abs((found['erf'][valid]*spacing-phys_centre)@n)
    record={'case':'anisotropic_tilted_plane','n':len(vertices),
            'accepted_fraction':float(valid.mean()),'erf_error_p95_physical_units':float(np.percentile(error,95))}
    assert record['accepted_fraction']>.9 and record['erf_error_p95_physical_units']<.2, record
    records.append(record)
    output.write_text(json.dumps({'description':'Analytic synthetic normal-blur fields; no real CT accuracy claim',
                                  'random_seed':20261102,'records':records},indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('profile_validation.json'))
    args=parser.parse_args()
    validate(args.output)
