#!/usr/bin/env python3
"""Broad 3D PSF, voxel-noise and rejection checks for adaptive profiles."""
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.special import ndtr
from adaptive_profile import adaptive_refine,estimate_volume_noise,FLAGS


def run(output):
    rng=np.random.default_rng(20261102)
    shape=(80,84,96)
    q=np.moveaxis(np.indices(shape,dtype=float),0,-1)
    centre=np.array([39.7,41.3,47.2])
    n=np.array([.42,.3,.856]);n/=np.linalg.norm(n)
    e1=np.cross(n,[0,1,0]);e1/=np.linalg.norm(e1)
    e2=np.cross(n,e1)
    signed=(q-centre)@n
    count=240
    tangent=rng.uniform(-8,8,(count,2))
    points_on_surface=centre+tangent[:,0,None]*e1+tangent[:,1,None]*e2
    shift_seed=rng.uniform(-.9,.9,count)
    points=points_on_surface+shift_seed[:,None]*n
    normals=np.broadcast_to(n,points.shape)
    cases=[]
    for sigma in (1.,3.,6.):
        raw=.12+.65*ndtr(signed/sigma)+.0015*signed
        raw+=rng.normal(0,.008,shape)
        cases.append((f'broad_psf_sigma_{sigma}',raw,True,False))
    cupping=.12+.65*ndtr(signed/6)+.001*signed+.00005*signed**2
    cupping+=rng.normal(0,.008,shape)
    cases.append(('broad_psf_6_with_quadratic_cupping_model_mismatch',cupping,True,True))
    linear=.4+.004*signed+rng.normal(0,.008,shape)
    cases.append(('negative_linear_ramp',linear,False,False))
    wall=.12+.65*(ndtr(signed)-ndtr(signed-4))+rng.normal(0,.008,shape)
    cases.append(('negative_wall_two_opposing_edges_width4',wall,False,False))
    split=.12+.32*ndtr(signed+3)+.33*ndtr(signed-3)+rng.normal(0,.008,shape)
    cases.append(('negative_false_bridge_two_rising_edges_separation6',split,False,False))
    records=[]
    for label,raw,is_edge,mismatch in cases:
        start=perf_counter()
        result=adaptive_refine(raw.astype(np.float32),points,normals)
        accepted=result['accepted'];error=(result['points_zyx']-centre)@n
        counts={str(int(r)):int(np.count_nonzero(result['chosen_radius_vox']==r)) for r in (6,12,24)}
        record={'case':label,'n':count,'accepted_count':int(accepted.sum()),
                'accepted_fraction':float(accepted.mean()),'chosen_radius_counts':counts,
                'estimated_original_voxel_noise':result['noise_sigma_volume'],
                'true_noise_sigma':.008,'elapsed_seconds':perf_counter()-start,
                'accepted_error_signed_mean_vox':float(np.mean(error[accepted])) if accepted.any() else None,
                'accepted_error_abs_p95_vox':float(np.percentile(abs(error[accepted]),95)) if accepted.any() else None,
                'accepted_fitted_sigma_median_vox':float(np.median(result['sigma_vox'][accepted])) if accepted.any() else None,
                'accepted_model_se_median_vox':float(np.median(result['position_se_model_only_vox'][accepted])) if accepted.any() else None,
                'rejection_counts_by_radius':{str(r):{label:int(np.count_nonzero(result['all_flags'][:,ri]&bit))
                    for bit,label in FLAGS.items()} for ri,r in enumerate((6,12,24))}}
        assert np.isfinite(result['points_zyx']).all()
        assert np.array_equal(result['points_zyx'][~accepted],points[~accepted])
        assert np.max(abs(result['shift_vox']))<=3.0001
        if is_edge and not mismatch:
            assert record['accepted_fraction']>.75,record
            assert record['accepted_error_abs_p95_vox']<.25,record
        elif not is_edge:
            assert record['accepted_fraction']<.03,record
        records.append(record)
        np.savez_compressed(output.parent/f'validation_{label}.npz',
                             **{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        print(json.dumps(record),flush=True)
    # Supply known original-grid noise separately: supports the same broad
    # physical transition without deriving a noise floor from interpolation.
    raw=.12+.65*ndtr(signed/6)+rng.normal(0,.008,shape)
    result=adaptive_refine(raw.astype(np.float32),points,normals,noise_sigma=.008)
    assert result['accepted'].mean()>.75
    # No extrapolation: points whose longest profile exits the volume retain
    # bit1 in that window even if a shorter window happens to be supported.
    near=np.array([[4,4,4]],float)
    out=adaptive_refine(raw.astype(np.float32),near,np.array([[-1,-1,-1]],float),noise_sigma=.008)
    assert out['all_flags'][0,1]&1 and out['all_flags'][0,2]&1
    output.write_text(json.dumps({'description':'Known analytic 3D normal-blur fields; no real CT accuracy claim',
                                  'random_seed':20261102,'records':records},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    run(args.output)
