#!/usr/bin/env python3
"""Joint opposing-edge synthetic checks; reports coverage as well as error."""
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.special import ndtr
from double_edge_fit import fit_double_profiles,select_opposing_profiles


def summary(result,truth):
    ok=result['accepted']; finite=np.isfinite(result['boundaries']).all(axis=1)
    errors=result['boundaries']-truth
    flags,counts=np.unique(result['flags'],return_counts=True)
    return {'n':len(ok),'accepted_fraction':float(ok.mean()),
            'accepted_count':int(ok.sum()),
            'candidate_count':int(finite.sum()),
            'flag_counts':{str(int(f)):int(n) for f,n in zip(flags,counts)},
            'accepted_signed_bias_left_right':np.mean(errors[ok],axis=0).tolist() if ok.any() else None,
            'accepted_absolute_boundary_error_p95':float(np.percentile(abs(errors[ok]),95)) if ok.any() else None,
            'accepted_absolute_width_error_p95':float(np.percentile(abs(np.diff(errors[ok],axis=1)),95)) if ok.any() else None,
            'all_finite_candidates_absolute_boundary_error_p95':float(np.percentile(abs(errors[finite]),95)) if finite.any() else None,
            'median_model_only_boundary_se':float(np.median(result['boundary_standard_errors_model_only'][ok])) if ok.any() else None}


def run(output):
    rng=np.random.default_rng(210244)
    t=np.arange(-7,7.001,.25)
    records=[]
    for polarity in (1,-1):
        for width in (1.5,2.6,4.):
            n=40;centres=rng.uniform(-.4,.4,n)
            truth=np.stack([centres-width/2,centres+width/2],axis=1)
            sigmas=rng.uniform(.9,1.1,n)
            a=rng.uniform(.35,.45,n) if polarity<0 else rng.uniform(.1,.2,n)
            amplitude=rng.uniform(.55,.7,n)
            slope=rng.uniform(-.002,.002,n)
            profiles=(a[:,None]+slope[:,None]*t+polarity*amplitude[:,None]*
                      (ndtr((t-truth[:,0,None])/sigmas[:,None])-
                       ndtr((t-truth[:,1,None])/sigmas[:,None])))
            profiles+=rng.normal(0,.008,profiles.shape)
            start=perf_counter();result=fit_double_profiles(t,profiles,polarity)
            record={'case':('wall' if polarity>0 else 'gap')+f'_width_{width}',
                    'width_vox':width,'sigma_range_vox':[.9,1.1],
                    'elapsed_seconds':perf_counter()-start,**summary(result,truth)}
            assert np.all(result['parameters'][result['accepted'],4]>=.7)
            assert np.all(result['parameters'][result['accepted'],4]<=8)
            # Identifiable wider pairs should have useful coverage and small
            # error. Narrow pairs may be rejected for covariance or bounds.
            if width>=2.6:
                assert record['accepted_fraction']>.8,record
                assert record['accepted_absolute_boundary_error_p95']<.2,record
            records.append(record);print(json.dumps(record),flush=True)
    # Severe unresolved case: unknown amplitude, width .8 and PSF sigma1.8
    # are strongly interchangeable. Returning a numerical optimum is not
    # sufficient evidence that two separate boundaries were measured.
    n=40;width=.8;sigma=1.8
    centres=rng.uniform(-.3,.3,n)
    truth=np.stack([centres-width/2,centres+width/2],axis=1)
    profiles=.15+.65*(ndtr((t-truth[:,0,None])/sigma)-ndtr((t-truth[:,1,None])/sigma))
    profiles+=rng.normal(0,.012,profiles.shape)
    result=fit_double_profiles(t,profiles)
    record={'case':'unresolved_width_0p8_sigma_1p8',**summary(result,truth)}
    assert record['accepted_fraction']<.2,record
    records.append(record);print(json.dumps(record),flush=True)
    # A profile with no feature must be rejected, including deterministic
    # flat data; it must not turn into a fitted wall at the bounds.
    flat=np.full((4,len(t)),.2)
    result=fit_double_profiles(t,flat)
    assert not result['accepted'].any()
    records.append({'case':'flat_no_feature','accepted_fraction':float(result['accepted'].mean())})
    output.write_text(json.dumps({'model':'two opposite Gaussian-blurred steps with shared sigma and linear background',
                                  'description':'Analytic profiles only; no real CT dimensional accuracy claim',
                                  'random_seed':210244,'t':t.tolist(),'records':records},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('double_edge_validation.json'))
    run(p.parse_args().output)
