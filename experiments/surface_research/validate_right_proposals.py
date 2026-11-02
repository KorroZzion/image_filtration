"""Known broad plane plus a no-boundary ramp; validates proposal-to-raw chain."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from right_boundary_study import candidate_path
from adaptive_profile import adaptive_refine

def run(out):
    rng=np.random.default_rng(20261103)
    zz,yy,xx=np.indices((17,180,180))
    boundary=85+.10*(yy-90)+.05*(zz-8)
    coarse=boundary[:,:,0]+3
    raw=.15+.55*ndtr((boundary-xx)/6)+rng.normal(0,.008,xx.shape)
    pts,normals,scales,limit,_=candidate_path(raw.astype(np.float32),coarse,8,(550,730,0,180))
    fit=adaptive_refine(raw,pts,normals)
    accepted=fit['accepted']&~limit
    q=fit['points_zyx'][accepted]
    error=abs(q[:,2]-(85+.1*(q[:,1]-90)+.05*(q[:,0]-8)))/np.sqrt(1+.1**2+.05**2)
    assert accepted.mean()>.8
    assert np.percentile(error,95)<.25
    ramp=.4+.001*(boundary-xx)+rng.normal(0,.008,xx.shape)
    rp,rn,_,rl,_=candidate_path(ramp.astype(np.float32),coarse,8,(550,730,0,180))
    rejected=adaptive_refine(ramp,rp,rn)
    assert not np.any(rejected['accepted']&~rl), 'A path in a ramp is not a surface measurement'
    record={'known_sigma6_plane':{'candidates':len(pts),'accepted':int(accepted.sum()),'p95_error_vox':float(np.percentile(error,95))},
            'no_boundary_linear_ramp':{'candidates':len(rp),'accepted':0},
            'note':'Analytical fields; not physical CT metrology validation'}
    out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
