"""Controlled phantom comparisons; none of these errors measure real CT."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import erf
from skimage import filters
from boundary_study import plt
from metrology_study import fit_profiles,fit_surface,profile_model
from test_canny3d import phantom
from canny3d_study import plane_segments
from matplotlib.collections import LineCollection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for seed in [0,1,2]:
        raw,truth,sdf,roi,axis,radius=phantom(56)
        raw+=np.random.default_rng(100+seed).normal(0,.002,raw.shape).astype(np.float32)
        mask=raw>filters.threshold_otsu(raw[roi])
        models={}
        for name in ['gradient','farid','psf']:
            result=fit_surface(raw,mask,slice(0,len(raw)),method=name)
            base,moved,faces,accepted,shift,info=result[:6]
            errorstats={}
            for key,vertices in [('initial',base),('refined',moved)]:
                p=vertices-(len(raw)-1)/2
                sphere=np.linalg.norm(p,axis=1)
                cylinder=np.sqrt(np.maximum(np.sum(p*p,axis=1)-(p@axis)**2,0))
                error=abs(np.minimum(radius-sphere,cylinder-len(raw)*.105))
                errorstats[key]={'median_vox':float(np.median(error)),
                                 'p95_vox':float(np.percentile(error,95))}
            record={'phantom':'voxelized sphere and inclined bore, Gaussian blur, intensity drift and noise',
                    'noise_seed':100+seed,'method':name,'errors_against_analytic_surfaces':errorstats,
                    'accepted_fraction':float(accepted.mean())}
            print(record,flush=True);records.append(record);models[name]=(moved,faces)
        if seed==0:
            fig,axes=plt.subplots(1,4,figsize=(16,4.5))
            for ax,title in zip(axes,['Known analytic geometry','3D Canny + Gaussian gradient','3D Canny + Farid gradient','Raw profile PSF fit']):
                ax.imshow(raw[28],cmap='gray',vmin=.1,vmax=.75);ax.axis('off');ax.set_title(title)
                ax.contour(sdf[28],levels=[0],colors=['#ff9d00'],linewidths=.8)
            for ax,(v,f) in zip(axes[1:],models.values()):
                lines=plane_segments(v,f,28.)
                ax.add_collection(LineCollection(lines[:,:,[1,0]],colors='#00ff70',linewidths=.8))
            fig.suptitle('Orange=analytic truth; green=estimated. Phantom only.')
            fig.tight_layout();fig.savefig(args.output/'phantom_surfaces.png',dpi=160);plt.close(fig)
    rng=np.random.default_rng(73);t=np.arange(-6,8.001,.35)
    displacement=rng.uniform(-.6,.6,80);width=rng.uniform(1.5,4.5,80);sigma=rng.uniform(.8,1.4,80)
    cdf=lambda d:.5*(1+erf((t[None]-d[:,None])/(np.sqrt(2)*sigma[:,None])))
    profiles=.12+.006*t[None]+.5*(cdf(displacement)-cdf(displacement+width))
    profiles+=rng.normal(0,.002,profiles.shape)
    pair,_,se=fit_profiles(profiles,t,width+.25)
    single,_,_=fit_profiles(profiles,t)
    profiles_result={}
    for key,fit in [('single_interface',single),('paired_interfaces',pair)]:
        error=abs(fit[:,3]-displacement)
        resolved=width>2.5*sigma
        profiles_result[key]={'median_vox':float(np.median(error)),
            'p95_vox':float(np.percentile(error,95)),
            'resolved_cases':int(resolved.sum()),'resolved_p95_vox':float(np.percentile(error[resolved],95))}
    chosen=int(np.argmin(abs(width-2.5)))
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(t,profiles[chosen],'o',markersize=3,label='Simulated CT samples')
    ax.plot(t,profile_model(single[chosen:chosen+1],t)[0][0],label='Single interface model')
    ax.plot(t,profile_model(pair[chosen:chosen+1],t,True)[0][0],label='Two-interface model')
    for d in [displacement[chosen],displacement[chosen]+width[chosen]]:ax.axvline(d,c='black',ls='--',lw=.8)
    ax.set_xlabel('Distance along normal, voxels');ax.set_ylabel('Intensity')
    ax.set_title('Thin-wall phantom: dashed lines are the true surfaces');ax.legend()
    fig.tight_layout();fig.savefig(args.output/'thin_wall_profiles.png',dpi=160);plt.close(fig)
    (args.output/'phantom_benchmark.json').write_text(json.dumps({
        'real_CT_accuracy_measured':False,'volume_cases':records,'thin_wall_profiles':profiles_result,
        'limitations':'Synthetic data; Gaussian PSF assumed, CT systematic errors absent.'},indent=2)+'\n')


if __name__=='__main__':main()
