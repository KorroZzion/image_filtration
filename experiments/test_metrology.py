"""Independent analytic imaging cases for the experimental CT surface methods."""
import unittest
import numpy as np
from scipy import ndimage as ndi
from scipy.special import erf
from metrology_study import fit_profiles,local_fit,fit_surface,recover_thin_walls,farid_gradients


class TestMetrology(unittest.TestCase):
    def test_farid_derivatives_detect_all_axes_with_correct_scale_and_sign(self):
        z,y,x=np.indices((21,23,25),dtype=np.float32)
        image=.1*z+.2*y+.3*x
        _,g,_=farid_gradients(image)
        for field,expected in zip(g,[.1,.2,.3]):
            self.assertTrue(np.allclose(field[4:-4,4:-4,4:-4],expected,atol=1e-6))

    def test_measured_pair_can_propose_a_missing_sheet(self):
        z,y,x=np.indices((13,68,64))
        truth=((x>=20)&(x<=24)&(y>8)&(y<59))|((x>=33)&(x<=36)&(y>8)&(y<59))
        raw=ndi.gaussian_filter((.12+.5*truth).astype(np.float32),.9)
        raw+=np.random.default_rng(12).normal(0,.001,raw.shape)
        baseline=truth&(x<28)
        recovered,support,info=recover_thin_walls(raw,baseline)
        target=truth&(x>28)&(y>18)&(y<49)
        recall=float(recovered[target].mean())
        precision=float(truth[support].mean())
        print('Missing sheet recovery',recall,precision,info,flush=True)
        self.assertGreater(recall,.65)
        self.assertGreater(precision,.95)

    def test_blurred_step_with_background_drift(self):
        rng=np.random.default_rng(15)
        t=np.arange(-6,8.001,.35)
        displacement=rng.uniform(-.8,.8,60)
        sigma=rng.uniform(.65,1.5,60)
        cdf=.5*(1+erf((t[None]-displacement[:,None])/(np.sqrt(2)*sigma[:,None])))
        profiles=.12+.02*t[None]+.48*cdf+rng.normal(0,.002,cdf.shape)
        fit,rmse,se=fit_profiles(profiles,t)
        error=abs(fit[:,3]-displacement)
        print('Step + drift P95',np.percentile(error,95),flush=True)
        self.assertLess(float(np.percentile(error,95)),.08)
        self.assertLess(float(np.median(rmse)),.004)

    def test_opposing_blurred_surfaces_need_pair_model(self):
        rng=np.random.default_rng(73)
        t=np.arange(-6,8.001,.35)
        displacement=rng.uniform(-.6,.6,80)
        width=rng.uniform(1.5,4.5,80)
        sigma=rng.uniform(.8,1.4,80)
        cdf=lambda d:.5*(1+erf((t[None]-d[:,None])/(np.sqrt(2)*sigma[:,None])))
        profiles=.12+.006*t[None]+.5*(cdf(displacement)-cdf(displacement+width))
        profiles+=rng.normal(0,.002,profiles.shape)
        pair,_,_=fit_profiles(profiles,t,width+ .25)
        single,_,_=fit_profiles(profiles,t)
        pair_error=abs(pair[:,3]-displacement)
        single_error=abs(single[:,3]-displacement)
        print('Thin slab P95, paired',np.percentile(pair_error,95),'single',np.percentile(single_error,95),flush=True)
        # Below roughly two blur sigmas, width, amplitude and PSF are strongly
        # coupled; this broad case deliberately includes those ambiguities.
        self.assertLess(float(np.percentile(pair_error,95)),.35)
        resolvable=width>2.5*sigma
        self.assertLess(float(np.percentile(pair_error[resolvable],95)),.08)
        self.assertLess(float(np.median(pair_error)),float(np.median(single_error))*.3)

    def test_local_volume_energy_recovers_an_erased_internal_web(self):
        z,y,x=np.indices((17,88,88))
        truth=(y>10)&(y<77)&(x>12)&(x<75)
        truth &= ~(((y>24)&(y<64)&(x>22)&(x<39))|((y>24)&(y<64)&(x>44)&(x<65)))
        intensity=.12+truth*(.35+.2*y/88)
        raw=ndi.gaussian_filter(intensity.astype(np.float32),.8)
        raw+=np.random.default_rng(1).normal(0,.003,raw.shape)
        initial=truth.copy();initial[:,25:64,39:45]=False
        recovered,_,_=local_fit(raw,initial,scale=7,rounds=4,inner=20)
        target=truth&(y>28)&(y<60)&(x>39)&(x<44)
        recall=float(recovered[target].mean())
        dice=2*(recovered&truth).sum()/(recovered.sum()+truth.sum())
        print('Erased web recovered',recall,'volume Dice',dice,flush=True)
        self.assertGreater(recall,.9)
        self.assertGreater(dice,.97)

    def test_psf_surface_has_actual_z_motion_on_oblique_boundary(self):
        from test_canny3d import phantom
        from skimage import filters
        raw,truth,_,roi,axis,radius=phantom(56)
        mask=raw>filters.threshold_otsu(raw[roi])
        base,moved,faces,accepted,shift,info,quality=fit_surface(raw,mask,slice(0,len(raw)))
        p=moved-(len(raw)-1)/2
        radial=np.linalg.norm(p,axis=1)
        axial=p@axis
        cylinder=np.sqrt(np.maximum(np.sum(p*p,axis=1)-axial*axial,0))
        error=abs(np.minimum(radius-radial,cylinder-len(raw)*.105))
        print('PSF 3D phantom P95',np.percentile(error,95),info,flush=True)
        self.assertGreater(accepted.mean(),.5)
        self.assertLess(float(np.percentile(error,95)),.8)
        self.assertGreater(float(np.percentile(abs(moved[:,0]-base[:,0]),95)),.05)


if __name__=='__main__':unittest.main(verbosity=2)
