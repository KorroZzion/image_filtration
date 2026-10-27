"""Checks against analytic 3D geometry, not against the implementation itself."""
import unittest
import numpy as np
from scipy import ndimage as ndi
from skimage import filters
from boundary_study import largest

from canny3d_study import (gradients,nms3d,canny_variants,local_high_threshold,
                           oriented_watershed,refine_mesh,plane_segments)


def phantom(size=72):
    coordinates=np.indices((size,size,size),dtype=np.float32)
    p=coordinates-(size-1)/2
    radius=size*.34
    axis=np.array([.65,.25,.718],np.float32);axis/=np.linalg.norm(axis)
    along=np.einsum('i,ijkl->jkl',axis,p)
    distance=np.sqrt(np.maximum(np.sum(p*p,axis=0)-along*along,0))
    sdf=np.minimum(radius-np.sqrt(np.sum(p*p,axis=0)),distance-size*.105)
    truth=sdf>0
    gain=.42+.20*coordinates[2]/(size-1)
    raw=.12+gain*truth
    raw=ndi.gaussian_filter(raw,1.0)
    raw+=np.random.default_rng(42).normal(0,.003,raw.shape)
    roi=np.sqrt(np.sum(p*p,axis=0))<radius+6
    return raw.astype(np.float32),truth,sdf,roi,axis,radius


class TestCanny3D(unittest.TestCase):
    def test_exact_slice_plane_intersections_keep_cut_edges(self):
        vertices=np.array([[0,0,0],[0,2,0],[1,1,1]],np.float32)
        faces=np.array([[0,1,2]])
        segments=plane_segments(vertices,faces,0.)
        self.assertEqual(segments.shape,(1,2,2))
        self.assertAlmostEqual(float(np.linalg.norm(segments[0,1]-segments[0,0])),2.)
        self.assertEqual(len(plane_segments(vertices,faces,2.)),0)

    def test_detects_a_boundary_visible_only_in_z(self):
        raw=np.zeros((33,40,40),np.float32);raw[16:]=.6
        smooth,g,mag,_=gradients(raw)
        self.assertGreater(float(abs(g[0]).max()),.1)
        self.assertLess(float(abs(g[1]).max()),1e-7)
        self.assertLess(float(abs(g[2]).max()),1e-7)
        edges=canny_variants(raw,smooth,g,mag,np.ones(raw.shape,bool))
        detected=np.argwhere(edges['supported'])
        self.assertGreater(len(detected),1000)
        self.assertTrue(np.all((detected[:,0]>=15)&(detected[:,0]<=16)))

    def test_adaptive_threshold_responds_to_local_noise(self):
        rng=np.random.default_rng(17)
        smooth=np.full((13,64,128),.2,np.float32)
        residual=rng.normal(0,.001,smooth.shape).astype(np.float32)
        residual[:,:,64:]*=40
        magnitude=np.full(smooth.shape,.005,np.float32)
        nms=np.ones(smooth.shape,bool)
        high,_=local_high_threshold(smooth+residual,smooth,magnitude,nms,nms,tile=32)
        self.assertGreater(float(high[:,:,100:].mean()),float(high[:,:,:24].mean())*3)

    def test_surface_detection_and_material_side_on_3d_bored_sphere(self):
        raw,truth,sdf,roi,_,_=phantom()
        smooth,g,mag,_=gradients(raw)
        edges=canny_variants(raw,smooth,g,mag,roi)
        detected=edges['supported']
        precision=float(np.mean(abs(sdf[detected])<2))
        distance=ndi.distance_transform_edt(~detected)
        recall=float(np.mean(distance[(abs(sdf)<.7)&roi]<2))
        material,_,_=oriented_watershed(smooth,g,mag,edges,roi)
        dice=2*np.count_nonzero(material&truth)/(material.sum()+truth.sum())
        print('Analytic 3D phantom: tolerance-2 precision',precision,'recall',recall,'mask Dice',dice,flush=True)
        self.assertGreater(precision,.95)
        self.assertGreater(recall,.9)
        self.assertGreater(dice,.94)

    def test_xyz_refinement_localizes_analytic_surfaces(self):
        raw,truth,sdf,roi,axis,radius=phantom()
        for mode in ['gaussian','diffusion']:
            with self.subTest(mode=mode):
                smooth,g,mag,_=gradients(raw,mode)
                edges=canny_variants(raw,smooth,g,mag,roi)
                inferred=largest(smooth>filters.threshold_otsu(smooth[roi]))
                outer=np.stack([ndi.binary_fill_holes(plane) for plane in inferred])
                material,_,_=oriented_watershed(smooth,g,mag,edges,roi,outer)
                base,moved,faces,accepted,shift,info=refine_mesh(material,smooth,g,mag,edges['coherence'],slice(0,len(raw)),edges['adaptive'])
                p=moved-(len(raw)-1)/2
                sphere=np.linalg.norm(p,axis=1)
                along=p@axis
                cylinder=np.sqrt(np.maximum(np.sum(p*p,axis=1)-along*along,0))
                error=abs(np.minimum(radius-sphere,cylinder-len(raw)*.105))
                print('Analytic surface',mode,'XYZ refinement P95 error:',float(np.percentile(error,95)),info,flush=True)
                self.assertGreater(float(accepted.mean()),.6)
                self.assertGreater(float(np.percentile(abs(moved[:,0]-base[:,0]),95)),.05)
                self.assertLess(float(np.percentile(error,95)),1.)
                self.assertLessEqual(float(abs(shift).max()),2.001)


if __name__=='__main__':
    unittest.main(verbosity=2)
