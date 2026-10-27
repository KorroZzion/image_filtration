"""Record genuine 3D phantom metrics; real CT accuracy needs reference labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage import filters
from boundary_study import largest
from test_canny3d import phantom
from canny3d_study import gradients,canny_variants,oriented_watershed,refine_mesh


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw,truth,sdf,roi,axis,radius=phantom()
    records=[]
    for mode in ['gaussian','diffusion']:
        smooth,g,mag,filter_info=gradients(raw,mode)
        edges=canny_variants(raw,smooth,g,mag,roi)
        record={'filter':mode,**filter_info,'canny':{}}
        for name in ['standard','adaptive','supported']:
            edge=edges[name]
            distance=ndi.distance_transform_edt(~edge)
            record['canny'][name]={'precision_tolerance_2vox':float(np.mean(abs(sdf[edge])<2)),
                                   'recall_tolerance_2vox':float(np.mean(distance[(abs(sdf)<.7)&roi]<2)),
                                   'edge_voxels':int(edge.sum())}
        inferred=largest(smooth>filters.threshold_otsu(smooth[roi]))
        outer=np.stack([ndi.binary_fill_holes(p) for p in inferred])
        material,_,seg_info=oriented_watershed(smooth,g,mag,edges,roi,outer)
        base,moved,faces,accepted,shift,refine_info=refine_mesh(material,smooth,g,mag,edges['coherence'],slice(0,len(raw)),edges['adaptive'])
        def errors(vertices):
            p=vertices-(len(raw)-1)/2
            r=np.linalg.norm(p,axis=1)
            axial=p@axis
            tubular=np.sqrt(np.maximum(np.sum(p*p,axis=1)-axial*axial,0))
            return abs(np.minimum(radius-r,tubular-len(raw)*.105))
        record['segmentation_dice']=float(2*np.count_nonzero(material&truth)/(material.sum()+truth.sum()))
        record['seed_surface_error_p95_vox']=float(np.percentile(errors(base),95))
        record['refined_surface_error_p95_vox']=float(np.percentile(errors(moved),95))
        record['refinement']=refine_info
        records.append(record)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
