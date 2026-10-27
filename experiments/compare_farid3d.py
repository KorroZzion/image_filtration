"""Farid--Simoncelli 3D gradients with the same adaptive Canny and seed mask."""
import argparse,json
from pathlib import Path
import numpy as np
from boundary_study import load,plt
from matplotlib.collections import LineCollection
from canny3d_study import save_ply,plane_segments
from metrology_study import fit_surface
from compare_metrology import geometry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    reference=Path('results/canny3d_v2/supported_refinement/block_0738_0762')
    raw=np.stack([load(Path('CT_Test'),i)[459:1370,286:697] for i in range(730,771)])
    core=slice(8,33)
    mask=np.pad(np.load(reference/'initializer_mask.npz')['mask'],((8,8),(0,0),(0,0)),mode='edge')
    print('Farid 3D, adaptive Canny and XYZ refinement',flush=True)
    base,moved,faces,accepted,shift,info=fit_surface(raw,mask,core,method='farid')
    check=geometry(base,moved,faces);info['geometry']=check
    info.update({'source_first':738,'source_last':762,'context_first':730,'context_last':770,
                 'crop_box_y0_y1_x0_x1':[459,1370,286,697],
                 'real_CT_accuracy_measured':False,
                 'context_mask_note':'Geometry context held constant; intensities are real CT slices.'})
    np.savez_compressed(args.output/'surface_coordinates.npz',seed_zyx=base,refined_zyx=moved,
        faces_zyx=faces,accepted=accepted,normal_shift_vox=shift,source_first=738,crop_origin_yx=[459,286])
    if check['passed']:save_ply(args.output/'surface_voxels.ply',moved,faces,(459,286))
    old=np.load(reference/'surface_coordinates.npz')
    fig,axes=plt.subplots(1,3,figsize=(12,12))
    for ax,title in zip(axes,['Original CT','Previous Gaussian derivative','Farid 7-tap derivative (3D)']):
        ax.imshow(raw[20],cmap='gray',vmin=.06,vmax=.8);ax.set_title(title,fontsize=10);ax.axis('off')
    for ax,v,f in [(axes[1],old['refined_zyx'],old['faces_zyx']),(axes[2],moved,faces)]:
        lines=plane_segments(v,f,12)
        ax.add_collection(LineCollection(lines[:,:,[1,0]],colors='#00ff70',linewidths=.6))
    fig.tight_layout();fig.savefig(args.output/'farid_comparison_0750.png',dpi=150);plt.close(fig)
    (args.output/'metrics.json').write_text(json.dumps(info,indent=2)+'\n');print(info,flush=True)


if __name__=='__main__':main()
