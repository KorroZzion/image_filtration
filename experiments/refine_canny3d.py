"""Canny-supported XYZ refinement of explicit, preserved material baselines.

Initializer selection is explicit per block, not a universal classifier.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from scipy import ndimage as ndi
import numpy as np
from PIL import Image
from boundary_study import load,segment_slice,plt
from matplotlib.collections import LineCollection
from canny3d_study import gradients,canny_variants,refine_mesh,plane_segments,save_ply


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('CT_Test'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--canny-results',type=Path,required=True)
    parser.add_argument('--thick-initializer',type=Path,required=True)
    parser.add_argument('--thin-initializer',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    summaries=[]
    for first,last in [(500,524),(738,762),(976,1000)]:
        name=f'block_{first:04d}_{last:04d}'
        source=args.canny_results/name
        meta=json.loads((source/'metrics.json').read_text())
        cf,cl=meta['context_first'],meta['context_last']
        y0,y1,x0,x1=meta['crop_box_y0_y1_x0_x1']
        raw=np.stack([load(args.input,i)[y0:y1,x0:x1] for i in range(cf,cl+1)])
        if first==500:
            initializer=args.thick_initializer/name/'material_mask.npz'
            if not initializer.exists():
                initializer=args.thick_initializer/name/'initializer_mask.npz'
            core_mask=np.load(initializer)['mask']
            provenance={'kind':'3D oriented-edge watershed without the rejected outer-support prior',
                        'initializer_sha256':hashlib.sha256(initializer.read_bytes()).hexdigest()}
        elif first==738:
            core_mask=[]
            for index in range(first,last+1):
                full=load(args.input,index);field=segment_slice(full)
                fy0,fy1,fx0,fx1=field['box']
                mask=np.zeros(full.shape,bool);mask[fy0:fy1,fx0:fx1]=field['watershed']
                core_mask.append(mask[y0:y1,x0:x1])
            core_mask=np.stack(core_mask)
            provenance={'kind':'previous slice-wise marker-controlled watershed; no threshold sweep'}
        else:
            core_mask=[];hashes={}
            for index in range(first,last+1):
                path=args.thin_initializer/f'material_{index:04d}.png'
                with Image.open(path) as image:mask=np.asarray(image)>0
                core_mask.append(mask[y0:y1,x0:x1]);hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            core_mask=np.stack(core_mask)
            provenance={'kind':'preserved previous successful joint 3D watershed masks',
                        'source_image_sha256':hashes}
        # Geometry outside the exported core is held constant only for storage;
        # derivatives always use the real CT context. Cut-plane vertices freeze.
        core=slice(first-cf,last-cf+1)
        mask=np.pad(core_mask,((core.start,len(raw)-core.stop),(0,0),(0,0)),mode='edge')
        roi=ndi.binary_dilation(ndi.binary_fill_holes(mask),iterations=8)
        smooth,g,mag,_=gradients(raw,'gaussian')
        edges=canny_variants(raw,smooth,g,mag,roi)
        print(name,'supported XYZ refinement',flush=True)
        base,moved,faces,accepted,shift,info=refine_mesh(mask,smooth,g,mag,edges['coherence'],core,edges['adaptive'])
        out=args.output/name;out.mkdir()
        save_ply(out/'surface_refined_xyz.ply',moved,faces,(y0,x0))
        np.savez_compressed(out/'surface_coordinates.npz',seed_zyx=base,refined_zyx=moved,
                            faces_zyx=faces,accepted=accepted,normal_shift_vox=shift)
        np.savez_compressed(out/'initializer_mask.npz',mask=core_mask,source_slices=np.arange(first,last+1),
                            crop_origin_yx=[y0,x0])
        index=(first+last)//2;z=index-first;rz=index-cf
        before=plane_segments(base,faces,float(z));after=plane_segments(moved,faces,float(z))
        fig,axes=plt.subplots(1,4,figsize=(13,10))
        for ax,title in zip(axes,['Original CT','Preserved initial surface','Adaptive 3D Canny','Canny-supported XYZ refinement']):
            ax.imshow(raw[rz],cmap='gray',vmin=.06,vmax=.8);ax.set_title(title,fontsize=9);ax.axis('off')
        axes[1].add_collection(LineCollection(before[:,:,[1,0]],colors='#00ff70',linewidths=.6))
        rgba=np.zeros((*raw[rz].shape,4));rgba[edges['adaptive'][rz]]=[0,1,.4,1];axes[2].imshow(rgba)
        axes[3].add_collection(LineCollection(after[:,:,[1,0]],colors='#00ff70',linewidths=.6))
        fig.suptitle(f'img_{index:04d}: initial topology retained; unsupported points remain unchanged')
        fig.tight_layout();fig.savefig(out/f'comparison_{index:04d}.png',dpi=145);plt.close(fig)
        # Zoom makes subvoxel position changes visible without implying a truth label.
        fig,ax=plt.subplots(figsize=(7,8));ax.imshow(raw[rz],cmap='gray',vmin=.06,vmax=.8)
        ax.add_collection(LineCollection(before[:,:,[1,0]],colors='#ff9d00',linewidths=1,label='Initial surface'))
        ax.add_collection(LineCollection(after[:,:,[1,0]],colors='#00ff70',linewidths=.7,label='XYZ refined'))
        ax.set_xlim(70,min(raw.shape[2]-1,300));ax.set_ylim(min(raw.shape[1]-1,600),180)
        ax.legend(loc='lower left');ax.set_title(f'img_{index:04d}: orange=initial, green=refined');ax.axis('off')
        fig.tight_layout();fig.savefig(out/f'position_detail_{index:04d}.png',dpi=145);plt.close(fig)
        with gzip.open(out/f'segments_{index:04d}.csv.gz','wt',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['source_file','slice','x0','y0','x1','y1'])
            for line in after:writer.writerow([f'img_{index:04d}.tif',index,line[0,1]+x0,line[0,0]+y0,line[1,1]+x0,line[1,0]+y0])
        corecounts={key:int(edges[key][core].sum()) for key in ['standard','adaptive','supported']}
        record={'first':first,'last':last,'center':index,'initializer':provenance,
                'crop_box_y0_y1_x0_x1':[y0,y1,x0,x1],'canny_core_counts':corecounts,
                'refinement':info,'mesh_vertices':len(base),'mesh_faces':len(faces)}
        (out/'metrics.json').write_text(json.dumps(record,indent=2)+'\n');summaries.append(record)
    (args.output/'metrics.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__=='__main__':main()
