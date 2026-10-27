"""Export controlled comparisons of region fitting, Hessian and PSF methods."""
import argparse
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import filters
from boundary_study import load,plt,contours
from matplotlib.collections import LineCollection
from canny3d_study import plane_segments,save_ply
from metrology_study import fit_surface,recover_thin_walls


def geometry(vertices,moved,faces):
    tri=vertices[faces]
    c0=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    tri=moved[faces];c1=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    ends=moved[unique[counts==1],0]
    boundary_ok=((abs(ends)<1e-4).all(axis=1)|
                 (abs(ends-vertices[:,0].max())<1e-4).all(axis=1))
    info={'finite':bool(np.isfinite(moved).all()),'non_manifold_edges':int((counts>2).sum()),
          'boundary_edges_away_from_cuts':int((~boundary_ok).sum()),
          'degenerate_faces':int((np.linalg.norm(c1,axis=1)<1e-7).sum()),
          'flipped_faces':int((np.sum(c0*c1,axis=1)<=0).sum()),
          'max_displacement_vox':float(np.linalg.norm(moved-vertices,axis=1).max()),
          'global_triangle_self_intersections_checked':False}
    info['passed']=info['finite'] and all(info[x]==0 for x in [
        'non_manifold_edges','boundary_edges_away_from_cuts','degenerate_faces','flipped_faces'])
    return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('CT_Test'))
    parser.add_argument('--regions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for first,last in [(500,524),(738,762),(976,1000)]:
        block=f'block_{first:04d}_{last:04d}'
        ref=Path('results/canny3d_v2/supported_refinement')/block
        meta=json.loads((ref/'metrics.json').read_text())
        y0,y1,x0,x1=meta['crop_box_y0_y1_x0_x1']
        cf,cl=max(500,first-8),min(1000,last+8)
        raw=np.stack([load(args.input,i)[y0:y1,x0:x1] for i in range(cf,cl+1)])
        core=slice(first-cf,last-cf+1)
        original=np.load(ref/'initializer_mask.npz')['mask']
        initial=np.pad(original,((core.start,len(raw)-core.stop),(0,0),(0,0)),mode='edge')
        threshold=float(filters.threshold_otsu(raw))
        initial[:core.start]=raw[:core.start]>threshold
        initial[core.stop:]=raw[core.stop:]>threshold
        candidates={'preserved_topology':initial}
        out=args.output/block;out.mkdir()
        if first==738:
            for scale in [6,12]:
                p=args.regions/block/f'local_rsf_{scale}.npz'
                data=np.load(p)
                assert np.array_equal(data['source_slices'],np.arange(first,last+1))
                assert np.array_equal(data['crop_origin_yx'],[y0,x0])
                target=initial.copy();target[core]=data['mask']
                candidates[f'local_rsf_{scale}']=target
                shutil.copyfile(p,out/p.name)
            base=candidates['local_rsf_6']
            print(block,'Hessian + paired-profile wall proposals',flush=True)
            recovered,support,info=recover_thin_walls(raw,base)
            candidates['profile_wall_proposals']=recovered
            np.savez_compressed(out/'profile_wall_proposals.npz',mask=recovered[core],
                added=(recovered&~base)[core],crop_origin_yx=[y0,x0],source_slices=np.arange(first,last+1))
            (out/'recovery_metrics.json').write_text(json.dumps(info,indent=2)+'\n')
            shutil.copyfile(args.regions/block/'metrics.json',out/'region_metrics.json')
        old=np.load(ref/'surface_coordinates.npz')
        old_vertices,old_faces=old['refined_zyx'],old['faces_zyx']
        models={}
        # Scale 12 is a parameter sensitivity control; no redundant mesh fit.
        for name,mask in candidates.items():
            if name=='local_rsf_12':continue
            print(block,name,'raw CT PSF surface fit',flush=True)
            result=fit_surface(raw,mask,core)
            base,moved,faces,accepted,shift,info,quality=result
            check=geometry(base,moved,faces)
            path=out/name;path.mkdir()
            np.savez_compressed(path/'surface_coordinates.npz',seed_zyx=base,refined_zyx=moved,
                faces_zyx=faces,accepted=accepted,normal_shift_vox=shift,
                crop_origin_yx=[y0,x0],source_first=first,voxel_size_um=[75.4]*3,**quality)
            if check['passed']:
                save_ply(path/'surface_voxels.ply',moved,faces,(y0,x0))
                # Millimetres; origin at image pixel (0,0), source slice 500.
                physical=moved.copy();physical[:,0]+=first-500
                physical[:,1]+=y0;physical[:,2]+=x0;physical*=.0754
                save_ply(path/'surface_mm.ply',physical,faces,(0,0))
                ply=path/'surface_mm.ply'
                content=ply.read_bytes().replace(b'comment units=voxels; open Z cut planes',
                    b'comment units=mm; z=0 at source slice 500; open Z cut planes')
                ply.write_bytes(content)
            info.update({'geometry':check,'first':first,'last':last,'method':name,
                'crop_box_y0_y1_x0_x1':[y0,y1,x0,x1],
                'context_first':cf,'context_last':cl,
                'warning':'No real CT accuracy score: reference STL absent. Profile proposals are experimental.'})
            (path/'metrics.json').write_text(json.dumps(info,indent=2)+'\n')
            models[name]=(moved,faces,accepted)
            records.append(info);print(info,flush=True)
        indices=[(first+last)//2] if first!=738 else [738,744,750,756,762]
        for index in indices:
            z=index-first;rz=index-cf
            entries=[('Previous adaptive Canny + XYZ',plane_segments(old_vertices,old_faces,z))]
            for name,(v,f,a) in models.items():
                entries.append((name+' + PSF',plane_segments(v,f,z)))
            fig,axes=plt.subplots(1,len(entries)+1,figsize=(4*(len(entries)+1),12))
            for ax in axes:ax.imshow(raw[rz],cmap='gray',vmin=.06,vmax=.8);ax.axis('off')
            axes[0].set_title('Original CT',fontsize=10)
            for ax,(title,lines) in zip(axes[1:],entries):
                ax.set_title(title,fontsize=9)
                ax.add_collection(LineCollection(lines[:,:,[1,0]],colors='#00ff70',linewidths=.6))
            fig.tight_layout();fig.savefig(out/f'comparison_{index:04d}.png',dpi=145);plt.close(fig)
            if first==738:
                fig,axes=plt.subplots(1,3,figsize=(15,9))
                for ax in axes:
                    ax.imshow(raw[rz],cmap='gray',vmin=.06,vmax=.8);ax.axis('off')
                    ax.set_xlim(110,360);ax.set_ylim(480,50)
                axes[0].set_title('Original CT: weak walls')
                for ax,key in zip(axes[1:],['preserved_topology','profile_wall_proposals']):
                    v,f,a=models[key]
                    good_faces=a[f].all(axis=1)
                    good=plane_segments(v,f[good_faces],z);weak=plane_segments(v,f[~good_faces],z)
                    ax.add_collection(LineCollection(good[:,:,[1,0]],colors='#00ff70',linewidths=.7))
                    ax.add_collection(LineCollection(weak[:,:,[1,0]],colors='#ff9d00',linewidths=.7))
                    ax.set_title(key+'\ngreen=fit passed; orange=unconfirmed',fontsize=9)
                fig.tight_layout();fig.savefig(out/f'weak_walls_{index:04d}.png',dpi=160);plt.close(fig)
            for name,(v,f,a) in models.items():
                segments=plane_segments(v,f,z)
                with gzip.open(out/name/f'contours_{index:04d}.csv.gz','wt',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(['source_file','x0_px','y0_px','x1_px','y1_px'])
                    for line in segments:writer.writerow([f'img_{index:04d}.tif',line[0,1]+x0,line[0,0]+y0,line[1,1]+x0,line[1,0]+y0])
    (args.output/'metrics.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
