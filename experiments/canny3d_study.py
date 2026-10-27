#!/usr/bin/env python3
"""True 3D adaptive Canny, oriented material/air seeds and surface refinement.

All intensity operations use original uint16 values scaled by 65535. Surface
coordinates remain in voxels; equal 75.4 um spacing is inherited, not calibrated.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure, segmentation
from boundary_study import load, largest, bounding_box, contours, plt
from matplotlib.collections import LineCollection


def diffusion(image, iterations=8):
    """Stable six-neighbour Perona-Malik filter with no-flux boundaries."""
    residual = image - ndi.gaussian_filter(image, .6)
    noise = float(np.median(abs(residual[::2, ::3, ::3]))) / .6745
    kappa = float(np.clip(4 * noise, .015, .08))
    result = image.copy()
    for _ in range(iterations):
        update = np.zeros_like(result)
        for axis in range(3):
            left, right = [slice(None)] * 3, [slice(None)] * 3
            left[axis], right[axis] = slice(None, -1), slice(1, None)
            left, right = tuple(left), tuple(right)
            delta = result[right] - result[left]
            flux = delta * np.exp(-(delta / kappa) ** 2)
            update[left] += flux
            update[right] -= flux
        result += .12 * update
    return result, {'diffusion_kappa': kappa, 'noise_mad': noise, 'iterations': iterations}


def gradients(image, mode='gaussian'):
    if mode == 'diffusion':
        source, info = diffusion(image)
        sigma = .7
    elif mode == 'gaussian':
        source, info, sigma = image, {}, 1.0
    else:
        raise ValueError(mode)
    smooth = ndi.gaussian_filter(source, sigma, mode='nearest')
    gradient = []
    for axis in range(3):
        order = [0, 0, 0]
        order[axis] = 1
        gradient.append(ndi.gaussian_filter(source, sigma, order=order, mode='nearest'))
    magnitude = np.sqrt(sum(g * g for g in gradient))
    return smooth, gradient, magnitude, info


def nms3d(magnitude, gradient):
    """Trilinear comparison at p +/- normalized(grad I), including Z."""
    result = np.zeros(magnitude.shape, bool)
    for start in range(1, magnitude.shape[0] - 1, 3):
        stop = min(start + 3, magnitude.shape[0] - 1)
        center = magnitude[start:stop]
        coordinates = np.indices(center.shape, dtype=np.float32)
        coordinates[0] += start
        direction = np.stack([g[start:stop] / np.maximum(center, 1e-8) for g in gradient])
        plus = ndi.map_coordinates(magnitude, coordinates + direction, order=1, mode='nearest')
        minus = ndi.map_coordinates(magnitude, coordinates - direction, order=1, mode='nearest')
        result[start:stop] = (center > 1e-7) & (center >= plus) & (center >= minus)
    result[:, :1] = False
    result[:, -1:] = False
    result[:, :, :1] = False
    result[:, :, -1:] = False
    return result


def local_high_threshold(raw, smooth, magnitude, nms, roi, tile=64):
    samples = magnitude[nms & roi]
    if len(samples) < 16:
        raise ValueError('Insufficient 3D NMS candidates')
    global_otsu = max(float(filters.threshold_otsu(samples)), 1e-5)
    global_standard = float(np.percentile(samples, 90))
    nz, ny, nx = raw.shape
    zs = list(range(0, nz, 4))
    if zs[-1] != nz - 1:
        zs.append(nz - 1)
    grid = np.empty((len(zs), (ny+tile-1)//tile, (nx+tile-1)//tile), np.float32)
    residual = raw - smooth
    for iz, z in enumerate(zs):
        for iy in range(grid.shape[1]):
            for ix in range(grid.shape[2]):
                area = (slice(max(0,z-2), min(nz,z+3)),
                        slice(max(0,iy*tile-12), min(ny,(iy+1)*tile+12)),
                        slice(max(0,ix*tile-12), min(nx,(ix+1)*tile+12)))
                values = magnitude[area][nms[area] & roi[area]]
                local = float(filters.threshold_otsu(values)) if len(values) >= 32 else global_otsu
                residual_samples = residual[area][:, ::3, ::3]
                median = np.median(residual_samples)
                noise = float(np.median(abs(residual_samples-median))) / .6745
                blended = .65 * local + .35 * global_otsu
                grid[iz,iy,ix] = max(float(np.clip(blended,.35*global_otsu,1.75*global_otsu)),
                                     .9 * noise)
    high = ndi.zoom(grid, np.array(raw.shape)/np.array(grid.shape), order=1)
    assert high.shape == raw.shape
    return high, {'global_otsu_high':global_otsu, 'standard_high_p90':global_standard,
                  'adaptive_high_min':float(high[roi].min()),
                  'adaptive_high_max':float(high[roi].max()), 'tile_xy':tile, 'tile_z':5}


def edge_quality(smooth, gradient, magnitude, candidates):
    mean_norm = ndi.gaussian_filter(magnitude, 1)
    coherence = np.sqrt(sum(ndi.gaussian_filter(g, 1)**2 for g in gradient))
    coherence /= np.maximum(mean_norm, 1e-7)
    contrast = np.zeros(smooth.shape, np.float32)
    points = np.argwhere(candidates).astype(np.float32)
    if len(points):
        indices = tuple(points.astype(np.int32).T)
        normal = np.stack([g[indices] for g in gradient],axis=1)
        normal /= np.maximum(magnitude[indices][:,None],1e-7)
        plus = ndi.map_coordinates(smooth, (points+2*normal).T, order=1, mode='nearest')
        minus = ndi.map_coordinates(smooth,(points-2*normal).T, order=1, mode='nearest')
        contrast[indices] = plus - minus
    return coherence, contrast


def hysteresis(weak, strong):
    return ndi.binary_propagation(strong, structure=np.ones((3,3,3)), mask=weak)


def canny_variants(raw, smooth, gradient, magnitude, roi):
    nms = nms3d(magnitude, gradient)
    high, info = local_high_threshold(raw,smooth,magnitude,nms,roi)
    base = nms & roi
    standard = hysteresis(base & (magnitude>=.4*info['standard_high_p90']),
                          base & (magnitude>=info['standard_high_p90']))
    adaptive = hysteresis(base & (magnitude>=.4*high),base & (magnitude>=high))
    coherence, contrast = edge_quality(smooth,gradient,magnitude,base & (magnitude>=.4*high))
    weak = base & (magnitude>=.4*high) & (coherence>.25) & (contrast>.0125)
    strong = weak & (magnitude>=high) & (coherence>.45) & (contrast>.025)
    supported = hysteresis(weak,strong)
    info.update({'nms_candidates':int(base.sum()),'standard_edges':int(standard.sum()),
                 'adaptive_edges':int(adaptive.sum()),'supported_edges':int(supported.sum())})
    return {'standard':standard,'adaptive':adaptive,'supported':supported,
            'high':high,'coherence':coherence,'contrast':contrast,'info':info}


def oriented_watershed(smooth, gradient, magnitude, edges, roi, outer_support=None):
    # Weak hysteresis continuations can guide a surface barrier but must not
    # independently seed an air pore or a material island inside CT texture.
    candidates = (edges['supported'] & (magnitude>=edges['high']) &
                  (edges['contrast']>.06) & (edges['coherence']>.65))
    integer = np.argwhere(candidates)
    if len(integer)<16:
        raise ValueError('Insufficient supported oriented edges')
    indices = tuple(integer.T)
    points = integer.astype(np.float32)
    normals = np.stack([g[indices] for g in gradient],axis=1)
    normals /= np.maximum(magnitude[indices][:,None],1e-7)
    strength = (edges['contrast'][indices] * edges['coherence'][indices]).astype(np.float32)
    votes = []
    for sign in [-1,1]:
        targets = np.rint(points+sign*1.8*normals).astype(np.int32)
        for axis in range(3):
            targets[:,axis] = np.clip(targets[:,axis],0,smooth.shape[axis]-1)
        vote = np.zeros(smooth.shape,np.float32)
        flat = np.ravel_multi_index(tuple(targets.T),smooth.shape)
        np.maximum.at(vote.ravel(),flat,strength)
        votes.append(vote)
    air, material = votes
    markers = np.zeros(smooth.shape,np.uint8)
    markers[(air>material*1.2)&(air>0)] = 1
    markers[(material>air*1.2)&(material>0)] = 2
    outer_seed_count=0
    if outer_support is not None:
        # A weak wall may be darker than a bright CT reconstruction streak
        # in its air channel. Gradient sign alone then mislabels the wall.
        # A narrow material seed inside the independently estimated outside
        # contour stabilizes that classification; holes are not filled in
        # the resulting segmentation.
        for z in range(len(markers)):
            distance=ndi.distance_transform_edt(outer_support[z])
            skin=(distance>=1.5)&(distance<=2.5)
            outer_seed_count+=int(skin.sum())
            markers[z,skin]=2
    # Only the spatial ROI boundary supplies additional air markers. No
    # intensity threshold fixes the inside of a bright channel as material.
    rim = roi & ~ndi.binary_erosion(roi, structure=np.ones((1,3,3)))
    markers[rim] = 1
    elevation = magnitude * (1 + 1.5*edges['supported'])
    print('  oriented seeds:',int((markers==1).sum()),'air,',int((markers==2).sum()),'material',flush=True)
    labels = segmentation.watershed(elevation,markers,mask=roi)
    foreground = labels==2
    components,n = ndi.label(foreground,structure=np.ones((3,3,3)))
    counts = np.bincount(components.ravel());counts[0]=0
    if not counts.any():
        raise ValueError('No material component')
    kept = components==counts.argmax()
    return kept, markers, {'components_before_cleanup':int(n),'outer_support_seed_voxels':outer_seed_count,
                           'removed_material_voxels':int(foreground.sum()-kept.sum()),
                           'retention_assumption':'one connected scanned part in the selected block'}


def refine_mesh(mask, smooth, gradient, magnitude, coherence, core, canny_support=None):
    """Refine the marching-cubes vertices in XYZ, with cut planes frozen."""
    vertices,faces,_,_ = measure.marching_cubes(mask[core].astype(np.float32),.5,
                                              allow_degenerate=False)
    points = vertices.copy();points[:,0] += core.start
    proposed = points.copy()
    accepted = np.zeros(len(points),bool)
    shifts = np.zeros(len(points),np.float32)
    edge_distance=ndi.distance_transform_edt(~canny_support) if canny_support is not None else None
    unsupported_count=0
    ts = np.arange(-2,2.001,.25,dtype=np.float32)
    for start in range(0,len(points),8192):
        stop=min(start+8192,len(points));p=points[start:stop]
        normal=np.stack([ndi.map_coordinates(g,p.T,order=1,mode='nearest') for g in gradient],axis=1)
        norm=np.linalg.norm(normal,axis=1)
        normal/=np.maximum(norm[:,None],1e-7)
        xyz=p[:,None,:]+ts[None,:,None]*normal[:,None,:]
        profile=ndi.map_coordinates(smooth,xyz.reshape(-1,3).T,order=1,mode='nearest').reshape(len(p),-1)
        derivative=np.zeros(profile.shape,np.float32)
        for axis,g in enumerate(gradient):
            sampled=ndi.map_coordinates(g,xyz.reshape(-1,3).T,order=1,mode='nearest').reshape(len(p),-1)
            derivative+=sampled*normal[:,axis,None]
        k=derivative.argmax(axis=1);r=np.arange(len(p))
        k0=np.clip(k,1,len(ts)-2)
        prev,peak,nxt=derivative[r,k0-1],derivative[r,k0],derivative[r,k0+1]
        denom=prev-2*peak+nxt
        fraction=np.divide(.5*(prev-nxt),denom,out=np.zeros_like(denom),where=abs(denom)>1e-8)
        shift=ts[k0]+.25*np.clip(fraction,-.5,.5)
        contrast=profile[:,ts==2].ravel()-profile[:,ts==-2].ravel()
        coh=ndi.map_coordinates(coherence,p.T,order=1,mode='nearest')
        good=(k>0)&(k<len(ts)-1)&(contrast>.025)&(coh>.35)&(norm>.001)
        if edge_distance is not None:
            supported=ndi.map_coordinates(edge_distance,p.T,order=1,mode='nearest')<=1.5
            unsupported_count+=int((~supported).sum())
            good &= supported
        cut=np.isclose(vertices[start:stop,0],0)|np.isclose(vertices[start:stop,0],mask[core].shape[0]-1)
        good &= ~cut
        shifts[start:stop]=np.where(good,shift,0)
        accepted[start:stop]=good
        proposed[start:stop]=p+normal*np.where(good,shift,0)[:,None]
    # Prevent flips/collapses introduced by independent vertex displacement.
    tri=points[faces]
    original_cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    backtracked=np.zeros(len(points),bool)
    for _ in range(40):
        tri=proposed[faces]
        new_cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        bad=(np.sum(original_cross*new_cross,axis=1)<=0)|(np.linalg.norm(new_cross,axis=1)<1e-7)
        if not bad.any():break
        ids=np.unique(faces[bad]);backtracked[ids]=True
        # Local line search approaches the known-valid seed mesh continuously.
        # A clipped vertex is flagged as NOT reaching the gradient maximum.
        proposed[ids]=.5*(proposed[ids]+points[ids])
        accepted[ids]=False;shifts[ids]*=.5
    if bad.any():
        raise ValueError('Mesh refinement leaves inverted faces')
    proposed[:,0]-=core.start
    return vertices,proposed,faces,accepted,shifts,{'geometry_backtracked_vertices':int(backtracked.sum()),
        'vertices_without_nearby_canny':unsupported_count,
        'accepted_vertex_fraction':float(accepted.mean()),
        'abs_shift_p95_vox':float(np.percentile(abs(shifts),95)),
        'abs_z_displacement_p95_vox':float(np.percentile(abs(proposed[:,0]-vertices[:,0]),95))}


def plane_segments(vertices,faces,z):
    tri=vertices[faces]
    selected=(tri[:,:,0].min(axis=1)<=z)&(tri[:,:,0].max(axis=1)>=z)
    tri=tri[selected]
    if not len(tri):return np.empty((0,2,2),np.float32)
    hits=np.zeros((len(tri),3,2),np.float32)
    valid=np.zeros((len(tri),3),bool)
    for i,(a,b) in enumerate([(0,1),(1,2),(2,0)]):
        p,q=tri[:,a],tri[:,b]
        ok=(((p[:,0]<=z)&(q[:,0]>=z))|((q[:,0]<=z)&(p[:,0]>=z))) & (abs(q[:,0]-p[:,0])>1e-8)
        t=np.divide(z-p[:,0],q[:,0]-p[:,0],out=np.zeros(len(p),np.float32),where=abs(q[:,0]-p[:,0])>1e-8)
        hits[:,i]=p[:,1:]+t[:,None]*(q[:,1:]-p[:,1:])
        valid[:,i]=ok
    # At a vertex exactly on the plane, two edge hits may coincide. Keep
    # the farthest valid pair instead of dropping the entire triangle.
    pairs=np.array([[0,1],[0,2],[1,2]])
    distance=np.stack([np.sum((hits[:,a]-hits[:,b])**2,axis=1)
                       * (valid[:,a]&valid[:,b]) for a,b in pairs],axis=1)
    chosen=distance.argmax(axis=1)
    selected=distance.max(axis=1)>1e-10
    rows=np.flatnonzero(selected)
    pair=pairs[chosen[selected]]
    return hits[rows[:,None],pair]


def save_ply(path,vertices,faces,origin):
    xyz=np.ascontiguousarray(vertices[:,[2,1,0]],dtype='<f4')
    xyz[:,0]+=origin[1];xyz[:,1]+=origin[0]
    # Reversing ZYX to XYZ reverses handedness; restore face winding.
    faces=faces[:,[0,2,1]]
    header=('ply\nformat binary_little_endian 1.0\ncomment units=voxels; open Z cut planes\n'
            f'element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n'
            f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
    records=np.empty(len(faces),dtype=[('n','u1'),('indices','<i4',(3,))])
    records['n']=3;records['indices']=faces
    with path.open('wb') as stream:
        stream.write(header.encode());stream.write(xyz.tobytes());stream.write(records.tobytes())


def run_block(args,first,last):
    output=args.output/f'block_{first:04d}_{last:04d}'
    output.mkdir(parents=True,exist_ok=False)
    context_first=max(500,first-4);context_last=min(1000,last+4)
    raw_full=np.stack([load(args.input,i) for i in range(context_first,context_last+1)])
    support=[]
    for raw in raw_full:
        sm=ndi.gaussian_filter(raw,1)
        support.append(ndi.binary_fill_holes(largest(sm>filters.threshold_otsu(sm))))
    support=np.stack(support)
    box=bounding_box(support.any(axis=0),padding=24)
    y0,y1,x0,x1=box
    raw=raw_full[:,y0:y1,x0:x1].copy()
    envelope=support[:,y0:y1,x0:x1]
    roi=ndi.maximum_filter(envelope,size=(1,25,25))
    del support
    core=slice(first-context_first,last-context_first+1)
    meta={'source_first':first,'source_last':last,'context_first':context_first,
          'context_last':context_last,'crop_box_y0_y1_x0_x1':box,
          'assumed_isotropic_spacing_um':75.4,'original_uint16_scaling':65535,
          'methods':{},'notes':['Mesh coordinates in voxels; accuracy on real CT requires reference contours.']}
    gaussian_saved=None
    for mode in ['gaussian','diffusion']:
        started=perf_counter();print(f'{first}..{last}: {mode} 3D gradient and adaptive Canny',flush=True)
        smooth,g,mag,filter_info=gradients(raw,mode)
        edges=canny_variants(raw,smooth,g,mag,roi)
        info={**filter_info,**edges['info']}
        masks={name:edges[name][core].copy() for name in ['standard','adaptive','supported']}
        for name,mask in masks.items():
            for z in range(len(mask)):
                Image.fromarray(mask[z].astype(np.uint8)*255).save(output/f'{mode}_{name}_{first+z:04d}.png')
        if mode=='gaussian':
            gaussian_saved=masks
        else:
            mask,markers,seg_info=oriented_watershed(smooth,g,mag,edges,roi,
                                                   None if args.no_outer_support else envelope)
            base,refined,faces,accepted,shifts,refine_info=refine_mesh(mask,smooth,g,mag,edges['coherence'],core)
            meta['segmentation']={**seg_info,**refine_info,'mesh_vertices':len(base),'mesh_faces':len(faces)}
            save_ply(output/'surface_refined_xyz.ply',refined,faces,(y0,x0))
            np.savez_compressed(output/'surface_coordinates.npz',seed_zyx=base,refined_zyx=refined,
                                faces_zyx=faces,accepted=accepted,normal_shift_vox=shifts,
                                source_first=first,crop_origin_yx=[y0,x0])
            np.savez_compressed(output/'material_mask.npz',mask=mask[core],source_slices=np.arange(first,last+1),
                                crop_origin_yx=[y0,x0])
            center=(first+last)//2
            chosen=sorted({first,last,center})
            if first<=988<=last:chosen.append(988)
            for index in sorted(set(chosen)):
                z=index-first;context_z=index-context_first
                original=raw[context_z]
                figure,axes=plt.subplots(1,5,figsize=(16,10))
                entries=[('Gaussian: global 3D Canny',gaussian_saved['standard'][z]),
                         ('Gaussian: adaptive 3D Canny',gaussian_saved['adaptive'][z]),
                         ('Diffusion: adaptive 3D Canny',masks['adaptive'][z]),
                         ('Contrast-supported adaptive Canny',masks['supported'][z])]
                for ax,(title,edge) in zip(axes,entries):
                    ax.imshow(original,cmap='gray',vmin=.06,vmax=.8)
                    rgba=np.zeros((*edge.shape,4));rgba[edge]=[0,1,.4,1]
                    ax.imshow(rgba);ax.set_title(title,fontsize=9);ax.axis('off')
                ax=axes[-1];ax.imshow(original,cmap='gray',vmin=.06,vmax=.8)
                segments=plane_segments(refined,faces,float(z))
                if len(segments):ax.add_collection(LineCollection(segments[:,:,[1,0]],colors='#00ff70',linewidths=.65))
                ax.set_title('Oriented seeds + XYZ refinement',fontsize=9);ax.axis('off')
                figure.suptitle(f'Original img_{index:04d}; genuine 3D derivatives / NMS / hysteresis')
                figure.tight_layout();figure.savefig(output/f'comparison_{index:04d}.png',dpi=140);plt.close(figure)
                fig,ax=plt.subplots(figsize=(5,10))
                ax.imshow(original,cmap='gray',vmin=.06,vmax=.8)
                if len(segments):ax.add_collection(LineCollection(segments[:,:,[1,0]],colors='#00ff70',linewidths=.6))
                ax.set_title(f'img_{index:04d}: refined 3D surface');ax.axis('off')
                fig.tight_layout();fig.savefig(output/f'refined_{index:04d}.png',dpi=140);plt.close(fig)
                with gzip.open(output/f'surface_segments_{index:04d}.csv.gz','wt',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(['source_file','slice','x0','y0','x1','y1'])
                    for segment in segments:
                        writer.writerow([f'img_{index:04d}.tif',index,segment[0,1]+x0,segment[0,0]+y0,
                                         segment[1,1]+x0,segment[1,0]+y0])
            for z in range(last-first+1):
                Image.fromarray(mask[core][z].astype(np.uint8)*255).save(output/f'material_{first+z:04d}.png')
                # Scientific diagnostics: compare local threshold and actual Gz.
            fig,axes=plt.subplots(1,3,figsize=(10,9))
            for ax,image,title in zip(axes,[abs(g[0][center-context_first]),mag[center-context_first],edges['high'][center-context_first]],
                                      ['Absolute Z derivative','3D gradient magnitude','Adaptive high threshold']):
                artist=ax.imshow(image,cmap='magma',vmin=0,vmax=np.percentile(image,99.5));ax.set_title(title);ax.axis('off')
                fig.colorbar(artist,ax=ax,shrink=.5)
            fig.tight_layout();fig.savefig(output/'gradient_and_threshold_maps.png',dpi=130);plt.close(fig)
        info['seconds']=perf_counter()-started;meta['methods'][mode]=info
        del smooth,g,mag,edges,masks;gc.collect()
        print('  completed in',round(info['seconds'],1),'seconds',flush=True)
    (output/'metrics.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Saved',output,flush=True)
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('CT_Test'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--blocks',default='976:1000')
    parser.add_argument('--no-outer-support',action='store_true',
                        help='Disable the experimental narrow outer-material prior')
    parser.add_argument('--outer-support',dest='no_outer_support',action='store_false',
                        help='Reproduce the rejected outer-prior diagnostic variant')
    parser.set_defaults(no_outer_support=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for spec in args.blocks.split(','):
        first,last=map(int,spec.split(':'))
        if not 500<=first<last<=1000:parser.error('Blocks must lie within source range 500..1000')
        records.append(run_block(args,first,last))
    (args.output/'all_metrics.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':
    main()
