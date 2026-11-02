#!/usr/bin/env python3
"""Edge-side 3D marker-controlled watershed, an experimental XCT comparison.

Gradient candidates supply both air and material markers, independently of a
global grey threshold. Measurements must be refined on the unmodified data.
This is an engineering adaptation, not an exact reproduction of a paper.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import re
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'ct-surface-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'ct-surface-cache'))
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True,exist_ok=True)
os.environ.setdefault('OMP_NUM_THREADS', '2')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure, segmentation
from skimage.morphology import h_minima, h_maxima

if (Path(__file__).parent/'previous/experiments').is_dir():
    sys.path.insert(0, str(Path(__file__).parent / 'previous/experiments'))
import boundary_study as old


def block(source, center, half=6):
    available=sorted(int(m.group(1)) for p in source.glob('img_*.tif')
                     if (m:=re.fullmatch(r'img_(\d+)\.tif',p.name)))
    if not available or center not in available:
        raise ValueError(f'No img_XXXX.tif data for requested centre {center}')
    ids = list(range(max(available[0], center-half), min(available[-1], center+half)+1))
    missing=set(ids)-set(available)
    if missing:
        raise ValueError(f'Missing real neighbouring slices: {sorted(missing)}')
    raw = np.stack([old.load(source, i) for i in ids])
    boxes = []
    for im in raw[::max(1,len(ids)//4)]:
        sm=ndi.gaussian_filter(im,1)
        boxes.append(old.bounding_box(old.largest(sm>filters.threshold_otsu(sm)), 40))
    box=(min(b[0] for b in boxes),max(b[1] for b in boxes),
         min(b[2] for b in boxes),max(b[3] for b in boxes))
    y0,y1,x0,x1=box
    return raw[:,y0:y1,x0:x1], ids, box


def envelope_for(raw):
    env=[]
    for im in raw:
        sm=ndi.gaussian_filter(im,1)
        init=old.largest(sm>filters.threshold_otsu(sm))
        # Envelope is a generous support only, not a measured boundary.
        e=ndi.binary_fill_holes(init)
        e=ndi.binary_dilation(e, iterations=14)
        env.append(e)
    env=np.stack(env)
    env=ndi.maximum_filter(env,size=(3,1,1))
    return env


def edge_markers(raw, sigma=.85, low=.010, high=.028, offset=2.0,
                 contrast=.035, seed_high=.60):
    smooth=ndi.gaussian_filter(raw,(sigma,sigma,sigma))
    grads=np.stack(np.gradient(smooth), axis=0)
    mag=np.sqrt(np.sum(grads*grads,axis=0))
    env=envelope_for(raw)
    cand=(mag>low)&env
    cand[0]=False; cand[-1]=False
    cand[:,[0,-1],:]=False; cand[:,:,[0,-1]]=False
    p=np.array(np.nonzero(cand)).T
    # Avoid allocating coordinates for the entire volume.
    norm=grads[:,p[:,0],p[:,1],p[:,2]].T
    norm/=np.maximum(np.linalg.norm(norm,axis=1,keepdims=True),1e-8)
    val=mag[tuple(p.T)]
    vm=ndi.map_coordinates(mag,(p-norm).T,order=1,mode='nearest')
    vp=ndi.map_coordinates(mag,(p+norm).T,order=1,mode='nearest')
    good=(val>=vm)&(val>=vp)
    nms=np.zeros(raw.shape,bool); nms[tuple(p[good].T)]=True
    connected=ndi.binary_propagation(nms&(mag>=high),mask=nms,
                                    structure=np.ones((3,3,3)))
    choose=connected[tuple(p.T)]&good
    p=p[choose]; norm=norm[choose]
    lo=ndi.map_coordinates(smooth,(p-offset*norm).T,order=1,mode='nearest')
    hi=ndi.map_coordinates(smooth,(p+offset*norm).T,order=1,mode='nearest')
    keep=(hi-lo)>contrast
    p=p[keep]; norm=norm[keep]
    air=np.rint(p-offset*norm).astype(int)
    material=np.rint(p+offset*norm).astype(int)
    shape=np.array(raw.shape)
    a=np.zeros(raw.shape,bool); m=np.zeros_like(a)
    for points,out in [(air,a),(material,m)]:
        valid=np.all((points>=0)&(points<shape),axis=1)
        out[tuple(points[valid].T)]=True
    conflicts=a&m
    a[conflicts]=False; m[conflicts]=False
    # Strong interiors may anchor broad homogeneous material; air never comes
    # from an inverted global Otsu mask, which destroyed weak walls previously.
    m|=ndi.binary_erosion((smooth>seed_high)&env,iterations=2)
    a|=~env
    m&=env; a[m]=False
    markers=np.zeros(raw.shape,np.uint8); markers[a]=1; markers[m]=2
    return smooth,mag,markers,connected,{
        'edge_candidates':len(p),'marker_conflicts':int(conflicts.sum()),
        'air_markers':int(a.sum()),'material_markers':int(m.sum()),
        'sigma':sigma,'low':low,'high':high,'offset':offset,'contrast':contrast}


def segment(raw, **kwargs):
    seed_mode=kwargs.pop('seed_mode','sides')
    sm,mag,markers,edges,meta=edge_markers(raw,**kwargs)
    if seed_mode=='extrema':
        env=envelope_for(raw)
        side_material=markers==2
        # A low gradient side is not necessarily air. For this alternative,
        # only persistent grey minima provide internal air markers.
        air=h_minima(sm,.04).astype(bool)
        air &= ndi.distance_transform_edt(env)>3
        span=ndi.maximum_filter(sm,size=(3,19,19))-sm
        air &= span>.08
        # Preserve maxima in weak walls even below the global Otsu threshold.
        material=h_maxima(sm,.03).astype(bool)&env&(sm>.20)
        ridge=sm-ndi.gaussian_filter(sm,(1.5,3,3))
        material|=(ridge>.018)&env&(sm>.25)
        lo=ndi.minimum_filter(sm,size=(3,25,25))
        hi=ndi.maximum_filter(sm,size=(3,25,25))
        material|=side_material&((sm-lo)>.60*(hi-lo))&((hi-lo)>.06)
        material|=ndi.binary_erosion((sm>.60)&env,iterations=2)
        material &= ~ndi.binary_dilation(air,iterations=2)
        markers=np.zeros(raw.shape,np.uint8)
        markers[~env|air]=1;markers[material]=2
        meta['seed_mode']='3D extrema + material ridges'
        meta['air_markers']=int((markers==1).sum())
        meta['material_markers']=int((markers==2).sum())
    mask=segmentation.watershed(mag,markers,connectivity=1)==2
    return mask,sm,markers,edges,meta


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--centers',default='500,625,750,875,988')
    ap.add_argument('--half',type=int,default=6)
    ap.add_argument('--sigma',type=float,default=.85)
    ap.add_argument('--low',type=float,default=.010)
    ap.add_argument('--high',type=float,default=.028)
    ap.add_argument('--offset',type=float,default=2)
    ap.add_argument('--contrast',type=float,default=.035)
    ap.add_argument('--seed-mode',choices=['sides','extrema'],default='sides')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    metrics=[]
    for center in map(int,args.centers.split(',')):
        start=time.monotonic()
        raw,ids,box=block(args.source,center,args.half)
        mask,sm,markers,edges,meta=segment(raw,sigma=args.sigma,low=args.low,
            high=args.high,offset=args.offset,contrast=args.contrast,seed_mode=args.seed_mode)
        z=ids.index(center)
        np.savez_compressed(args.out/f'block_{center:04d}.npz',mask=mask,
            source_slices=ids,box_y0_y1_x0_x1=box)
        y0,y1,x0,x1=box
        original=old.load(args.source,center)
        full=np.zeros(original.shape,np.uint8); full[y0:y1,x0:x1]=mask[z]*255
        Image.fromarray(full).save(args.out/f'material_{center:04d}.png')
        prior=old.segment_slice(original)
        refined=old.normal_refinement(prior['smooth'],prior['watershed'])
        previous=np.zeros(original.shape,bool)
        py0,py1,px0,px1=prior['box']
        previous[py0:py1,px0:px1]=old.rasterize(refined,prior['smooth'].shape)
        oldmask=previous[y0:y1,x0:x1]
        fig,axs=plt.subplots(1,4,figsize=(15,10))
        old.plot_mask(axs[0],raw[z],title='Original CT')
        old.plot_mask(axs[1],raw[z],mask=oldmask,title='Previous watershed + XY gradient')
        old.plot_mask(axs[2],raw[z],mask=mask[z],title=f'3D {args.seed_mode} markers + watershed')
        axs[3].imshow(raw[z],cmap='gray',vmin=.06,vmax=.8)
        axs[3].imshow(np.ma.masked_where(markers[z]==0,markers[z]),alpha=.6,
                      cmap='coolwarm',vmin=1,vmax=2)
        axs[3].axis('off'); axs[3].set_title('Markers: blue air, red material')
        fig.suptitle(f'img_{center:04d} | actual block {ids[0]}-{ids[-1]} | candidate boundaries, not calibrated measurements')
        fig.tight_layout(); fig.savefig(args.out/f'comparison_{center:04d}.png',dpi=140); plt.close(fig)
        record={'slice':center,'block':ids,'box':box,'shape':list(raw.shape),
                **meta,**old.diagnostics(mask[z]),'seconds':round(time.monotonic()-start,2)}
        metrics.append(record); print(json.dumps(record),flush=True)
        (args.out/'metrics.json').write_text(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
