#!/usr/bin/env python3
"""Reproducible CT surface experiments; writes only to --output.

Input filenames remain the source of slice identifiers. No padding, index
remapping, hole filling in the result, or modification of the CT data is used.
Metrics are diagnostics, not accuracy scores against unknown ground truth.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ct-boundary-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/ct-boundary-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import filters, measure, segmentation
from skimage.draw import polygon
from skimage.morphology import h_minima


def load(source: Path, index: int) -> np.ndarray:
    with Image.open(source / f'img_{index:04d}.tif') as image:
        values = np.asarray(image)
    if values.dtype != np.uint16 or values.ndim != 2:
        raise ValueError('Expected single-channel uint16 TIFF')
    return values.astype(np.float32) / 65535.0


def largest(mask):
    labels, n = ndi.label(mask)
    if not n:
        raise ValueError('No material candidate found')
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == counts.argmax()


def bounding_box(mask, padding=24):
    y, x = np.where(mask)
    return (max(0, int(y.min()) - padding), min(mask.shape[0], int(y.max()) + padding + 1),
            max(0, int(x.min()) - padding), min(mask.shape[1], int(x.max()) + padding + 1))


def markers_for(smooth, initial, high=None):
    # Low threshold supplies air seeds in the holes. High threshold supplies
    # material seeds in the ribs. Intermediate intensities remain UNLABELLED.
    # Fixing all Otsu-positive pixels as material incorrectly fills channels.
    if high is None:
        high = float(filters.threshold_multiotsu(smooth, classes=3)[-1])
    markers = np.zeros(initial.shape, np.uint8)
    markers[ndi.distance_transform_edt(~initial) > 2] = 1
    envelope = ndi.binary_fill_holes(initial)  # support ONLY, never final mask
    material = ndi.binary_erosion((smooth > high) & envelope, iterations=1)
    markers[material] = 2
    if not np.any(markers == 2):
        raise ValueError('No confident material seed')
    return markers, high


def segment_slice(raw, sigma=1.0, run_watershed=True):
    smooth = ndi.gaussian_filter(raw, sigma)
    low = float(filters.threshold_otsu(smooth))
    initial = largest(smooth > low)
    box = bounding_box(initial)
    y0, y1, x0, x1 = box
    s = smooth[y0:y1, x0:x1]
    initial = initial[y0:y1, x0:x1]
    markers, high = markers_for(s, initial)
    elevation = filters.sobel(s)
    ws = segmentation.watershed(elevation, markers) == 2 if run_watershed else None
    return {'raw': raw[y0:y1, x0:x1], 'smooth': s, 'box': box,
            'initial': initial, 'markers': markers, 'low': low, 'high': high,
            'watershed': ws, 'elevation': elevation}


def extrema_markers(field):
    """Alternative for bright channels missed by the global low threshold.

    This is deliberately compared separately: on thin walls, extrema-derived
    air seeds can be wrong. Do not silently substitute this for the default.
    """
    smooth, initial = field['smooth'], field['initial']
    envelope = ndi.binary_fill_holes(initial)
    air = h_minima(smooth, .04).astype(bool)
    air &= ndi.distance_transform_edt(envelope) > 3
    plateau = float(np.percentile(smooth[smooth > field['high']], 80))
    ridge = smooth - ndi.gaussian_filter(smooth, 3)
    material = ((smooth > plateau) | ((smooth > field['low']) & (ridge > .025))) & envelope
    markers = np.zeros(smooth.shape, np.uint8)
    markers[ndi.distance_transform_edt(~envelope) > 2] = 1
    markers[air] = 1
    markers[ndi.binary_erosion(material, iterations=1)] = 2
    return markers


def contours(mask):
    return measure.find_contours(mask.astype(np.float32), 0.5,
                                 fully_connected='high', positive_orientation='low')


def normal_refinement(smooth, mask, radius=3.0):
    """Compare local material/air mid-level and maximum positive gradient.

    All displacements are bounded. Weak/ambiguous profiles keep their seed
    position and are explicitly flagged. Coordinates are interpolated, which
    does not imply metrological subvoxel accuracy.
    """
    signed = ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)
    ny = ndi.gaussian_filter(signed, 1.2, order=(1, 0))
    nx = ndi.gaussian_filter(signed, 1.2, order=(0, 1))
    ts = np.arange(-6, 6.001, .25, dtype=np.float32)
    results = []
    for cid, pts in enumerate(contours(mask)):
        if not np.allclose(pts[0], pts[-1]):
            raise ValueError('Contour touches ROI border; increase padding')
        # Remove duplicated endpoint for circular smoothing, then restore it.
        pts = pts[:-1]
        norms = np.stack([ndi.map_coordinates(field, pts.T, order=1)
                          for field in (ny, nx)], axis=1)
        norms /= np.maximum(np.linalg.norm(norms, axis=1, keepdims=True), 1e-6)
        coords = pts[:, None, :] + norms[:, None, :] * ts[None, :, None]
        profiles = ndi.map_coordinates(smooth, coords.reshape(-1, 2).T,
                                       order=1, mode='nearest').reshape(len(pts), -1)
        air = np.median(profiles[:, ts <= -3], axis=1)
        material = np.median(profiles[:, ts >= 3], axis=1)
        contrast = material - air
        derivative = np.gradient(profiles, .25, axis=1)
        search = np.flatnonzero(abs(ts) <= radius)
        k = search[np.argmax(derivative[:, search], axis=1)]
        row = np.arange(len(pts))
        prev = derivative[row, np.maximum(k - 1, 0)]
        peak = derivative[row, k]
        nxt = derivative[row, np.minimum(k + 1, len(ts) - 1)]
        denom = prev - 2 * peak + nxt
        fractional = np.divide(.5 * (prev - nxt), denom,
                               out=np.zeros_like(denom), where=abs(denom) > 1e-8)
        grad_shift = ts[k] + np.clip(fractional, -.5, .5) * .25
        valid_grad = ((contrast > .02) & (k > search[0]) & (k < search[-1]) &
                      (peak > .003))
        target = (material + air) / 2
        crossing = ((profiles[:, :-1] <= target[:, None]) &
                    (profiles[:, 1:] > target[:, None]) &
                    (abs(ts[:-1])[None, :] <= radius))
        distance = abs(ts[:-1][None, :] - grad_shift[:, None])
        choice = np.argmin(np.where(crossing, distance, np.inf), axis=1)
        before = profiles[row, choice]
        after = profiles[row, choice + 1]
        fraction = (target - before) / np.maximum(after - before, 1e-8)
        iso_shift = ts[choice] + .25 * np.clip(fraction, 0, 1)
        valid_iso = valid_grad & crossing.any(axis=1) & (abs(iso_shift) <= radius)
        refined = {}
        for name, shift, valid in [('gradient', grad_shift, valid_grad),
                                   ('local50', iso_shift, valid_iso)]:
            shift = np.where(valid, shift, 0)
            if len(shift) >= 8:
                shift = ndi.gaussian_filter1d(shift, .75, mode='wrap')
            shift = np.where(valid, shift, 0)
            moved = pts + shift[:, None] * norms
            refined[name] = np.vstack([moved, moved[0]])
            refined[name + '_shift'] = shift
            refined[name + '_valid'] = valid
        area = np.sum(pts[:, 1] * np.roll(pts[:, 0], -1) -
                      np.roll(pts[:, 1], -1) * pts[:, 0]) / 2
        results.append({'id': cid, 'seed': np.vstack([pts, pts[0]]),
                        'area': area, 'contrast': contrast, **refined})
    # skimage's orientation consistently distinguishes holes from material.
    outer_sign = np.sign(max(results, key=lambda c: abs(c['area']))['area'])
    for item in results:
        item['kind'] = 'outer' if np.sign(item['area']) == outer_sign else 'inner'
    return results


def rasterize(refined, shape, method='gradient'):
    mask = np.zeros(shape, bool)
    # Even/odd polygon fill preserves internal air cavities.
    for item in refined:
        pts = item[method]
        yy, xx = polygon(pts[:, 0], pts[:, 1], shape)
        mask[yy, xx] ^= True
    return mask


def diagnostics(mask):
    labels, components = ndi.label(mask)
    holes = ndi.binary_fill_holes(mask) & ~mask
    ids, _ = ndi.label(holes)
    counts = np.bincount(ids.ravel())[1:]
    return {'material_pixels': int(mask.sum()), 'components': int(components),
            'holes_ge_25px': int(np.count_nonzero(counts >= 25)),
            'hole_area_pixels': int(holes.sum())}


def plot_mask(ax, raw, mask=None, lines=None, title=''):
    ax.imshow(raw, cmap='gray', vmin=.06, vmax=.80)
    if mask is not None:
        lines = contours(mask)
    for line in lines or []:
        ax.plot(line[:, 1], line[:, 0], color='#00ff70', lw=.6)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def write_points(path, index, items, box):
    y0, _, x0, _ = box
    with gzip.open(path, 'wt', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_file', 'slice_number', 'contour_id', 'boundary_kind',
                         'x_px', 'y_px', 'shift_px', 'local_contrast_u16',
                         'gradient_supported', 'local50_x_px', 'local50_y_px',
                         'local50_supported'])
        for item in items:
            for j, point in enumerate(item['gradient'][:-1]):
                iso = item['local50'][j]
                writer.writerow([f'img_{index:04d}.tif', index, item['id'], item['kind'],
                                 f'{point[1]+x0:.5f}', f'{point[0]+y0:.5f}',
                                 f'{item["gradient_shift"][j]:.5f}',
                                 f'{item["contrast"][j]*65535:.2f}',
                                 int(item['gradient_valid'][j]),
                                 f'{iso[1]+x0:.5f}', f'{iso[0]+y0:.5f}',
                                 int(item['local50_valid'][j])])


def save_slice(output, index, raw, field, items):
    output.mkdir(parents=True, exist_ok=True)
    y0, y1, x0, x1 = field['box']
    mask = np.zeros(raw.shape, np.uint8)
    mask[y0:y1, x0:x1] = rasterize(items, field['smooth'].shape) * 255
    Image.fromarray(mask).save(output / f'material_{index:04d}.png')
    write_points(output / f'contours_{index:04d}.csv.gz', index, items, field['box'])
    fig, ax = plt.subplots(figsize=(5, 10))
    plot_mask(ax, field['raw'], lines=[item['gradient'] for item in items],
              title=f'img_{index:04d}: all material-air boundaries')
    for item in items:
        bad = ~item['gradient_valid']
        if np.any(bad):
            points = item['gradient'][:-1][bad]
            ax.scatter(points[:, 1], points[:, 0], s=1, color='orange')
    fig.tight_layout()
    fig.savefig(output / f'overlay_{index:04d}.png', dpi=145)
    plt.close(fig)
    return mask


def study(args):
    records = []
    for index in args.study_slices:
        started = perf_counter()
        raw = load(args.input, index)
        field = segment_slice(raw)
        markers = field['markers']
        rw = segmentation.random_walker(field['smooth'], markers, beta=200,
                                         mode='cg_j', tol=1e-5) == 2
        refined = normal_refinement(field['smooth'], field['watershed'])
        local50 = rasterize(refined, field['smooth'].shape, 'local50')
        final = rasterize(refined, field['smooth'].shape, 'gradient')
        alternate = segmentation.watershed(field['elevation'], extrema_markers(field)) == 2
        alt_refined = normal_refinement(field['smooth'], alternate)
        variants = [('Global Otsu', field['initial']),
                    ('ROI Otsu', field['smooth'] > filters.threshold_otsu(field['smooth'])),
                    ('Seeded random walker', rw), ('Seeded watershed', field['watershed']),
                    ('Watershed + local ISO50', local50),
                    ('Watershed + gradient peak', final)]
        fig, axes = plt.subplots(1, len(variants), figsize=(18, 11))
        for ax, (name, mask) in zip(axes, variants):
            plot_mask(ax, field['raw'], mask=mask, title=name)
            records.append({'slice': index, 'method': name, **diagnostics(mask)})
        fig.suptitle(f'img_{index:04d}: green = candidate material-air boundary')
        fig.tight_layout()
        fig.savefig(args.output / f'comparison_{index:04d}.png', dpi=140)
        plt.close(fig)
        # Inspect the actual low-contrast channel, not just whole-image scores.
        if index == 988:
            fig, axes = plt.subplots(2, 3, figsize=(13, 10))
            for ax, (name, mask) in zip(axes.ravel(), variants):
                plot_mask(ax, field['raw'], mask=mask, title=name)
                ax.set_xlim(90, 290)
                ax.set_ylim(540, 240)
            fig.tight_layout()
            fig.savefig(args.output / 'channel_detail_0988.png', dpi=160)
            plt.close(fig)
        save_slice(args.output / 'study', index, raw, field, refined)
        save_slice(args.output / 'extrema_seeds', index, raw, field, alt_refined)
        records.append({'slice': index, 'method': 'Extrema-seeded watershed',
                        **diagnostics(alternate)})
        fig, axes = plt.subplots(1, 2, figsize=(9, 11))
        plot_mask(axes[0], field['raw'], lines=[c['gradient'] for c in refined],
                  title='Threshold seeds + gradient')
        plot_mask(axes[1], field['raw'], lines=[c['gradient'] for c in alt_refined],
                  title='Local-extrema seeds + gradient')
        fig.tight_layout()
        fig.savefig(args.output / f'seed_comparison_{index:04d}.png', dpi=145)
        plt.close(fig)
        valid = np.concatenate([c['gradient_valid'] for c in refined])
        shifts = np.concatenate([c['gradient_shift'] for c in refined])
        gradpoints = np.concatenate([c['gradient'][:-1] for c in refined])
        isopoints = np.concatenate([c['local50'][:-1] for c in refined])
        disagreement = np.linalg.norm(gradpoints - isopoints, axis=1)
        records.append({'slice': index, 'method': 'refinement_diagnostics',
                        'supported_fraction': float(valid.mean()),
                        'shift_abs_median_px': float(np.median(abs(shifts))),
                        'shift_abs_p95_px': float(np.percentile(abs(shifts), 95)),
                        'iso_gradient_disagreement_p95_px': float(np.percentile(disagreement, 95)),
                        'low_threshold_u16': field['low'] * 65535,
                        'high_seed_threshold_u16': field['high'] * 65535})
        # Sensitivity: same anatomy, modest smoothing/seed changes.
        base_points = gradpoints + [field['box'][0], field['box'][2]]
        for sigma, high_factor in [(.75, 1), (1.25, 1), (1, .9), (1, 1.1)]:
            other = segment_slice(raw, sigma)
            changed_markers, _ = markers_for(other['smooth'], other['initial'],
                                             other['high'] * high_factor)
            changed = segmentation.watershed(other['elevation'], changed_markers) == 2
            points = normal_refinement(other['smooth'], changed)
            points = np.concatenate([c['gradient'][:-1] for c in points])
            points += [other['box'][0], other['box'][2]]
            d1 = cKDTree(points).query(base_points)[0]
            d2 = cKDTree(base_points).query(points)[0]
            records.append({'slice': index, 'method': 'parameter_sensitivity',
                            'sigma': sigma, 'high_seed_factor': high_factor,
                            'symmetric_distance_p95_px': float(np.percentile(np.r_[d1, d2], 95)),
                            'max_distance_px': float(max(d1.max(), d2.max()))})
        print(f'Study img_{index:04d}: {perf_counter()-started:.1f}s; '
              f'supported profiles {valid.mean():.1%}', flush=True)
    (args.output / 'study_metrics.json').write_text(json.dumps(records, indent=2) + '\n')


def volume(args):
    indices = list(range(args.first, args.last + 1))
    raw = np.stack([load(args.input, index) for index in indices])
    lows, boxes = [], []
    for image in raw:
        sm = ndi.gaussian_filter(image, 1)
        low = float(filters.threshold_otsu(sm))
        lows.append(low)
        boxes.append(bounding_box(largest(sm > low)))
    box = (min(b[0] for b in boxes), max(b[1] for b in boxes),
           min(b[2] for b in boxes), max(b[3] for b in boxes))
    y0, y1, x0, x1 = box
    cropped = raw[:, y0:y1, x0:x1]
    smooth = ndi.gaussian_filter(cropped, (.7, 1, 1))
    markers = np.zeros(smooth.shape, np.uint8)
    for z in range(len(indices)):
        initial = largest(smooth[z] > lows[z])
        markers[z], _ = markers_for(smooth[z], initial)
    print(f'3D watershed: {smooth.shape}, {len(indices)} distinct original slices', flush=True)
    mask3d = segmentation.watershed(filters.sobel(smooth), markers) == 2
    results, metrics = [], []
    for z, index in enumerate(indices):
        field = {'raw': cropped[z], 'smooth': smooth[z], 'box': box}
        items = normal_refinement(smooth[z], mask3d[z])
        full = save_slice(args.output / 'volume', index, raw[z], field, items)
        results.append(full[y0:y1, x0:x1] != 0)
        valid = np.concatenate([c['gradient_valid'] for c in items])
        metrics.append({'slice': index, 'gradient_supported_fraction': float(valid.mean()),
                        **diagnostics(full != 0)})
    result = np.stack(results)
    np.savez_compressed(args.output / 'material_volume.npz', mask=result,
                        source_slices=indices, xy_origin_yx=[y0, x0],
                        spacing_um_assumed=args.spacing_um)
    # No Z padding: the patch is OPEN at the two ends of the selected stack.
    level = ndi.gaussian_filter(result.astype(np.float32), (.5, .6, .6))
    vertices, faces, _, _ = measure.marching_cubes(level, .5, allow_degenerate=False)
    xyz = np.ascontiguousarray(vertices[:, [2, 1, 0]], dtype='<f4')
    xyz[:, 0] += x0
    xyz[:, 1] += y0
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment units=voxels; z=0 at first source slice; open test-window patch\n'
              f'element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n'
              f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
    with (args.output / 'surface_patch_voxels.ply').open('wb') as stream:
        stream.write(header.encode())
        stream.write(xyz.tobytes())
        face_data = np.empty(len(faces), dtype=[('n', 'u1'), ('indices', '<i4', (3,))])
        face_data['n'] = 3
        face_data['indices'] = faces
        stream.write(face_data.tobytes())
    metadata = {'source_slices': indices, 'source_pattern': 'img_{slice:04d}.tif',
                'input_shape_zyx': list(raw.shape), 'crop_box_y0_y1_x0_x1': box,
                'mesh_units': 'voxels', 'mesh_z_origin_source_slice': args.first,
                'voxel_size_um_from_existing_cpp_not_independently_calibrated': args.spacing_um,
                'mesh_is_open_at_stack_ends': True,
                'mesh_vertices': len(vertices), 'mesh_faces': len(faces),
                'slices': metrics}
    (args.output / 'volume_metrics.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved surface patch: {len(vertices)} vertices, {len(faces)} faces', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('CT_Test'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study-slices', type=lambda s: list(map(int, s.split(','))),
                        default=[500, 625, 750, 875, 988])
    parser.add_argument('--first', type=int, default=976)
    parser.add_argument('--last', type=int, default=1000)
    parser.add_argument('--spacing-um', type=float, default=75.4)
    parser.add_argument('--skip-study', action='store_true')
    parser.add_argument('--skip-volume', action='store_true')
    args = parser.parse_args()
    if args.last < args.first:
        parser.error('--last must be >= --first')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.skip_study:
        study(args)
    if not args.skip_volume:
        volume(args)


if __name__ == '__main__':
    main()
