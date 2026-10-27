"""Hybrid experiment: reinitialize the outer support, propagate cavity seeds.

Avoids the observed outer-boundary drift of track_boundaries.py. The initial
cavity set still matters; a new cavity without an air seed may be missed.
"""
import argparse
import json
from pathlib import Path

from boundary_study import *


def advance_channels(raw, previous):
    field = segment_slice(raw, run_watershed=False)
    y0, y1, x0, x1 = field['box']
    previous_holes = ndi.binary_fill_holes(previous) & ~previous
    holes = previous_holes[y0:y1, x0:x1]
    prev = previous[y0:y1, x0:x1]
    envelope = ndi.binary_fill_holes(field['initial'])
    markers = np.zeros(prev.shape, np.uint8)
    # Recompute exterior support in every frame instead of propagating it.
    markers[ndi.distance_transform_edt(~envelope) > 2] = 1
    # Do not force an old cavity to stay open after it has become material.
    # Retain a seed only where current intensity is below its surrounding wall.
    possible_air = (ndi.distance_transform_edt(holes) > 2) & envelope
    labels, _ = ndi.label(holes)
    accepted_air = np.zeros(holes.shape, bool)
    for label, bounds in enumerate(ndi.find_objects(labels), start=1):
        if bounds is None:
            continue
        a, b = bounds
        area = (slice(max(0,a.start-6),min(holes.shape[0],a.stop+6)),
                slice(max(0,b.start-6),min(holes.shape[1],b.stop+6)))
        hole = labels[area] == label
        ring = ndi.binary_dilation(hole, iterations=5) & ~hole & prev[area]
        values = field['smooth'][area]
        if np.count_nonzero(ring) < 8:
            continue
        wall_level = float(np.median(values[ring]))
        accepted_air[area] |= hole & possible_air[area] & (values < wall_level - .02)
    markers[accepted_air] = 1
    near_holes = ndi.binary_dilation(holes, iterations=3)
    new_material = (field['smooth'] > field['high']) & envelope & ~near_holes
    core = (ndi.distance_transform_edt(prev) > 3) & envelope
    markers[core | ndi.binary_erosion(new_material, iterations=1)] = 2
    material = segmentation.watershed(field['elevation'], markers) == 2
    result = np.zeros(raw.shape, bool)
    result[y0:y1, x0:x1] = material
    return result, field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('CT_Test'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed-mask', type=Path, required=True)
    parser.add_argument('--seed-slice', type=int, default=988)
    parser.add_argument('--first', type=int, default=500)
    parser.add_argument('--last', type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    masks = args.output / 'masks'
    masks.mkdir(exist_ok=True)
    with Image.open(args.seed_mask) as image:
        initial = np.asarray(image) > 0
    records = []

    def save(index, mask, raw, field):
        Image.fromarray(mask.astype(np.uint8)*255).save(masks / f'material_{index:04d}.png')
        records.append({'slice': index, **diagnostics(mask)})
        if index in [500, 625, 750, 875, 988, 1000]:
            y0, y1, x0, x1 = field['box']
            items = normal_refinement(field['smooth'], mask[y0:y1,x0:x1])
            save_slice(args.output / 'refined_examples', index, raw, field, items)

    raw = load(args.input, args.seed_slice)
    save(args.seed_slice, initial, raw, segment_slice(raw))
    for order in [range(args.seed_slice-1, args.first-1, -1),
                  range(args.seed_slice+1, args.last+1)]:
        mask = initial.copy()
        for index in order:
            raw = load(args.input, index)
            mask, field = advance_channels(raw, mask)
            save(index, mask, raw, field)
            if index % 50 == 0:
                print('Hybrid reached', index, flush=True)
    records.sort(key=lambda item:item['slice'])
    assert [item['slice'] for item in records] == list(range(args.first,args.last+1))
    (args.output / 'hybrid_metrics.json').write_text(json.dumps(records,indent=2)+'\n')
    print('Completed',len(records),'real source slices',flush=True)


if __name__ == '__main__':
    main()
