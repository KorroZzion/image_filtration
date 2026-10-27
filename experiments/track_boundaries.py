"""Test slice-to-slice marker propagation and its round-trip consistency.

This experiment depends on its initial segmentation. It can close cavities
but cannot reliably discover a new isolated cavity with no propagated seed.
"""
import argparse
import json
from pathlib import Path

from boundary_study import *


def advance(raw, previous, band=3):
    box = bounding_box(previous, padding=24)
    y0, y1, x0, x1 = box
    prev = previous[y0:y1, x0:x1]
    smooth = ndi.gaussian_filter(raw[y0:y1, x0:x1], 1)
    inside = ndi.distance_transform_edt(prev)
    outside = ndi.distance_transform_edt(~prev)
    markers = np.zeros(prev.shape, np.uint8)
    markers[outside > band] = 1
    markers[inside > band] = 2
    mask = segmentation.watershed(filters.sobel(smooth), markers) == 2
    full = np.zeros(raw.shape, bool)
    full[y0:y1, x0:x1] = mask
    return full, {'raw': raw[y0:y1, x0:x1], 'smooth': smooth, 'box': box}


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
    reverse = args.output / 'reverse'
    forward = args.output / 'forward'
    reverse.mkdir(exist_ok=True)
    forward.mkdir(exist_ok=True)
    with Image.open(args.seed_mask) as image:
        initial = np.asarray(image) > 0
    mask = initial.copy()
    records = []
    for index in range(args.seed_slice - 1, args.first - 1, -1):
        mask, field = advance(load(args.input, index), mask)
        Image.fromarray(mask.astype(np.uint8)*255).save(reverse / f'material_{index:04d}.png')
        if index in [500, 625, 750, 875]:
            items = normal_refinement(field['smooth'], mask[field['box'][0]:field['box'][1],
                                                            field['box'][2]:field['box'][3]])
            save_slice(args.output / 'reverse_examples', index, load(args.input, index), field, items)
        if index % 50 == 0:
            print('Reverse reached', index, flush=True)
    # A second pass starts from the propagated first slice. This is a useful
    # independent cycle-consistency diagnostic, not ground-truth validation.
    for index in range(args.first, args.last + 1):
        raw = load(args.input, index)
        if index != args.first:
            mask, field = advance(raw, mask)
        Image.fromarray(mask.astype(np.uint8)*255).save(forward / f'material_{index:04d}.png')
        record = {'slice': index, **diagnostics(mask)}
        if index < args.seed_slice:
            with Image.open(reverse / f'material_{index:04d}.png') as image:
                other = np.asarray(image) > 0
            record['reverse_forward_iou'] = float((mask & other).sum() / (mask | other).sum())
        if index == args.seed_slice:
            record['seed_roundtrip_iou'] = float((mask & initial).sum() / (mask | initial).sum())
        if index in [500, 625, 750, 875, 988, 1000]:
            box = bounding_box(mask)
            y0, y1, x0, x1 = box
            field = {'raw': raw[y0:y1,x0:x1], 'smooth': ndi.gaussian_filter(raw,1)[y0:y1,x0:x1], 'box':box}
            items = normal_refinement(field['smooth'], mask[y0:y1,x0:x1])
            # Keep propagation masks distinct from refined curve rasterization.
            save_slice(args.output / 'refined_examples', index, raw, field, items)
        records.append(record)
        if index % 50 == 0:
            print('Forward reached', index, flush=True)
    (args.output / 'tracking_metrics.json').write_text(json.dumps(records, indent=2)+'\n')
    print('Roundtrip:', next(r for r in records if r['slice']==args.seed_slice), flush=True)


if __name__ == '__main__':
    main()
