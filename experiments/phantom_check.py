"""Independent localization check on known analytic material/air circles.

This verifies the refinement principle, not accuracy on the real CT scan.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from boundary_study import contours, normal_refinement, segment_slice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    y, x = np.mgrid[:320, :320]
    circles = [(160, 160, 112), (158, 155, 50), (221, 116, 9)]
    truth = (x - 160)**2 + (y - 160)**2 < 112**2
    for cx, cy, radius in circles[1:]:
        truth &= (x-cx)**2 + (y-cy)**2 > radius**2
    rng = np.random.default_rng(42)
    results = []
    for name, bias, noise in [('uniform', 0., .001), ('varying_brightness', .28, .001),
                               ('varying_brightness_noise', .28, .006)]:
        air = .10 + .025 * x / 320
        material_contrast = .40 + bias * (x / 320 - .5)
        image = air + material_contrast * truth
        image = ndi.gaussian_filter(image, 1.2)
        image += rng.normal(0, noise, image.shape)
        image = image.astype(np.float32)
        field = segment_slice(image)
        items = normal_refinement(field['smooth'], field['watershed'])
        candidates = {'global_otsu': contours(field['initial']),
                      'watershed': contours(field['watershed']),
                      'gradient_peak': [c['gradient'] for c in items],
                      'local50': [c['local50'] for c in items]}
        y0, _, x0, _ = field['box']
        case = {'condition': name}
        for method, lines in candidates.items():
            points = np.concatenate(lines) + [y0, x0]
            distances = np.stack([abs(np.hypot(points[:, 1]-cx, points[:, 0]-cy)-radius)
                                  for cx, cy, radius in circles])
            error = distances.min(axis=0)
            case[method] = {'median_error_px': float(np.median(error)),
                            'p95_error_px': float(np.percentile(error, 95)),
                            'max_error_px': float(error.max())}
        # A one-pixel localization failure on a simple, resolved phantom must
        # not be reported as successful surface localization.
        assert case['gradient_peak']['p95_error_px'] < 1.0, case
        assert case['local50']['p95_error_px'] < 1.0, case
        results.append(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
