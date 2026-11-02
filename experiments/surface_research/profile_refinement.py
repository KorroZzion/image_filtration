"""Subvoxel 3D normal-profile refinement with explicit rejection.

Coordinates are Z,Y,X voxel-centre indices. Normals are derived in physical
coordinates from the material mask, and raw data are sampled trilinearly along
those normals. max_shift and returned *_shift_vox are in units of the smallest
voxel spacing. These interpolated coordinates are not a certified accuracy.

The erf fit is an engineering blurred-step model, not a reconstruction of a
particular paper. It includes a linear background, Huber robust weighting, and
per-profile rejection. Single-edge models deliberately reject many thin-wall,
corner and overlapping-edge profiles; rejected points keep their seed position.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.special import ndtr


REJECTION_FLAGS = {
    1: 'profile_outside_volume_or_invalid_normal',
    2: 'nonpositive_or_low_snr_contrast',
    4: 'maximum_at_search_limit',
    8: 'strong_neighbour_or_nonmonotone_edge',
    16: 'no_local_half_level_crossing',
    32: 'poor_fit_residual_or_condition',
    64: 'fit_sigma_or_shift_at_limit',
    128: 'missing_local_plateau_or_excessive_background_slope',
    256: 'invalid_local_cubic',
}


def _erf_batch(t, profiles, mu, contrast, noise, max_shift):
    """Robust batched Gauss--Newton; no per-vertex scipy optimizers."""
    n = len(profiles)
    p = np.empty((n, 5), np.float64)
    p[:, 0] = np.median(profiles[:, t < -4], axis=1)
    p[:, 1] = 0
    p[:, 2] = np.maximum(contrast, 1e-8)
    p[:, 3] = np.clip(mu, -max_shift + .01, max_shift - .01)
    p[:, 4] = 1.0
    eye = np.eye(5)[None]
    condition_bad = np.zeros(n, bool)
    for iteration in range(13):
        u = (t[None] - p[:, 3, None]) / p[:, 4, None]
        cdf = ndtr(u)
        density = np.exp(-.5 * u * u) / np.sqrt(2 * np.pi)
        estimate = p[:, 0, None] + p[:, 1, None] * t + p[:, 2, None] * cdf
        residual = profiles - estimate
        # A robust scale, bounded below by endpoint noise and a fraction of
        # local contrast, avoids collapsing weights on nearly noiseless data.
        scale = np.maximum(np.median(abs(residual), axis=1) * 1.4826,
                           np.maximum(noise, abs(contrast) * .005))
        weights = np.minimum(1, 1.5 * scale[:, None] / np.maximum(abs(residual), 1e-12))
        jac = np.stack([np.ones_like(cdf), np.broadcast_to(t, cdf.shape), cdf,
                        -p[:, 2, None] * density / p[:, 4, None],
                        -p[:, 2, None] * u * density / p[:, 4, None]], axis=2)
        lhs = np.einsum('nki,nk,nkj->nij', jac, weights, jac)
        rhs = np.einsum('nki,nk,nk->ni', jac, weights, residual)
        diagonal = np.maximum(np.diagonal(lhs, axis1=1, axis2=2), 1e-7)
        lhs += eye * (diagonal * .003 + 1e-8)[:, None, :]
        try:
            delta = np.linalg.solve(lhs, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            # Damping normally prevents singularity. Failing this batch marks
            # it rejected instead of silently presenting a seed as a fit.
            condition_bad[:] = True
            break
        delta[:, 3] = np.clip(delta[:, 3], -.4, .4)
        delta[:, 4] = np.clip(delta[:, 4], -.25, .25)
        p += delta
        p[:, 2] = np.clip(p[:, 2], np.maximum(contrast * .15, 1e-8),
                           np.maximum(contrast * 3, 1e-7))
        p[:, 3] = np.clip(p[:, 3], -max_shift, max_shift)
        p[:, 4] = np.clip(p[:, 4], .35, 2.5)
    u = (t[None] - p[:, 3, None]) / p[:, 4, None]
    model = p[:, 0, None] + p[:, 1, None] * t + p[:, 2, None] * ndtr(u)
    residual = profiles - model
    rms = np.sqrt(np.mean(residual ** 2, axis=1))
    # Parameter covariance is a local model diagnostic; correlated CT noise,
    # PSF mismatch and calibration are not represented by this number.
    density = np.exp(-.5 * u * u) / np.sqrt(2 * np.pi)
    jac = np.stack([np.ones_like(u), np.broadcast_to(t, u.shape), ndtr(u),
                    -p[:, 2, None] * density / p[:, 4, None],
                    -p[:, 2, None] * u * density / p[:, 4, None]], axis=2)
    lhs = np.einsum('nki,nkj->nij', jac, jac)
    try:
        cov = np.linalg.inv(lhs + eye * 1e-8)
        mu_se = np.sqrt(np.maximum(cov[:, 3, 3], 0)) * rms
        jac_scale = np.sqrt(np.maximum(np.diagonal(lhs, axis1=1, axis2=2), 1e-20))
        dimensionless_lhs = lhs / (jac_scale[:, :, None] * jac_scale[:, None, :])
        condition_bad |= ~np.isfinite(mu_se) | (np.linalg.cond(dimensionless_lhs) > 1e9)
    except np.linalg.LinAlgError:
        condition_bad[:] = True
        mu_se = np.full(n, np.nan)
    return p, rms, mu_se, condition_bad


def refine_surface(raw, mask, vertices_zyx, spacing_zyx=(1, 1, 1),
                   max_shift=2, batch_size=2048, normals_physical_zyx=None):
    """Compare local half level, gradient peak, Taylor and erf fits in 3D.

    Returns a dict of coordinate arrays ``local50``, ``gradient``, ``taylor``, ``erf``;
    each has ``<method>_accepted``, ``<method>_shift_vox`` and
    ``<method>_flags``. Rejected coordinates equal the input seed exactly.
    ``erf_candidate_shift_vox`` preserves the fitted candidate for diagnostics.
    Physical coordinates are obtained by multiplying ZYX indices by spacing;
    shifts themselves are measured along the physical normal.

    Provided ``normals_physical_zyx`` must point from dark to bright and are
    normalized internally. They bypass EDT/derivative computation, supporting
    grayscale-gradient/NMS candidates independent of a binary mask topology.
    Without them the mask establishes topology; refinement cannot invent
    missing channels.
    Coordinate arrays are never smoothed along the surface or across rejected
    points. No claims of micrometre accuracy follow from a valid profile.
    """
    raw = np.asarray(raw, np.float32)
    mask = np.asarray(mask, bool)
    vertices = np.asarray(vertices_zyx, np.float64)
    spacing = np.asarray(spacing_zyx, np.float64)
    if raw.ndim != 3 or mask.shape != raw.shape:
        raise ValueError('raw and mask must have matching ZYX shape')
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError('vertices_zyx must be Nx3')
    if spacing.shape != (3,) or np.any(spacing <= 0) or not np.all(np.isfinite(spacing)):
        raise ValueError('spacing must be three positive finite numbers')
    if max_shift <= .1 or max_shift > 3:
        raise ValueError('max_shift must be in (.1, 3] reference voxels')
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(vertices)):
        raise ValueError('nonfinite input')
    unit = float(spacing.min())
    if normals_physical_zyx is None:
        signed = (ndi.distance_transform_edt(mask, sampling=spacing) -
                  ndi.distance_transform_edt(~mask, sampling=spacing)).astype(np.float32)
        normal = np.empty_like(vertices)
        for axis in range(3):
            order = [0, 0, 0]
            order[axis] = 1
            derivative = ndi.gaussian_filter(signed, 1.0, order=order) / spacing[axis]
            normal[:, axis] = ndi.map_coordinates(derivative, vertices.T, order=1, mode='nearest')
        length = np.linalg.norm(normal, axis=1)
    else:
        normal = np.asarray(normals_physical_zyx, dtype=np.float64).copy()
        if normal.shape != vertices.shape or not np.all(np.isfinite(normal)):
            raise ValueError('normals_physical_zyx must be finite Nx3 matching vertices')
        length = np.linalg.norm(normal, axis=1)
        if np.any(length < 1e-8):
            raise ValueError('provided physical normals must be nonzero')
    normal /= np.maximum(length[:, None], 1e-12)
    direction = normal * unit / spacing
    n = len(vertices)
    result = {'seed': vertices.copy(), 'normal_physical_zyx': normal,
              'reference_spacing': unit,
              'rejection_flags': REJECTION_FLAGS.copy()}
    for method in ('gradient', 'local50', 'taylor', 'erf'):
        result[method] = vertices.copy()
        result[method + '_accepted'] = np.zeros(n, bool)
        result[method + '_shift_vox'] = np.zeros(n)
        result[method + '_flags'] = np.zeros(n, np.uint16)
    for key in ('contrast', 'snr', 'noise', 'erf_sigma_vox', 'erf_rms',
                'erf_relative_rms', 'erf_mu_standard_error_model_only_vox',
                'erf_candidate_shift_vox', 'gradient_candidate_shift_vox',
                'local50_candidate_shift_vox', 'taylor_candidate_shift_vox'):
        result[key] = np.full(n, np.nan)
    t = np.arange(-6, 6.001, .25, dtype=np.float64)
    search = np.flatnonzero(abs(t) <= max_shift)
    raw_range = float(np.percentile(raw[::max(1, raw.shape[0] // 12)], 99) -
                      np.percentile(raw[::max(1, raw.shape[0] // 12)], 1))
    noise_floor = max(raw_range * 1e-4, 1e-8)
    for first in range(0, n, batch_size):
        last = min(n, first + batch_size)
        seed = vertices[first:last]
        dirs = direction[first:last]
        coords = seed[:, None] + dirs[:, None] * t[None, :, None]
        valid_bounds = (np.all(coords >= 0, axis=(1, 2)) &
                        np.all(coords <= np.asarray(raw.shape)[None, None] - 1, axis=(1, 2)) &
                        (length[first:last] > 1e-6))
        profiles = ndi.map_coordinates(raw, coords.reshape(-1, 3).T,
                                       order=1, mode='nearest').reshape(len(seed), -1)
        air = np.median(profiles[:, t <= -4], axis=1)
        material = np.median(profiles[:, t >= 4], axis=1)
        contrast = material - air
        # Endpoint noise is estimated after removing their own linear trends.
        end_residuals = []
        for select in (t <= -4, t >= 4):
            te = t[select]
            ve = profiles[:, select]
            slope = ((ve - ve.mean(axis=1)[:, None]) @ (te - te.mean())) / np.sum((te-te.mean())**2)
            end_residuals.append(ve - ve.mean(axis=1)[:, None] - slope[:, None]*(te-te.mean()))
        noise = np.maximum(np.median(abs(np.concatenate(end_residuals, axis=1)), axis=1) * 1.4826,
                           noise_floor)
        snr = contrast / noise
        filtered = ndi.gaussian_filter1d(profiles, .6 / .25, axis=1, mode='nearest')
        deriv = np.gradient(filtered, .25, axis=1)
        k = search[np.argmax(deriv[:, search], axis=1)]
        rows = np.arange(len(seed))
        before = deriv[rows, k-1]
        peak = deriv[rows, k]
        after = deriv[rows, k+1]
        denom = before - 2 * peak + after
        delta = np.divide(.5 * (before-after), denom,
                          out=np.zeros_like(peak), where=abs(denom)>1e-12)
        grad_shift = t[k] + np.clip(delta, -.5, .5)*.25
        # Extra rising/falling edges in the profile make a single-step model
        # ambiguous. A broad drift is tolerated; a strong reverse edge is not.
        negative_peak = np.max(-deriv[:, (t >= -4) & (t <= 4)], axis=1)
        nonmonotone = negative_peak > np.maximum(peak * .35, noise * 1.5)
        common_flags = np.zeros(len(seed), np.uint16)
        common_flags[~valid_bounds] |= 1
        common_flags[(contrast <= 8 * noise_floor) | (snr < 6)] |= 2
        common_flags[nonmonotone] |= 8
        grad_flags = common_flags.copy()
        grad_flags[(k == search[0]) | (k == search[-1])] |= 4
        target = (air + material) / 2
        rising = ((profiles[:, :-1] <= target[:, None]) &
                  (profiles[:, 1:] > target[:, None]) &
                  (abs(t[:-1])[None] <= max_shift))
        distance = abs(t[:-1][None] - grad_shift[:, None])
        chosen = np.argmin(np.where(rising, distance, np.inf), axis=1)
        low = profiles[rows, chosen]
        high = profiles[rows, chosen+1]
        frac = np.divide(target-low, high-low, out=np.zeros_like(target), where=abs(high-low)>1e-12)
        iso_shift = t[chosen] + .25*np.clip(frac, 0, 1)
        iso_flags = common_flags.copy()
        iso_flags[~rising.any(axis=1) | (abs(iso_shift) > max_shift)] |= 16
        # Cubic directional Taylor approximation near the smoothed gradient
        # peak. The stationary first derivative is at -c2/(3*c3). This is
        # a local polynomial implementation of the Taylor idea; not an exact
        # reproduction of any paper's derivative discretization.
        offsets = np.arange(-5, 6)
        dt = offsets * .25
        design = np.stack([np.ones_like(dt), dt, dt**2, dt**3], axis=1)
        values = profiles[rows[:, None], k[:, None] + offsets[None]]
        coefficients = values @ np.linalg.pinv(design).T
        c2, c3 = coefficients[:, 2], coefficients[:, 3]
        offset = np.divide(-c2, 3*c3, out=np.zeros_like(c2), where=abs(c3)>1e-12)
        taylor_shift = t[k] + offset
        taylor_flags = common_flags.copy()
        taylor_derivative = coefficients[:, 1] + 2*c2*offset + 3*c3*offset**2
        taylor_residual = np.sqrt(np.mean((values - coefficients @ design.T)**2, axis=1))
        taylor_flags[(c3 >= -1e-8) | (abs(offset) > 1.25) |
                     (abs(taylor_shift) > max_shift) |
                     (taylor_derivative <= 0) |
                     (taylor_residual > abs(contrast)*.15)] |= 256
        p, rms, mu_se, condition_bad = _erf_batch(t, profiles, grad_shift,
                                                np.maximum(contrast, 1e-7), noise, max_shift)
        relative_rms = rms / np.maximum(abs(p[:, 2]), noise_floor)
        fit_flags = common_flags.copy()
        fit_flags[(relative_rms > .08) | condition_bad | ~np.isfinite(p).all(axis=1)] |= 32
        fit_flags[(p[:, 4] <= .36) | (p[:, 4] >= 2.45) |
                  (abs(p[:, 3]) >= max_shift-.025)] |= 64
        # Plateau must exist for at least ~1 reference voxel at both ends;
        # broad PSFs/nearby interfaces are not extrapolated into acceptance.
        fit_flags[(abs(p[:, 1])*12 > abs(p[:, 2])*.8) |
                  (p[:, 3] - 2*p[:, 4] < -5) |
                  (p[:, 3] + 2*p[:, 4] > 5)] |= 128
        for method, shift, flags in [('gradient', grad_shift, grad_flags),
                                      ('local50', iso_shift, iso_flags),
                                      ('taylor', taylor_shift, taylor_flags),
                                      ('erf', p[:, 3], fit_flags)]:
            accepted = flags == 0
            result[method + '_accepted'][first:last] = accepted
            result[method + '_flags'][first:last] = flags
            result[method + '_shift_vox'][first:last] = np.where(accepted, shift, 0)
            result[method][first:last] = seed + dirs * np.where(accepted, shift, 0)[:, None]
            result[method + '_candidate_shift_vox'][first:last] = shift
        for key, values in [('contrast', contrast), ('snr', snr), ('noise', noise),
                            ('erf_sigma_vox', p[:, 4]), ('erf_rms', rms),
                            ('erf_relative_rms', relative_rms),
                            ('erf_mu_standard_error_model_only_vox', mu_se)]:
            result[key][first:last] = values
    return result
