from __future__ import annotations

import numpy as np
import pytest

from pyvista_krige._anisotropy import estimate_anisotropy


def gaussian_field(
    n: int,
    seed: int,
    *,
    angle: float = 0.0,
    scales: tuple[float, ...] = (8.0, 8.0),
    extent: tuple[float, ...] = (100.0, 100.0),
    nugget: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a Gaussian-covariance random field at random points, by Cholesky factorization.

    ``scales`` are the correlation lengths along the field's own axes, which are
    rotated by ``angle`` degrees counter-clockwise from +x in the xy plane.
    """
    rng = np.random.default_rng(seed)
    points = rng.uniform(size=(n, len(extent))) * extent
    delta = points[:, None, :] - points[None, :, :]
    theta = np.radians(angle)
    along = delta[..., 0] * np.cos(theta) + delta[..., 1] * np.sin(theta)
    across = -delta[..., 0] * np.sin(theta) + delta[..., 1] * np.cos(theta)
    scaled = [along / scales[0], across / scales[1]]
    scaled.extend(delta[..., axis] / scales[axis] for axis in range(2, len(extent)))
    covariance = np.exp(-sum(c**2 for c in scaled)) + (nugget + 1e-8) * np.eye(n)
    return points, np.linalg.cholesky(covariance) @ rng.standard_normal(n)


def angle_error(a: float, b: float) -> float:
    """Return the difference of two orientations in degrees, modulo 180."""
    difference = abs(a - b) % 180.0
    return min(difference, 180.0 - difference)


def test_estimate_recovers_angle_and_ratio_of_a_2d_field():
    # Correlation lengths 12 and 4 along 30 degrees, so the true ratio is 3.
    fits = [
        estimate_anisotropy(
            *gaussian_field(350, seed, angle=30.0, scales=(12.0, 4.0)),
            'gaussian',
            nlags=12,
        )
        for seed in range(5)
    ]
    scaling = np.array([fit['anisotropy_scaling'] for fit in fits])
    assert set(fits[0]) == {'anisotropy_scaling', 'anisotropy_angle'}
    assert all(angle_error(fit['anisotropy_angle'], 30.0) < 20.0 for fit in fits)
    # Directional variograms blur the axes together, so the estimate runs low.
    assert np.all((scaling > 1.4) & (scaling < 5.0))
    assert 1.8 < np.median(scaling) < 4.5


def test_estimate_recovers_a_layered_3d_field():
    # Horizontal correlation length 15, vertical 5. One field gives a vertical ratio anywhere
    # from about 1.5 to 5, so judge the median of several.
    fits = [
        estimate_anisotropy(
            *gaussian_field(400, seed, scales=(15.0, 15.0, 5.0), extent=(80.0, 80.0, 50.0)),
            'gaussian',
            nlags=12,
        )
        for seed in range(5)
    ]
    assert set(fits[0]) == {'anisotropy_scaling_y', 'anisotropy_scaling_z', 'anisotropy_angle_z'}
    assert 1.8 < np.median([fit['anisotropy_scaling_z'] for fit in fits]) < 4.5
    assert np.median([fit['anisotropy_scaling_y'] for fit in fits]) < 1.8


def test_estimate_finds_little_anisotropy_in_an_isotropic_field():
    fits = [
        estimate_anisotropy(*gaussian_field(350, seed), 'gaussian', nlags=12) for seed in range(5)
    ]
    scaling = [fit['anisotropy_scaling'] for fit in fits]
    assert max(scaling) < 1.8
    assert np.median(scaling) < 1.5


@pytest.mark.parametrize('model', ['linear', 'power', 'custom'])
def test_estimate_needs_a_bounded_model(model):
    points, values = gaussian_field(40, 0)
    with pytest.raises(ValueError, match="variogram_model='spherical'"):
        estimate_anisotropy(points, values, model, nlags=6)


def test_estimate_explains_too_few_pairs_per_direction():
    # Samples on a line leave most horizontal directions without any pairs.
    rng = np.random.default_rng(0)
    along = rng.uniform(0, 100, 40)
    angle = np.radians(11.25)
    points = np.column_stack([along * np.cos(angle), along * np.sin(angle)])
    with pytest.raises(ValueError, match='Too few sample pairs per direction'):
        estimate_anisotropy(points, rng.standard_normal(40), 'spherical', nlags=6)


def test_estimate_needs_vertical_pairs_in_3d():
    points, values = gaussian_field(60, 0, extent=(100.0, 100.0, 1e-3), scales=(8.0, 8.0, 8.0))
    with pytest.raises(ValueError, match='vertical'):
        estimate_anisotropy(points, values, 'spherical', nlags=6)
