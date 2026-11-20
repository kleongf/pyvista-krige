"""Tests for ``KrigingModel.simulate``, sequential Gaussian simulation."""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

import pyvista_krige


@pytest.fixture
def skewed() -> pv.PolyData:
    """Forty scattered samples of a smooth, lognormal-looking field over 10 by 10."""
    rng = np.random.default_rng(1)
    points = rng.uniform(0, 10, (40, 3)) * [1, 1, 0]
    samples = pv.PolyData(points)
    field = np.sin(points[:, 0] / 2) + 0.5 * np.cos(points[:, 1] / 3)
    samples['grade'] = np.exp(field + rng.normal(0, 0.3, 40))
    return samples


def _grid(size: int) -> pv.ImageData:
    return pv.ImageData(dimensions=(size, size, 1), spacing=(10 / (size - 1),) * 2 + (1,))


def _stack(result: pv.DataSet, n: int, scalars: str = 'grade') -> np.ndarray:
    return np.column_stack([result[f'{scalars} realization {i}'] for i in range(n)])


def test_writes_n_realizations_and_leaves_the_target_alone(skewed):
    model = skewed.krige.fit('grade', variogram_model='exponential')
    grid = _grid(8)
    result = model.simulate(grid, n=4, seed=0)
    names = [name for name in result.point_data if 'realization' in name]
    assert names == [f'grade realization {i}' for i in range(4)]
    assert grid.point_data.keys() == []
    assert result.n_points == grid.n_points


@pytest.mark.parametrize('transform', [None, 'log', 'normal_score'])
@pytest.mark.parametrize('nugget', [0.0, 0.05])
def test_realizations_equal_the_data_at_the_samples(skewed, transform, nugget):
    """Even with a nugget, a target point at a sample takes the sample value."""
    model = skewed.krige.fit(
        'grade',
        variogram_model='spherical',
        variogram_parameters={'sill': 1.0, 'range': 4.0, 'nugget': nugget},
        transform=transform,
    )
    target = pv.PolyData(np.vstack([skewed.points, _grid(6).points]))
    runs = _stack(model.simulate(target, n=5, seed=3), 5)
    np.testing.assert_array_equal(runs[: skewed.n_points], skewed['grade'][:, None].repeat(5, 1))


def test_honors_the_samples_with_anisotropy_in_3d():
    rng = np.random.default_rng(4)
    samples = pv.PolyData(rng.uniform(0, 10, (30, 3)))
    samples['v'] = samples.points @ [1.0, 0.5, -0.2] + rng.normal(0, 0.5, 30)
    model = samples.krige.fit(
        'v', variogram_model='exponential', anisotropy_scaling_z=3.0, anisotropy_angle_z=30.0
    )
    target = pv.PolyData(np.vstack([samples.points, rng.uniform(0, 10, (50, 3))]))
    runs = _stack(model.simulate(target, n=3, seed=0), 3, 'v')
    np.testing.assert_array_equal(runs[:30], samples['v'][:, None].repeat(3, 1))
    assert np.all(np.isfinite(runs))


def test_the_same_seed_repeats_and_another_seed_differs(skewed):
    model = skewed.krige.fit('grade', variogram_model='exponential')
    grid = _grid(8)
    first = _stack(model.simulate(grid, n=3, seed=7), 3)
    np.testing.assert_array_equal(first, _stack(model.simulate(grid, n=3, seed=7), 3))
    assert not np.allclose(first, _stack(model.simulate(grid, n=3, seed=8), 3))
    # The realizations differ from each other too.
    assert not np.allclose(first[:, 0], first[:, 1])


def test_realizations_stay_within_the_normal_score_limits(skewed):
    model = skewed.krige.fit('grade', variogram_model='exponential', transform='log')
    runs = _stack(model.simulate(_grid(10), n=20, seed=0), 20)
    table = pyvista_krige.NormalScoreTransform.from_samples(skewed['grade']).values
    assert runs.min() >= table[0]
    assert runs.max() <= table[-1]


def test_the_mean_and_variance_of_many_realizations_approach_kriging(skewed):
    """With every point in the neighborhood, only Monte Carlo error is left."""
    model = skewed.krige.fit('grade', variogram_model='exponential', transform='normal_score')
    grid = _grid(10)
    n = 1000
    runs = _stack(model.simulate(grid, n=n, seed=0, n_neighbors=200), n)
    kriged = model.predict(grid)
    scale = np.std(skewed['grade'])
    error = runs.mean(axis=1) - kriged['grade']
    assert np.sqrt(np.mean(error**2)) < 0.05 * scale
    uncertain = kriged['grade variance'] > 0.05 * scale**2
    ratio = runs.var(axis=1)[uncertain] / kriged['grade variance'][uncertain]
    assert 0.85 < np.median(ratio) < 1.15


def test_the_mean_is_close_to_kriging_with_the_default_neighborhood(skewed):
    """The target includes the sample points, which must not condition twice."""
    model = skewed.krige.fit('grade', variogram_model='exponential', transform='normal_score')
    target = pv.PolyData(np.vstack([skewed.points, _grid(20).points]))
    runs = _stack(model.simulate(target, n=200, seed=1), 200)
    error = runs.mean(axis=1) - model.predict(target)['grade']
    assert np.sqrt(np.mean(error**2)) < 0.15 * np.std(skewed['grade'])


def test_realizations_reproduce_the_variogram_at_short_lags(skewed):
    """Each point is conditioned on earlier ones, so neighbors vary as the model says."""
    model = skewed.krige.fit('grade', variogram_model='exponential', transform='normal_score')
    size, n = 20, 50
    grid = _grid(size)
    runs = _stack(model.simulate(grid, n=n, seed=0), n)
    scores = model.transform.forward(runs).reshape(size, size, n)
    variogram = model.variogram
    for lag in (1, 2, 4):
        experimental = 0.5 * np.mean((scores[:, lag:] - scores[:, :-lag]) ** 2)
        expected = variogram.function(variogram.parameters, lag * grid.spacing[0])
        assert 0.75 < experimental / expected < 1.25, lag


@pytest.mark.parametrize('transform', [None, 'normal_score'])
def test_realizations_reproduce_the_sample_histogram(skewed, transform):
    model = skewed.krige.fit('grade', variogram_model='exponential', transform=transform)
    runs = _stack(model.simulate(_grid(20), n=50, seed=2), 50)
    quantiles = np.linspace(0.1, 0.9, 9)
    error = np.quantile(runs, quantiles) - np.quantile(skewed['grade'], quantiles)
    assert np.abs(error).max() < 0.3 * np.std(skewed['grade'])


def test_summary_and_exceedance_match_the_realizations(skewed):
    model = skewed.krige.fit('grade', variogram_model='exponential')
    grid = _grid(8)
    runs = _stack(model.simulate(grid, n=40, seed=5), 40)
    summary = model.simulate(grid, n=40, seed=5, summary=True, exceed=2.5)
    assert not any('realization' in name for name in summary.point_data)
    np.testing.assert_allclose(summary['grade mean'], runs.mean(axis=1))
    for p in (10, 50, 90):
        np.testing.assert_allclose(summary[f'grade P{p}'], np.percentile(runs, p, axis=1))
    np.testing.assert_allclose(summary['P(grade > 2.5)'], (runs > 2.5).mean(axis=1))


def test_rejects_unsupported_input(skewed):
    model = skewed.krige.fit('grade')
    grid = _grid(4)
    with pytest.raises(TypeError, match='MultiBlock'):
        model.simulate(pv.MultiBlock([grid]), n=1)
    with pytest.raises(ValueError, match='n must be at least 1'):
        model.simulate(grid, n=0)
    with pytest.raises(ValueError, match='n_neighbors must be at least 1'):
        model.simulate(grid, n=1, n_neighbors=0)
    skewed['x'] = skewed.points[:, 0]
    universal = skewed.krige.fit('grade', method='universal', specified_drift=['x'])
    with pytest.raises(ValueError, match='ordinary kriging'):
        universal.simulate(grid, n=1)
