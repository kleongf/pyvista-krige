from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv
from scipy.spatial.distance import cdist
from scipy.special import ndtr

import pyvista_krige
from pyvista_krige import LogTransform
from pyvista_krige import NormalScoreTransform

# The log field below is Gaussian with this exponential covariance (range 5),
# which is PyKrige's 'exponential' model with range parameter 3 * 5.
VARIOGRAM = {'variogram_model': 'exponential', 'variogram_parameters': [1.0, 15.0, 0.0]}


@pytest.fixture(scope='module')
def lognormal_fields():
    """Thirty lognormal fields on a 20 x 20 grid, each with 40 random sample nodes."""
    side = 20
    xy = np.stack(np.meshgrid(np.arange(side), np.arange(side)), -1).reshape(-1, 2)
    points = np.column_stack([xy, np.zeros(len(xy))]).astype(float)
    covariance = np.exp(-cdist(xy, xy) / 5.0)
    lower = np.linalg.cholesky(covariance + 1e-10 * np.eye(len(xy)))
    fields = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        values = np.exp(lower @ rng.standard_normal(len(xy)))
        sampled = rng.choice(len(xy), 40, replace=False)
        unsampled = np.setdiff1d(np.arange(len(xy)), sampled)
        samples = pv.PolyData(points[sampled])
        samples['grade'] = values[sampled]
        fields.append((samples, pv.PolyData(points[unsampled]), values[unsampled]))
    return fields


@pytest.fixture
def skewed() -> pv.PolyData:
    rng = np.random.default_rng(3)
    samples = pv.PolyData(rng.uniform(0, 10, (50, 3)) * [1, 1, 0])
    samples['grade'] = np.exp(np.sin(samples.points[:, 0] / 2) + 0.5 * rng.standard_normal(50))
    return samples


def test_log_back_transform_is_unbiased_and_naive_exp_is_not(lognormal_fields):
    log_errors, naive_errors, means = [], [], []
    for samples, target, truth in lognormal_fields:
        model = samples.krige.fit('grade', transform='log', **VARIOGRAM)
        result = model.predict(target)
        log_errors.append(np.mean(result['grade'] - truth))
        naive_errors.append(np.mean(np.exp(result['grade transformed']) - truth))
        means.append(np.mean(truth))
    mean = np.mean(means)
    assert abs(np.mean(log_errors)) < 0.06 * mean
    assert np.mean(naive_errors) < -0.1 * mean


def test_exceedance_on_lognormal_field_is_closer_to_truth_with_transform(lognormal_fields):
    threshold = np.exp(1.0)
    transforms = (None, 'log', 'normal_score')
    scores = {transform: [] for transform in transforms}
    fractions = {transform: [] for transform in transforms}
    true_fractions = []
    for samples, target, truth in lognormal_fields:
        exceeds = truth > threshold
        true_fractions.append(exceeds.mean())
        for transform in transforms:
            model = samples.krige.fit('grade', transform=transform, **VARIOGRAM)
            result = pyvista_krige.exceedance(
                model.predict(target), 'grade', threshold, name='p', model=model
            )
            scores[transform].append(np.mean((result['p'] - exceeds) ** 2))
            fractions[transform].append(result['p'].mean())
    brier = {transform: np.mean(values) for transform, values in scores.items()}
    assert brier['log'] < 0.9 * brier[None]
    assert brier['normal_score'] < 0.95 * brier[None]
    miss = {t: abs(np.mean(f) - np.mean(true_fractions)) for t, f in fractions.items()}
    assert miss['log'] < miss[None]


@pytest.mark.parametrize('values', [[0.3, 1.0, 1.0, 2.5, 40.0, 7.0], np.linspace(-5, 5, 11)])
def test_normal_score_round_trip_at_the_samples(values):
    values = np.asarray(values, dtype=float)
    transform = NormalScoreTransform.from_samples(values)
    scores = transform.forward(values)
    assert np.all(np.isfinite(scores))
    back, variance = transform.back(scores, np.zeros_like(scores))
    np.testing.assert_allclose(back, values)
    np.testing.assert_allclose(variance, 0.0, atol=1e-9)


def test_normal_score_back_transform_ignores_round_off_variance():
    values = np.array([1.0, 2.0, 5.0, 9.0, 30.0])
    transform = NormalScoreTransform.from_samples(values)
    scores = transform.forward(values)
    # PyKrige leaves variances of this size at the samples of a zero-nugget model.
    back, variance = transform.back(scores, np.full_like(scores, 2e-10))
    np.testing.assert_allclose(back, values, rtol=1e-12)
    np.testing.assert_allclose(variance, 0.0, atol=1e-9)


def test_normal_scores_are_standard_normal_and_ties_share_a_score():
    values = np.repeat(np.arange(1.0, 101.0), 2)
    scores = NormalScoreTransform.from_samples(values).forward(values)
    assert scores[0] == scores[1]
    assert abs(scores.mean()) < 1e-12
    assert scores.std() == pytest.approx(1.0, abs=0.03)


def test_normal_score_tails_reach_the_limits():
    values = np.array([1.0, 2.0, 4.0, 8.0])
    default = NormalScoreTransform.from_samples(values)
    # The outermost segments extend half a sample spacing to probability 0 and 1.
    np.testing.assert_allclose(default.values[[0, -1]], [0.5, 10.0])
    limited = NormalScoreTransform.from_samples(values, (0.0, 20.0))
    np.testing.assert_allclose(limited.forward([0.0, 20.0]), [-np.inf, np.inf])
    back, _ = limited.back(np.array([-10.0, 10.0]), np.zeros(2))
    np.testing.assert_allclose(back, [0.0, 20.0], atol=1e-12)
    half_open = NormalScoreTransform.from_samples(values, (0.0, None))
    assert half_open.values[-1] == default.values[-1]


@pytest.mark.parametrize('estimate', [-0.5, 0.0, 1.3])
@pytest.mark.parametrize('variance', [0.05, 0.6])
def test_back_transforms_match_monte_carlo_moments(estimate, variance):
    normal = estimate + np.sqrt(variance) * np.random.default_rng(0).standard_normal(400_000)
    values = np.random.default_rng(1).lognormal(0.0, 1.0, 30)
    for transform in (LogTransform(), NormalScoreTransform.from_samples(values)):
        if isinstance(transform, LogTransform):
            draws = np.exp(normal)
        else:
            draws = np.interp(ndtr(normal), transform.cdf, transform.values)
        mean, var = transform.back(np.array([estimate]), np.array([variance]))
        assert mean[0] == pytest.approx(draws.mean(), rel=0.01)
        assert var[0] == pytest.approx(draws.var(), rel=0.03)


@pytest.mark.parametrize('transform', ['log', 'normal_score'])
def test_predict_back_transforms_and_honors_the_samples(skewed, transform):
    model = skewed.krige.fit('grade', transform=transform, variogram_model='spherical')
    assert model.transform.name == transform
    at_samples = model.predict(skewed)
    np.testing.assert_allclose(at_samples['grade'], skewed['grade'], rtol=1e-5)
    np.testing.assert_allclose(
        at_samples['grade transformed'], model.transform.forward(skewed['grade']), atol=1e-6
    )
    image = pv.ImageData(dimensions=(15, 15, 1), spacing=(10 / 14, 10 / 14, 1))
    result = model.predict(image)
    assert result.point_data.active_scalars_name == 'grade'
    mean, variance = model.transform.back(
        result['grade transformed'], result['grade transformed variance']
    )
    np.testing.assert_allclose(result['grade'], mean)
    np.testing.assert_allclose(result['grade variance'], variance)
    assert np.all(result['grade variance'] >= 0)


def test_log_model_krige_the_logarithms(skewed):
    from pykrige.ok import OrdinaryKriging

    model = skewed.krige.fit(
        'grade', transform='log', variogram_model='spherical', enable_statistics=True
    )
    oracle = OrdinaryKriging(
        *skewed.points[:, :2].T,
        np.log(skewed['grade']),
        variogram_model='spherical',
        enable_statistics=True,
    )
    np.testing.assert_allclose(model.variogram.parameters, oracle.variogram_model_parameters)
    np.testing.assert_allclose(model.statistics[:3], oracle.get_statistics())
    np.testing.assert_allclose(model.samples['grade'], skewed['grade'])


def test_transformed_uncertainty_follows_the_variance_options(skewed):
    model = skewed.krige.fit('grade', transform='log', variogram_model='spherical')
    target = pv.PolyData(np.array([[2.5, 2.5, 0.0], [7.5, 5.0, 0.0]]))
    full = model.predict(target)
    std = model.predict(target, std=True)
    np.testing.assert_allclose(std['grade std'], np.sqrt(full['grade variance']))
    np.testing.assert_allclose(
        std['grade transformed std'], np.sqrt(full['grade transformed variance'])
    )
    bare = model.predict(target, variance=False)
    assert set(bare.point_data.keys()) == {'grade', 'grade transformed'}


def test_max_std_and_where_null_the_transformed_arrays_too(skewed):
    model = skewed.krige.fit('grade', transform='normal_score', variogram_model='spherical')
    target = pv.PolyData(np.array([[5.0, 5.0, 0.0], [50.0, 50.0, 0.0]]))
    std = np.sqrt(model.predict(target)['grade variance'])
    result = model.predict(target, max_std=float(std.mean()), null_value=-1.0)
    np.testing.assert_array_equal(result['vtkValidPointMask'], [1, 0])
    for name in ('grade', 'grade variance', 'grade transformed', 'grade transformed variance'):
        assert result[name][1] == -1.0
        assert result[name][0] != -1.0


def test_cross_validation_errors_are_in_original_units(skewed):
    model = skewed.krige.fit('grade', transform='normal_score', variogram_model='spherical')
    holdout = model.cross_validate('loo')
    for i in (0, 17):
        keep = np.arange(skewed.n_points) != i
        train = pv.PolyData(skewed.points[keep])
        train['grade'] = skewed['grade'][keep]
        refit = train.krige.fit('grade', transform='normal_score', variogram_model='spherical')
        estimate = refit.predict(pv.PolyData(skewed.points[[i]]))['grade'][0]
        assert holdout.samples['grade error'][i] == pytest.approx(estimate - skewed['grade'][i])


def test_with_variogram_keeps_the_transform(skewed):
    model = skewed.krige.fit('grade', transform='log', variogram_model='spherical')
    refit = model.with_variogram('exponential')
    assert refit.transform is model.transform
    direct = skewed.krige.fit('grade', transform='log', variogram_model='exponential')
    target = pv.PolyData(np.array([[2.5, 2.5, 0.0]]))
    np.testing.assert_allclose(refit.predict(target)['grade'], direct.predict(target)['grade'])
    assert refit.cross_validate(5).rmse == pytest.approx(direct.cross_validate(5).rmse)


def test_exceedance_needs_the_model_for_a_transformed_result(skewed):
    model = skewed.krige.fit('grade', transform='log', variogram_model='spherical')
    result = model.predict(pv.PolyData(np.array([[2.5, 2.5, 0.0]])))
    with pytest.raises(ValueError, match='model='):
        pyvista_krige.exceedance(result, 'grade', 1.0)
    with pytest.raises(ValueError, match='variance and std'):
        pyvista_krige.exceedance(result, 'grade', 1.0, model=model, variance='grade variance')
    below_zero = pyvista_krige.exceedance(result, 'grade', -1.0, model=model, name='p')
    assert below_zero['p'][0] == 1.0


def test_exceedance_compares_transformed_threshold_with_transformed_estimate(skewed):
    from scipy.stats import norm

    model = skewed.krige.fit('grade', transform='normal_score', variogram_model='spherical')
    result = model.predict(pv.PolyData(np.array([[2.5, 2.5, 0.0], [8.0, 1.0, 0.0]])))
    prob = pyvista_krige.exceedance(result, 'grade', 1.5, model=model, name='p')
    expected = norm.sf(
        (model.transform.forward(1.5) - result['grade transformed'])
        / np.sqrt(result['grade transformed variance'])
    )
    np.testing.assert_allclose(prob['p'], expected)


def test_add_exceedance_forwards_the_model(skewed):
    model = skewed.krige.fit('grade', transform='log', variogram_model='spherical')
    image = model.predict(pv.ImageData(dimensions=(12, 12, 4), spacing=(1, 1, 1)))
    pl = pv.Plotter()
    with pytest.raises(ValueError, match='model='):
        pl.krige.add_exceedance(image, 'grade', 1.5)
    actor = pl.krige.add_exceedance(image, 'grade', 1.5, model=model)
    assert actor is not None
    pl.close()


@pytest.mark.parametrize(
    ('values', 'kwargs', 'match'),
    [
        ([1.0, 0.0, 2.0], {'transform': 'log'}, 'positive values'),
        ([1.0, -3.0, 2.0], {'transform': 'log'}, 'positive values'),
        ([2.0, 2.0, 2.0], {'transform': 'normal_score'}, 'two distinct values'),
        ([1.0, 2.0, 3.0], {'transform': 'boxcox'}, "None, 'log' or 'normal_score'"),
        ([1.0, 2.0, 3.0], {'transform': 'log', 'transform_limits': (0, 9)}, 'normal_score'),
        ([1.0, 2.0, 3.0], {'transform_limits': (0, 9)}, 'normal_score'),
        (
            [1.0, 2.0, 3.0],
            {'transform': 'normal_score', 'transform_limits': (1.5, 9)},
            'outside the sample range',
        ),
    ],
)
def test_fit_rejects_invalid_transforms(values, kwargs, match):
    samples = pv.PolyData(np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    samples['grade'] = np.array(values)
    with pytest.raises(ValueError, match=match):
        samples.krige.fit('grade', **kwargs)


def test_to_image_forwards_the_transform(skewed):
    image = skewed.krige.to_image(
        'grade', dimensions=(8, 8), transform='log', variogram_model='spherical'
    )
    assert 'grade transformed' in image.point_data
    assert np.nanmin(image['grade']) > 0
