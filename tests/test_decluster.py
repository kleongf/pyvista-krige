from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

from pyvista_krige import DeclusterScan

NAME = 'declustering weight'


def _polydata(xyz: np.ndarray, **arrays: np.ndarray) -> pv.PolyData:
    samples = pv.PolyData(xyz)
    for key, value in arrays.items():
        samples[key] = value
    return samples


def _regular_plus_cluster(*, cluster_value: float) -> tuple[pv.PolyData, np.ndarray]:
    """10 x 10 regular samples worth 1, plus 60 tightly clustered ones."""
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.linspace(0, 100, 10), np.linspace(0, 100, 10))
    regular = np.column_stack([gx.ravel(), gy.ravel()])
    cluster = rng.uniform(40, 50, (60, 2))
    xy = np.vstack([regular, cluster])
    values = np.r_[np.ones(100), np.full(60, cluster_value)]
    return _polydata(np.column_stack([xy, np.zeros(len(xy))]), value=values), regular


def test_uniform_grid_gets_equal_weights():
    gx, gy = np.meshgrid(np.arange(10.0), np.arange(10.0))
    samples = _polydata(np.column_stack([gx.ravel(), gy.ravel(), np.zeros(100)]))
    # Cells no larger than the spacing hold one sample, or all of them hold all.
    for cell in (0.5, 1.0):
        np.testing.assert_allclose(samples.krige.decluster(cell_size=cell)[NAME], 1.0)


def test_uniform_grid_with_larger_cells_is_nearly_unbiased():
    # Cells that straddle the grid edge hold fewer samples, so single weights vary
    # (about 0.6 to 2.5 here), but a trend across the grid keeps its mean.
    gx, gy = np.meshgrid(np.arange(10.0), np.arange(10.0))
    samples = _polydata(np.column_stack([gx.ravel(), gy.ravel(), np.zeros(100)]), x=gx.ravel())
    weights = samples.krige.decluster(cell_size=2.5)[NAME]
    declustered = np.average(samples['x'], weights=weights)
    assert abs(declustered - samples['x'].mean()) < 0.15


def test_cluster_lowers_weights_and_the_mean_moves_back():
    samples, _ = _regular_plus_cluster(cluster_value=3.0)
    weighted = samples.krige.decluster(cell_size=12.0)
    weights = weighted[NAME]
    # Regular samples keep more weight than the ones packed into the cluster.
    assert weights[:100].mean() > 1 > weights[100:].mean()
    naive = samples['value'].mean()
    declustered = np.average(samples['value'], weights=weights)
    assert abs(declustered - 1.0) < abs(naive - 1.0) / 2


def test_weights_sum_to_n():
    samples, _ = _regular_plus_cluster(cluster_value=3.0)
    for cell in (1.0, 12.0, 80.0, (5.0, 30.0)):
        weights = samples.krige.decluster(cell_size=cell)[NAME]
        assert weights.sum() == pytest.approx(samples.n_points)


def test_weights_do_not_depend_on_sample_order():
    samples, _ = _regular_plus_cluster(cluster_value=3.0)
    order = np.random.default_rng(5).permutation(samples.n_points)
    shuffled = _polydata(samples.points[order], value=samples['value'][order])
    expected = samples.krige.decluster(cell_size=12.0)[NAME][order]
    np.testing.assert_allclose(shuffled.krige.decluster(cell_size=12.0)[NAME], expected)
    scan = samples.krige.decluster_scan('value')
    shuffled_scan = shuffled.krige.decluster_scan('value')
    np.testing.assert_allclose(shuffled_scan.means, scan.means)
    assert shuffled_scan.best == scan.best


def test_offsets_remove_dependence_on_the_grid_origin():
    # Two points a hair apart share a cell for most grid origins but not all.
    xyz = np.array([[0.0, 0.0, 0], [0.9, 0, 0], [5, 5, 0], [9, 1, 0], [2, 8, 0]])
    samples = _polydata(xyz)
    single = samples.krige.decluster(cell_size=1.0, n_offsets=1)[NAME]
    averaged = samples.krige.decluster(cell_size=1.0, n_offsets=10)[NAME]
    assert single[0] == single[1]
    assert averaged[0] != single[0]


def test_returns_a_copy_and_records_the_cell_size():
    samples, _ = _regular_plus_cluster(cluster_value=3.0)
    weighted = samples.krige.decluster(cell_size=12.0)
    assert NAME not in samples.array_names
    assert weighted is not samples
    np.testing.assert_array_equal(weighted.field_data['declustering cell size'], [12.0, 12.0])
    renamed = samples.krige.decluster(cell_size=12.0, name='w')
    assert 'w' in renamed.point_data


def test_3d_anisotropic_cell():
    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 100, (60, 2))
    # Thin layers 1 apart, wide apart horizontally: only a flat cell separates them.
    z = rng.integers(0, 4, 60).astype(float)
    samples = _polydata(np.column_stack([xy, z]))
    flat = samples.krige.decluster(cell_size=(50.0, 50.0, 0.5))[NAME]
    tall = samples.krige.decluster(cell_size=(50.0, 50.0, 50.0))[NAME]
    assert flat.sum() == pytest.approx(60)
    assert tall.sum() == pytest.approx(60)
    # A thin dz splits samples by layer, so fewer share a cell than with a cube.
    assert flat.std() < tall.std()
    np.testing.assert_array_equal(
        samples.krige.decluster(cell_size=(50.0, 50.0, 0.5)).field_data['declustering cell size'],
        [50.0, 50.0, 0.5],
    )


def test_scalar_cell_size_is_a_cube_for_3d_samples(samples3d):
    cube = samples3d.krige.decluster(cell_size=4.0)[NAME]
    explicit = samples3d.krige.decluster(cell_size=(4.0, 4.0, 4.0))[NAME]
    np.testing.assert_array_equal(cube, explicit)


def test_flat_samples_ignore_z_for_a_scalar_cell_size(samples2d):
    weights = samples2d.krige.decluster(cell_size=2.0)[NAME]
    explicit = samples2d.krige.decluster(cell_size=(2.0, 2.0))[NAME]
    np.testing.assert_array_equal(weights, explicit)


def test_scan_picks_a_cell_size_that_lowers_the_mean():
    samples, _ = _regular_plus_cluster(cluster_value=3.0)
    scan = samples.krige.decluster_scan('value')
    assert isinstance(scan, DeclusterScan)
    assert scan.cell_sizes.shape == (20, 2)
    assert scan.means.shape == (20,)
    assert scan.means[scan.best] == scan.means.min()
    assert scan.means[scan.best] < samples['value'].mean() - 0.3
    # The lowest mean is an interior minimum, between the scan's ends.
    assert 0 < scan.best < len(scan.means) - 1
    weighted = samples.krige.decluster(scalars='value')
    np.testing.assert_allclose(weighted.field_data['declustering cell size'], scan.cell_size)
    expected = samples.krige.decluster(cell_size=scan.cell_size)[NAME]
    np.testing.assert_allclose(weighted[NAME], expected)


def test_maximize_for_samples_clustered_in_low_values():
    samples, _ = _regular_plus_cluster(cluster_value=-1.0)
    with pytest.warns(UserWarning, match='no interior extremum'):
        samples.krige.decluster_scan('value')
    scan = samples.krige.decluster_scan('value', maximize=True)
    assert scan.means[scan.best] == scan.means.max()
    assert scan.means[scan.best] > samples['value'].mean() + 0.3


def test_scan_in_3d_scales_the_cell_by_the_aspect(samples3d):
    scan = samples3d.krige.decluster_scan('temperature', aspect=(1, 1, 0.5), size_range=(1, 5))
    assert scan.cell_sizes.shape == (20, 3)
    np.testing.assert_allclose(scan.cell_sizes[:, 2], 0.5 * scan.cell_sizes[:, 0])
    assert scan.cell_sizes[0, 0] == pytest.approx(1)
    assert scan.cell_sizes[-1, 0] == pytest.approx(5)


def test_scan_ignores_nan_values():
    samples, _ = _regular_plus_cluster(cluster_value=3.0)
    values = samples['value'].copy()
    values[3] = np.nan
    samples['value'] = values
    scan = samples.krige.decluster_scan('value')
    assert np.isfinite(scan.means).all()


def test_cell_data_scalars_use_cell_centers_and_write_cell_data():
    grid = pv.ImageData(dimensions=(5, 5, 1))
    grid.cell_data['value'] = np.arange(grid.n_cells, dtype=float)
    weighted = grid.krige.decluster(cell_size=0.5, scalars='value')
    assert NAME in weighted.cell_data
    assert NAME not in weighted.point_data
    np.testing.assert_allclose(weighted.cell_data[NAME], 1.0)


def test_invalid_arguments(samples2d):
    with pytest.raises(ValueError, match='needs scalars'):
        samples2d.krige.decluster()
    with pytest.raises(KeyError, match='nope'):
        samples2d.krige.decluster(scalars='nope')
    with pytest.raises(ValueError, match='positive'):
        samples2d.krige.decluster(cell_size=0.0)
    with pytest.raises(ValueError, match='positive'):
        samples2d.krige.decluster(cell_size=(1.0, -1.0))
    with pytest.raises(ValueError, match='2- or 3-tuple'):
        samples2d.krige.decluster(cell_size=(1.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match='n_offsets'):
        samples2d.krige.decluster(cell_size=1.0, n_offsets=0)
    with pytest.raises(ValueError, match='size_range'):
        samples2d.krige.decluster_scan('elevation', size_range=(5.0, 1.0))
    with pytest.raises(ValueError, match='n_sizes'):
        samples2d.krige.decluster_scan('elevation', n_sizes=1)


def test_identical_points_need_an_explicit_size_range():
    samples = _polydata(np.zeros((5, 3)), value=np.arange(5.0))
    with pytest.raises(ValueError, match='size_range'):
        samples.krige.decluster_scan('value')


@pytest.mark.parametrize('spacing', [0.1, 0.3, 0.7, 1.1, 3.3, 7.7])
def test_lattice_with_cell_equal_to_spacing_is_not_split_by_round_off(spacing):
    # k * spacing / spacing can come out just below k, which would merge two cells.
    idx = np.arange(8)
    gx, gy, gz = np.meshgrid(idx, idx, idx[:4])
    xyz = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]) * spacing
    weights = _polydata(xyz).krige.decluster(cell_size=spacing)[NAME]
    np.testing.assert_allclose(weights, 1.0, rtol=1e-9)
