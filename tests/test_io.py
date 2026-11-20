from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

import pyvista_krige


def _grid(nx: int = 4, ny: int = 3, spacing=(2.0, 0.5), origin=(100.0, 200.0)) -> pv.ImageData:
    """A one-layer image whose values encode their own position, with two NaN cells."""
    grid = pv.ImageData(dimensions=(nx, ny, 1), spacing=(*spacing, 1.0), origin=(*origin, 0.0))
    values = np.arange(nx * ny, dtype=float) * 0.25 + 1.0
    values[[1, nx * ny - 2]] = np.nan
    grid['z'] = values
    return grid


def _assert_same(a: pv.ImageData, b: pv.ImageData, name: str = 'z') -> None:
    assert a.dimensions == b.dimensions
    np.testing.assert_allclose(a.origin, b.origin)
    np.testing.assert_allclose(a.spacing, b.spacing)
    np.testing.assert_array_equal(a[name], b[name])


@pytest.mark.parametrize('spacing', [(1.0, 1.0), (2.0, 0.5)])
def test_esri_round_trip(tmp_path: Path, spacing) -> None:
    grid = _grid(spacing=spacing)
    path = tmp_path / 'out.asc'
    pyvista_krige.write_esri_ascii(grid, path, 'z')
    back = pyvista_krige.read_esri_ascii(path, scalars='z')
    _assert_same(grid, back)
    header = path.read_text().lower()
    assert ('cellsize' in header) == (spacing[0] == spacing[1])
    assert ('dx ' in header) == (spacing[0] != spacing[1])


def test_esri_writes_north_first_and_nodata(tmp_path: Path) -> None:
    grid = _grid(nx=2, ny=2, spacing=(1.0, 1.0), origin=(0.0, 0.0))
    grid['z'] = [1.0, np.nan, 3.0, 4.0]  # south row first
    path = tmp_path / 'out.asc'
    pyvista_krige.write_esri_ascii(grid, path, 'z')
    assert path.read_text().splitlines()[-2:] == ['3.0 4.0', '1.0 -9999.0']


def test_esri_corner_and_center_registration(tmp_path: Path) -> None:
    body = '1 2 3\n4 5 6\n'
    corner = tmp_path / 'corner.asc'
    corner.write_text(
        'ncols 3\nnrows 2\nxllcorner 0\nyllcorner 0\ncellsize 10\nNODATA_value -9999\n' + body
    )
    center = tmp_path / 'center.asc'
    center.write_text(
        'ncols 3\nnrows 2\nxllcenter 0\nyllcenter 0\ncellsize 10\nNODATA_value -9999\n' + body
    )
    a = pyvista_krige.read_esri_ascii(corner)
    b = pyvista_krige.read_esri_ascii(center)
    np.testing.assert_allclose(a.origin, (5.0, 5.0, 0.0))
    np.testing.assert_allclose(b.origin, (0.0, 0.0, 0.0))
    np.testing.assert_array_equal(a['values'], b['values'])
    # The same file read both ways differs by half a cell.
    np.testing.assert_allclose(a.points - b.points, [[5.0, 5.0, 0.0]] * a.n_points)
    # North-to-south rows: the first file row is the top of the image.
    np.testing.assert_array_equal(a['values'].reshape(2, 3), [[4, 5, 6], [1, 2, 3]])


def test_esri_dx_dy_and_mixed_case_header(tmp_path: Path) -> None:
    path = tmp_path / 'dxdy.asc'
    path.write_text(
        'NCOLS 2\nNROWS 2\nXLLCORNER 10\nYLLCORNER 20\nDX 4\nDY 2\nNODATA_VALUE -1\n1 -1\n3 4\n'
    )
    grid = pyvista_krige.read_esri_ascii(path)
    np.testing.assert_allclose(grid.spacing, (4.0, 2.0, 1.0))
    np.testing.assert_allclose(grid.origin, (12.0, 21.0, 0.0))
    np.testing.assert_array_equal(grid['values'], [3, 4, 1, np.nan])


def test_esri_matches_pykrige(tmp_path: Path) -> None:
    kriging_tools = pytest.importorskip('pykrige.kriging_tools')
    path = tmp_path / 'dem.asc'
    path.write_text(
        'ncols 4\nnrows 3\nxllcorner 100\nyllcorner 200\ncellsize 5\nNODATA_value -9999\n'
        '1 2 -9999 4\n5 6 7 8\n9 10 11 -9999\n'
    )
    array, x, y, cellsize, nodata = kriging_tools.read_asc_grid(path)
    array = np.where(array == nodata, np.nan, array)
    grid = pyvista_krige.read_esri_ascii(path)
    nx, ny, _ = grid.dimensions
    np.testing.assert_array_equal(grid['values'].reshape(ny, nx), array)
    np.testing.assert_allclose(grid.points[:nx, 0], x)
    np.testing.assert_allclose(grid.points[::nx, 1], y)
    assert grid.spacing[0] == cellsize


def test_esri_reads_pykrige_output(tmp_path: Path) -> None:
    kriging_tools = pytest.importorskip('pykrige.kriging_tools')
    x, y = np.arange(4.0) * 2 + 1, np.arange(3.0) * 3 + 10
    z = np.arange(12.0).reshape(3, 4)
    path = tmp_path / 'pk.asc'
    kriging_tools.write_asc_grid(x, y, z, filename=str(path), style=1)
    grid = pyvista_krige.read_esri_ascii(path)
    np.testing.assert_allclose(grid.spacing, (2.0, 3.0, 1.0))
    np.testing.assert_allclose(grid.origin, (1.0, 10.0, 0.0))
    np.testing.assert_array_equal(grid['values'], z.ravel())


def test_esri_wrapped_rows_and_size_mismatch(tmp_path: Path) -> None:
    path = tmp_path / 'wrapped.asc'
    header = 'ncols 2\nnrows 2\nxllcenter 0\nyllcenter 0\ncellsize 1\nNODATA_value -9999\n'
    path.write_text(header + '1\n2 3\n4\n')
    np.testing.assert_array_equal(pyvista_krige.read_esri_ascii(path)['values'], [3, 4, 1, 2])
    path.write_text(header + '1 2 3\n')
    with pytest.raises(ValueError, match='found 3'):
        pyvista_krige.read_esri_ascii(path)


def test_esri_header_errors(tmp_path: Path) -> None:
    path = tmp_path / 'bad.asc'
    path.write_text('ncols 2\nnrows 1\ncellsize 1\nNODATA_value -9999\n1 2\n')
    with pytest.raises(ValueError, match='xllcorner'):
        pyvista_krige.read_esri_ascii(path)
    path.write_text('ncols 2\nnrows 1\nxllcorner 0\nyllcorner 0\nNODATA_value -9999\n1 2\n')
    with pytest.raises(ValueError, match='cellsize'):
        pyvista_krige.read_esri_ascii(path)


# A 3 column by 2 row grid. Columns run left to right and each runs from the top
# row down, so the data lines read: column 0 = (4, 1), column 1 = (5, null), column 2 = (6, 3).
ZMAP = """\
!
! A hand-written file.
!
@GRID FILE, GRID, 2
  15, 1.0E+30, , 3, 1
  2, 3, 100.0, 120.0, 200.0, 210.0
0.0, 0.0, 0.0
@
      4.000      1.000
      5.000 1.0E+30
! a comment inside the data
      6.000      3.000
"""


def test_zmap_hand_written_file(tmp_path: Path) -> None:
    path = tmp_path / 'surface.zmap'
    path.write_text(ZMAP)
    grid = pyvista_krige.read_zmap(path)
    assert grid.dimensions == (3, 2, 1)
    np.testing.assert_allclose(grid.origin, (100.0, 200.0, 0.0))
    np.testing.assert_allclose(grid.spacing, (10.0, 10.0, 1.0))
    # Rows from the south: (1, nan, 3) then (4, 5, 6).
    np.testing.assert_array_equal(grid['values'], [1, np.nan, 3, 4, 5, 6])
    # The top-left node (x = 100, y = 210) holds the first number in the file.
    top_left = grid.find_closest_point((100.0, 210.0, 0.0))
    assert grid['values'][top_left] == 4.0


def test_zmap_fixed_width_values_without_separators(tmp_path: Path) -> None:
    path = tmp_path / 'tight.zmap'
    path.write_text(
        '@GRID FILE, GRID, 2\n'
        '5, -999.0, , 1, 1\n'
        '2, 2, 0.0, 1.0, 0.0, 1.0\n'
        '0.0, 0.0, 0.0\n'
        '@\n'
        '-12.3-14.5\n'
        '-67.8 -1.5\n'
    )
    grid = pyvista_krige.read_zmap(path)
    np.testing.assert_array_equal(grid['values'], [-14.5, -1.5, -12.3, -67.8])


def test_zmap_implied_decimals(tmp_path: Path) -> None:
    path = tmp_path / 'implied.zmap'
    path.write_text(
        '@GRID FILE, GRID, 2\n'
        '6, -999.0, , 2, 1\n'
        '2, 2, 0.0, 1.0, 0.0, 1.0\n'
        '0.0, 0.0, 0.0\n'
        '@\n'
        '  1234   -50\n'
        '   300 -999.0\n'
    )
    grid = pyvista_krige.read_zmap(path)
    np.testing.assert_allclose(grid['values'], [-0.5, np.nan, 12.34, 3.0])


def test_zmap_round_trip(tmp_path: Path) -> None:
    grid = _grid(nx=4, ny=7, spacing=(2.0, 0.5))
    path = tmp_path / 'out.zmap'
    pyvista_krige.write_zmap(grid, path, 'z')
    back = pyvista_krige.read_zmap(path, scalars='z')
    _assert_same(grid, back)


def test_zmap_writer_layout(tmp_path: Path) -> None:
    grid = pv.ImageData(dimensions=(2, 3, 1), spacing=(10, 10, 1), origin=(0, 0, 0))
    grid['z'] = [1.0, 2.0, 3.0, 4.0, np.nan, 6.0]  # south row first
    path = tmp_path / 'out.zmap'
    pyvista_krige.write_zmap(grid, path, 'z', decimals=1)
    lines = path.read_text().splitlines()
    assert lines[3] == '@GRID FILE, GRID, 5'
    assert lines[5] == '3, 2, 0.0, 10.0, 0.0, 20.0'
    assert lines[7] == '@'
    # The top row is (nan, 6): column 0 runs null, 3, 1 and column 1 runs 6, 4, 2.
    assert [float(v) for v in lines[8].split()] == [-9999.0, 3.0, 1.0]
    assert [float(v) for v in lines[9].split()] == [6.0, 4.0, 2.0]


def test_zmap_wraps_long_columns(tmp_path: Path) -> None:
    grid = _grid(nx=2, ny=12)
    path = tmp_path / 'long.zmap'
    pyvista_krige.write_zmap(grid, path, 'z')
    data = path.read_text().splitlines()[8:]
    assert [len(line.split()) for line in data] == [5, 5, 2, 5, 5, 2]


def test_zmap_rejects_other_grid_types(tmp_path: Path) -> None:
    path = tmp_path / 'points.zmap'
    path.write_text('@POINTS, POINT, 3\n15, 1e30, , 3, 1\n1, 1, 0, 1, 0, 1\n0, 0, 0\n@\n')
    with pytest.raises(ValueError, match='GRID'):
        pyvista_krige.read_zmap(path)


@pytest.mark.parametrize('write', [pyvista_krige.write_esri_ascii, pyvista_krige.write_zmap])
def test_writers_reject_bad_input(tmp_path: Path, write) -> None:
    path = tmp_path / 'bad'
    volume = pv.ImageData(dimensions=(3, 3, 3))
    volume['z'] = np.arange(27.0)
    with pytest.raises(ValueError, match='single-layer'):
        write(volume, path, 'z')

    grid = _grid()
    with pytest.raises(KeyError, match='missing'):
        write(grid, path, 'missing')
    with pytest.raises(TypeError, match='ImageData'):
        write(grid.cast_to_structured_grid(), path, 'z')

    grid['vec'] = np.zeros((grid.n_points, 3))
    with pytest.raises(ValueError, match='one component'):
        write(grid, path, 'vec')

    grid['clash'] = np.full(grid.n_points, -9999.0)
    with pytest.raises(ValueError, match='-9999'):
        write(grid, path, 'clash')

    rotated = _grid()
    rotated.direction_matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match='axis-aligned'):
        write(rotated, path, 'z')
    assert not path.exists()
