"""Read and write ESRI ASCII and ZMAP Plus grids."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from numpy.typing import NDArray

_ESRI_KEYS = {
    'ncols',
    'nrows',
    'xllcorner',
    'xllcenter',
    'yllcorner',
    'yllcenter',
    'cellsize',
    'cell_size',
    'dx',
    'dy',
    'nodata_value',
    'nodatavalue',
}
_ESRI_DEFAULT_NODATA = -9999.0
_ZMAP_NODES_PER_LINE = 5
_ZMAP_MAX_WIDTH = 39


def _read_text(path: str | os.PathLike[str]) -> list[str]:
    with open(path, encoding='utf-8-sig') as f:
        return f.read().splitlines()


def _make_image(
    values: NDArray[np.floating],
    origin: tuple[float, float],
    spacing: tuple[float, float],
    scalars: str,
) -> pv.ImageData:
    """Return an image of ``values`` (rows from the bottom) with points at the nodes."""
    ny, nx = values.shape
    image = pv.ImageData(
        dimensions=(nx, ny, 1),
        spacing=(spacing[0], spacing[1], 1.0),
        origin=(origin[0], origin[1], 0.0),
    )
    image.point_data[scalars] = values.ravel()
    return image


def _single_layer(grid: pv.DataSet, scalars: str) -> NDArray[np.number]:
    """Return array ``scalars`` of a one-layer image as ``(ny, nx)``, rows from the bottom."""
    if not isinstance(grid, pv.ImageData):
        msg = f'Expected a pyvista.ImageData, got {type(grid).__name__}.'
        raise TypeError(msg)
    nx, ny, nz = grid.dimensions
    if nz != 1:
        msg = f'Expected a single-layer image with dimensions[2] == 1, got {grid.dimensions}.'
        raise ValueError(msg)
    if not np.allclose(np.asarray(grid.direction_matrix), np.eye(3)):
        msg = 'The image must be axis-aligned (its direction matrix must be the identity).'
        raise ValueError(msg)
    if min(grid.spacing[:2]) <= 0:
        msg = f'The image spacing must be positive, got {grid.spacing}.'
        raise ValueError(msg)
    if scalars not in grid.point_data:
        msg = f"Point array '{scalars}' not found. Available: {list(grid.point_data)}."
        raise KeyError(msg)
    values = np.asarray(grid.point_data[scalars])
    if values.ndim != 1:
        msg = f"Point array '{scalars}' must have one component, got shape {values.shape}."
        raise ValueError(msg)
    return values.reshape(ny, nx)


def _write_text(path: str | os.PathLike[str], text: str) -> None:
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


# ESRI ASCII


def read_esri_ascii(path: str | os.PathLike[str], *, scalars: str = 'values') -> pv.ImageData:
    r"""Read an ESRI ASCII grid (``.asc``) into an image.

    Values are point data at the cell centers. ``NODATA_value`` cells become NaN.

    Parameters
    ----------
    path : str or os.PathLike
        Path of the ``.asc`` file.

    scalars : str, default: 'values'
        Name of the point array that holds the grid values.

    Returns
    -------
    pyvista.ImageData
        A one-layer image whose points are the cell centers. The header may give
        ``xllcorner`` and ``yllcorner`` (the lower-left cell corner) or ``xllcenter``
        and ``yllcenter`` (the lower-left cell center), and either ``cellsize`` or
        ``dx`` and ``dy``. Rows are stored from north to south in the file and are
        flipped here, so the image's first row is the southernmost.

    Examples
    --------
    >>> import pyvista_krige
    >>> import tempfile
    >>> from pathlib import Path
    >>> text = (
    ...     'ncols 3\nnrows 2\nxllcorner 0\nyllcorner 0\ncellsize 10\n'
    ...     'NODATA_value -9999\n1 2 3\n4 -9999 6\n'
    ... )
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / 'dem.asc'
    ...     _ = path.write_text(text)
    ...     grid = pyvista_krige.read_esri_ascii(path)
    >>> grid.origin  # the lower-left cell center
    (5.0, 5.0, 0.0)
    >>> grid['values'].reshape(2, 3)  # rows from the south
    pyvista_ndarray([[ 4., nan,  6.],
                     [ 1.,  2.,  3.]])

    """
    lines = _read_text(path)
    header: dict[str, float] = {}
    first_data = len(lines)
    for i, line in enumerate(lines):
        parts = line.split()
        if not parts:
            continue
        key = parts[0].lower()
        if key not in _ESRI_KEYS:
            first_data = i
            break
        if len(parts) != 2:
            msg = f'Bad ESRI ASCII header line: {line!r}.'
            raise ValueError(msg)
        header[key] = float(parts[1])

    for key in ('ncols', 'nrows'):
        if key not in header:
            msg = f'ESRI ASCII header is missing {key!r}.'
            raise ValueError(msg)
    ncols, nrows = int(header['ncols']), int(header['nrows'])

    if 'dx' in header and 'dy' in header:
        dx, dy = header['dx'], header['dy']
    elif 'cellsize' in header or 'cell_size' in header:
        dx = dy = header.get('cellsize', header.get('cell_size'))
    else:
        msg = "ESRI ASCII header needs 'cellsize' or both 'dx' and 'dy'."
        raise ValueError(msg)

    if 'xllcorner' in header and 'yllcorner' in header:
        x0, y0 = header['xllcorner'] + dx / 2, header['yllcorner'] + dy / 2
    elif 'xllcenter' in header and 'yllcenter' in header:
        x0, y0 = header['xllcenter'], header['yllcenter']
    else:
        msg = (
            "ESRI ASCII header needs 'xllcorner' and 'yllcorner', or 'xllcenter' and 'yllcenter'."
        )
        raise ValueError(msg)

    nodata = header.get('nodata_value', header.get('nodatavalue', _ESRI_DEFAULT_NODATA))
    # The body is read as one token stream, so row wrapping does not matter.
    body = ' '.join(lines[first_data:]).split()
    if len(body) != nrows * ncols:
        msg = f'Header says {nrows} x {ncols} = {nrows * ncols} values, found {len(body)}.'
        raise ValueError(msg)
    values = np.array(body, dtype=float).reshape(nrows, ncols)
    values[values == nodata] = np.nan
    return _make_image(values[::-1], (x0, y0), (dx, dy), scalars)


def write_esri_ascii(
    grid: pv.ImageData,
    path: str | os.PathLike[str],
    scalars: str,
    *,
    nodata: float = _ESRI_DEFAULT_NODATA,
) -> None:
    """Write a point array of a one-layer image as an ESRI ASCII grid (``.asc``).

    The points are written as cell centers (``xllcenter``, ``yllcenter``). A square
    grid gets ``cellsize`` and any other gets ``dx`` and ``dy``. NaN becomes ``nodata``.

    Parameters
    ----------
    grid : pyvista.ImageData
        An axis-aligned image with ``dimensions[2] == 1``.

    path : str or os.PathLike
        Path of the ``.asc`` file to write.

    scalars : str
        Name of a one-component point array.

    nodata : float, default: -9999.0
        Value written for NaN. Raises ``ValueError`` if a real value equals it.

    Examples
    --------
    >>> import numpy as np
    >>> import pyvista as pv
    >>> import pyvista_krige
    >>> import tempfile
    >>> from pathlib import Path
    >>> grid = pv.ImageData(dimensions=(3, 2, 1), spacing=(10, 5, 1))
    >>> grid['z'] = np.arange(6.0)
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / 'out.asc'
    ...     pyvista_krige.write_esri_ascii(grid, path, 'z')
    ...     print(path.read_text())
    ncols 3
    nrows 2
    xllcenter 0.0
    yllcenter 0.0
    dx 10.0
    dy 5.0
    NODATA_value -9999.0
    3.0 4.0 5.0
    0.0 1.0 2.0
    <BLANKLINE>

    """
    values = _single_layer(grid, scalars)
    nan = np.isnan(values) if values.dtype.kind == 'f' else np.zeros(values.shape, dtype=bool)
    if np.any(values[~nan] == nodata):
        msg = f'The array holds the nodata value {nodata}. Choose another with nodata=.'
        raise ValueError(msg)
    ny, nx = values.shape
    dx, dy = grid.spacing[:2]
    x0, y0 = grid.origin[:2]
    header = [f'ncols {nx}', f'nrows {ny}', f'xllcenter {x0!r}', f'yllcenter {y0!r}']
    header += [f'cellsize {dx!r}'] if dx == dy else [f'dx {dx!r}', f'dy {dy!r}']
    header.append(f'NODATA_value {float(nodata)!r}')
    # astype(str) gives the shortest text that reads back to the same value.
    text = np.where(nan, str(float(nodata)), values.astype(str))
    rows = [' '.join(row) for row in text[::-1]]
    _write_text(path, '\n'.join([*header, *rows]) + '\n')


# ZMAP Plus


def _zmap_fields(line: str, count: int, what: str) -> list[str]:
    fields = [f.strip() for f in line.split(',')]
    if len(fields) != count:
        msg = f'ZMAP {what} line needs {count} comma-separated fields: {line!r}.'
        raise ValueError(msg)
    return fields


def read_zmap(path: str | os.PathLike[str], *, scalars: str = 'values') -> pv.ImageData:
    """Read a ZMAP Plus grid into an image.

    Null values become NaN.

    Parameters
    ----------
    path : str or os.PathLike
        Path of the ZMAP file. Only ``GRID`` files are supported.

    scalars : str, default: 'values'
        Name of the point array that holds the grid values.

    Returns
    -------
    pyvista.ImageData
        A one-layer image. The header's minimum and maximum x and y are the
        coordinates of the outer nodes, so the values are point data at those
        nodes and the spacing is ``(xmax - xmin) / (ncols - 1)``. GDAL reads the
        same bounds as cell edges by default, which shifts the grid by half a cell.
        The Landmark description of the format calls them node values, so this
        reader follows that.

    Notes
    -----
    The values are fixed-width fields stored column by column, starting at the
    upper-left node and running down each column. Lines starting with ``!`` are
    comments. A value written without a decimal point is read as an integer
    scaled by the header's decimal count.

    Examples
    --------
    >>> import pyvista_krige
    >>> import tempfile
    >>> from pathlib import Path
    >>> text = '''!
    ... @GRID FILE, GRID, 4
    ...   20, -9999.0, , 1, 1
    ...   3, 2, 0.0, 10.0, 0.0, 20.0
    ... 0.0, 0.0, 0.0
    ... @
    ...   1.0  2.0  3.0
    ...   4.0  -9999.0  6.0
    ... '''
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / 'surface.zmap'
    ...     _ = path.write_text(text)
    ...     grid = pyvista_krige.read_zmap(path)
    >>> grid.dimensions, grid.spacing
    ((2, 3, 1), (10.0, 10.0, 1.0))
    >>> grid['values'].reshape(3, 2)  # rows from the south
    pyvista_ndarray([[ 3.,  6.],
                     [ 2., nan],
                     [ 1.,  4.]])

    """
    lines = [line for line in _read_text(path) if not line.startswith('!')]
    if len(lines) < 5 or not lines[0].startswith('@'):
        msg = 'Not a ZMAP Plus grid: no @ header line after the comments.'
        raise ValueError(msg)
    first = _zmap_fields(lines[0], 3, 'first header')
    if not first[1].upper().startswith('GRID'):
        msg = f"Only ZMAP 'GRID' files are supported, got {first[1]!r}."
        raise ValueError(msg)
    width_s, null_s, null_text, decimals_s, start_s = _zmap_fields(lines[1], 5, 'second header')
    nrows_s, ncols_s, xmin_s, xmax_s, ymin_s, ymax_s = _zmap_fields(lines[2], 6, 'third header')
    if not lines[4].startswith('@'):
        msg = 'ZMAP header must end with a line holding @.'
        raise ValueError(msg)

    width, decimals, start = int(width_s), int(decimals_s), int(start_s)
    nrows, ncols = int(nrows_s), int(ncols_s)
    xmin, xmax, ymin, ymax = (float(s) for s in (xmin_s, xmax_s, ymin_s, ymax_s))
    if nrows < 2 or ncols < 2:
        msg = f'A ZMAP grid needs at least 2 rows and 2 columns, got {nrows} x {ncols}.'
        raise ValueError(msg)
    null = float(null_s or null_text)

    body = [line for line in lines[5:] if line.strip()]
    tokens = ' '.join(body).split()
    if len(tokens) != nrows * ncols:
        # Wide values can run together, so retry by slicing the fixed-width fields.
        tokens = [
            line[i : i + width].strip()
            for line in body
            for i in range(start - 1, len(line), width)
        ]
        tokens = [t for t in tokens if t]
    if len(tokens) != nrows * ncols:
        msg = f'Header says {nrows} x {ncols} = {nrows * ncols} values, found {len(tokens)}.'
        raise ValueError(msg)

    # Digits without a decimal point carry an implied decimal point.
    values = np.array(
        [float(t) if '.' in t or 'e' in t.lower() else float(t) / 10**decimals for t in tokens]
    )
    raw = np.array([float(t) for t in tokens])
    values[raw == null] = np.nan
    # Columns left to right, each one from the top row down, so flip to rows from the south.
    values = values.reshape(ncols, nrows).T[::-1]
    spacing = ((xmax - xmin) / (ncols - 1), (ymax - ymin) / (nrows - 1))
    return _make_image(values, (xmin, ymin), spacing, scalars)


def write_zmap(
    grid: pv.ImageData,
    path: str | os.PathLike[str],
    scalars: str,
    *,
    null_value: float = -9999.0,
    decimals: int = 7,
) -> None:
    """Write a point array of a one-layer image as a ZMAP Plus grid.

    The points are written as nodes, so the header bounds are the outer point
    coordinates. NaN becomes ``null_value``.

    Parameters
    ----------
    grid : pyvista.ImageData
        An axis-aligned image with ``dimensions[2] == 1`` and at least 2 points
        along x and y.

    path : str or os.PathLike
        Path of the ZMAP file to write.

    scalars : str
        Name of a one-component point array.

    null_value : float, default: -9999.0
        Value written for NaN. Raises ``ValueError`` if a real value equals it.

    decimals : int, default: 7
        Digits after the decimal point. Values are rounded to this many.

    Examples
    --------
    >>> import numpy as np
    >>> import pyvista as pv
    >>> import pyvista_krige
    >>> import tempfile
    >>> from pathlib import Path
    >>> grid = pv.ImageData(dimensions=(2, 3, 1), spacing=(10, 10, 1))
    >>> grid['z'] = np.arange(6.0)
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / 'out.zmap'
    ...     pyvista_krige.write_zmap(grid, path, 'z', decimals=1)
    ...     print(path.read_text())
    !
    ! Created by pyvista-krige.
    !
    @GRID FILE, GRID, 5
    4, -9999.0, , 1, 1
    3, 2, 0.0, 10.0, 0.0, 20.0
    0.0, 0.0, 0.0
    @
     4.0 2.0 0.0
     5.0 3.0 1.0
    <BLANKLINE>

    """
    values = _single_layer(grid, scalars)
    ny, nx = values.shape
    if nx < 2 or ny < 2:
        msg = f'A ZMAP grid needs at least 2 points along x and y, got {nx} x {ny}.'
        raise ValueError(msg)
    if decimals < 1:
        msg = 'decimals must be at least 1.'
        raise ValueError(msg)
    nan = np.isnan(values) if values.dtype.kind == 'f' else np.zeros(values.shape, dtype=bool)
    if np.any(values[~nan] == null_value):
        msg = f'The array holds the null value {null_value}. Choose another with null_value=.'
        raise ValueError(msg)

    # Top row first, then one column at a time from the left.
    columns = np.where(nan, null_value, values.astype(float))[::-1].T
    text = np.char.mod(f'%.{decimals}f', columns)
    # One space of padding keeps the fields apart for readers that split on whitespace.
    width = max(len(max(text.ravel().tolist(), key=len)), 1) + 1
    if width > _ZMAP_MAX_WIDTH:
        msg = f'Values need {width} characters, more than the {_ZMAP_MAX_WIDTH} ZMAP allows.'
        raise ValueError(msg)
    text = np.char.rjust(text, width)

    x0, y0 = grid.origin[:2]
    dx, dy = grid.spacing[:2]
    header = [
        '!',
        '! Created by pyvista-krige.',
        '!',
        f'@GRID FILE, GRID, {_ZMAP_NODES_PER_LINE}',
        f'{width}, {float(null_value)!r}, , {decimals}, 1',
        f'{ny}, {nx}, {x0!r}, {x0 + dx * (nx - 1)!r}, {y0!r}, {y0 + dy * (ny - 1)!r}',
        '0.0, 0.0, 0.0',
        '@',
    ]
    body = [
        ''.join(column[i : i + _ZMAP_NODES_PER_LINE])
        for column in text
        for i in range(0, ny, _ZMAP_NODES_PER_LINE)
    ]
    _write_text(path, '\n'.join([*header, *body]) + '\n')
