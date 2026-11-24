"""Regular grids and compressed tiles of gridded values.

A gridded variable at one resolution and one time point is a dense 2D array
(latitude, longitude). Instead of one database row per value, the array is cut
into square tiles of `tile_size` × `tile_size` cells, and each tile is stored
as one row holding the raw float32 values. The grid points themselves are
implicit: cell (i, j) of a grid lies at
(lat0 + i * step, lon0 + j * step).

NaN means "no data" (e.g. ocean), 0 means 0. Tiles that are entirely NaN are
not stored. Edge tiles are padded with NaN.

This module is pure numpy and knows nothing about the database.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

# float32, little endian, row-major (latitude, then longitude)
TILE_DTYPE = np.dtype("<f4")
DEFAULT_TILE_SIZE = 100
# tolerance (in degrees) when matching a coordinate to a grid cell
# coordinates are stored rounded to 4 decimal places elsewhere
COORD_ATOL = 0.5e-4


@dataclass(frozen=True)
class GridDef:
    """A regular latitude/longitude grid, both axes ascending.

    Attributes:
        lat0 (float): Latitude of row 0 (the southernmost row).
        lon0 (float): Longitude of column 0 (the westernmost column).
        step (float): Grid spacing in degrees, the same on both axes.
        n_lat (int): Number of rows.
        n_lon (int): Number of columns.
        tile_size (int): Cells per tile side.
    """

    lat0: float
    lon0: float
    step: float
    n_lat: int
    n_lon: int
    tile_size: int = DEFAULT_TILE_SIZE

    def __post_init__(self):
        if not self.step > 0:
            raise ValueError("The grid step must be > 0.")
        if self.n_lat < 1 or self.n_lon < 1:
            raise ValueError("A grid needs at least one row and one column.")
        if self.tile_size < 1:
            raise ValueError("The tile size must be >= 1.")

    @property
    def n_tile_rows(self) -> int:
        return -(-self.n_lat // self.tile_size)

    @property
    def n_tile_cols(self) -> int:
        return -(-self.n_lon // self.tile_size)

    def latitudes(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Latitudes of rows `start` to `stop` (exclusive)."""
        stop = self.n_lat if stop is None else stop
        return self.lat0 + np.arange(start, stop, dtype=np.float64) * self.step

    def longitudes(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Longitudes of columns `start` to `stop` (exclusive)."""
        stop = self.n_lon if stop is None else stop
        return self.lon0 + np.arange(start, stop, dtype=np.float64) * self.step


def _axis_step(coords: np.ndarray, step: float | None) -> float:
    """Spacing of an ascending regular axis, or `step` if given and consistent."""
    if len(coords) < 2:
        if step is None:
            raise ValueError("Need at least two coordinate values or an explicit step.")
        return float(step)
    diffs = np.diff(coords)
    found = float(np.mean(diffs))
    if found <= 0:
        raise ValueError("Coordinates must be ascending.")
    if not np.allclose(diffs, found, atol=COORD_ATOL):
        raise ValueError("Coordinates are not regularly spaced.")
    if step is not None and not np.isclose(found, step, atol=COORD_ATOL):
        raise ValueError(f"Coordinate spacing {found} does not match step {step}.")
    return float(step) if step is not None else found


def grid_def_from_coords(
    lats: np.ndarray,
    lons: np.ndarray,
    step: float | None = None,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> GridDef:
    """Describe the regular grid spanned by ascending `lats` and `lons`.

    Args:
        lats (np.ndarray): Ascending, regularly spaced latitudes.
        lons (np.ndarray): Ascending, regularly spaced longitudes.
        step (float | None): Grid spacing. Needed if an axis has a single
            value; otherwise checked against the coordinates. Defaults to None.
        tile_size (int): Cells per tile side.

    Returns:
        GridDef: The grid.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    if len(lats) == 0 or len(lons) == 0:
        raise ValueError("Empty coordinates.")
    lat_step = _axis_step(lats, step)
    lon_step = _axis_step(lons, step if len(lons) > 1 else lat_step)
    if not np.isclose(lat_step, lon_step, atol=COORD_ATOL):
        raise ValueError("Latitude and longitude spacing differ.")
    return GridDef(
        lat0=float(lats[0]),
        lon0=float(lons[0]),
        step=lat_step,
        n_lat=len(lats),
        n_lon=len(lons),
        tile_size=tile_size,
    )


def cell_index(grid: GridDef, lat: float, lon: float) -> tuple[int, int] | None:
    """Row and column of the grid cell at exactly (`lat`, `lon`).

    Returns:
        tuple[int, int] | None: (i, j), or None if the point is not a grid
            point of `grid` (off the grid or between grid points).
    """
    fi = (lat - grid.lat0) / grid.step
    fj = (lon - grid.lon0) / grid.step
    i, j = round(fi), round(fj)
    if not (0 <= i < grid.n_lat and 0 <= j < grid.n_lon):
        return None
    if abs(i - fi) * grid.step > COORD_ATOL or abs(j - fj) * grid.step > COORD_ATOL:
        return None
    return i, j


def _axis_range(start: float, n: int, step: float, low: float, high: float):
    """Half-open index range of the axis values within [low, high]."""
    first = max(0, int(np.ceil((low - start) / step - COORD_ATOL / step)))
    stop = min(n, int(np.floor((high - start) / step + COORD_ATOL / step)) + 1)
    return first, stop


def area_to_index_range(
    grid: GridDef, area: tuple[float, float, float, float] | None
) -> tuple[int, int, int, int] | None:
    """Cells of `grid` that lie within `area`, boundaries included.

    Args:
        grid (GridDef): The grid.
        area (tuple | None): (north, west, south, east), or None for the whole
            grid.

    Returns:
        tuple[int, int, int, int] | None: Half-open ranges
            (i_start, i_stop, j_start, j_stop), or None if no cell is in the
            area.
    """
    if area is None:
        return 0, grid.n_lat, 0, grid.n_lon
    north, west, south, east = area
    i0, i1 = _axis_range(grid.lat0, grid.n_lat, grid.step, south, north)
    j0, j1 = _axis_range(grid.lon0, grid.n_lon, grid.step, west, east)
    if i0 >= i1 or j0 >= j1:
        return None
    return i0, i1, j0, j1


def tile_range(
    grid: GridDef, i0: int, i1: int, j0: int, j1: int
) -> tuple[int, int, int, int]:
    """Tiles that hold the cells [i0, i1) × [j0, j1).

    Returns:
        tuple[int, int, int, int]: Half-open ranges
            (tile_row_start, tile_row_stop, tile_col_start, tile_col_stop).
    """
    ts = grid.tile_size
    return i0 // ts, (i1 - 1) // ts + 1, j0 // ts, (j1 - 1) // ts + 1


def encode_tile(values: np.ndarray) -> bytes:
    """Raw bytes of a tile: float32, little endian, row-major."""
    return np.ascontiguousarray(values, dtype=TILE_DTYPE).tobytes()


def decode_tile(data: bytes | memoryview, tile_size: int) -> np.ndarray:
    """Inverse of `encode_tile`. Returns a read-only (tile_size, tile_size) view."""
    values = np.frombuffer(data, dtype=TILE_DTYPE)
    if values.size != tile_size * tile_size:
        raise ValueError(
            f"Tile has {values.size} values, expected {tile_size * tile_size}."
        )
    return values.reshape(tile_size, tile_size)


def array_to_tiles(
    values: np.ndarray, tile_size: int
) -> Iterator[tuple[int, int, bytes]]:
    """Cut a 2D (lat, lon) array into encoded tiles.

    Edge tiles are padded with NaN. Tiles that are entirely NaN are skipped.

    Args:
        values (np.ndarray): Values on the full grid, rows ascending in
            latitude, columns ascending in longitude.
        tile_size (int): Cells per tile side.

    Yields:
        tuple[int, int, bytes]: (tile_row, tile_col, encoded tile).
    """
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {values.shape}.")
    n_lat, n_lon = values.shape
    padded = np.full(
        (-(-n_lat // tile_size) * tile_size, -(-n_lon // tile_size) * tile_size),
        np.nan,
        dtype=TILE_DTYPE,
    )
    padded[:n_lat, :n_lon] = values
    n_rows = padded.shape[0] // tile_size
    n_cols = padded.shape[1] // tile_size
    # (tile_row, tile_col, cell row, cell col)
    blocks = padded.reshape(n_rows, tile_size, n_cols, tile_size).swapaxes(1, 2)
    has_data = ~np.isnan(blocks).all(axis=(2, 3))
    for tile_row, tile_col in zip(*np.nonzero(has_data)):
        yield int(tile_row), int(tile_col), encode_tile(blocks[tile_row, tile_col])


def tiles_to_array(
    tiles: Iterable[tuple[int, int, bytes | memoryview]],
    tile_size: int,
    i0: int,
    i1: int,
    j0: int,
    j1: int,
) -> np.ndarray:
    """Assemble the cells [i0, i1) × [j0, j1) from encoded tiles.

    Cells in tiles that are not given are NaN. Tiles outside the range are
    ignored.

    Args:
        tiles (Iterable): (tile_row, tile_col, encoded tile) triples.
        tile_size (int): Cells per tile side.
        i0, i1, j0, j1 (int): Half-open row and column range to return.

    Returns:
        np.ndarray: float32 array of shape (i1 - i0, j1 - j0).
    """
    out = np.full((i1 - i0, j1 - j0), np.nan, dtype=TILE_DTYPE)
    for tile_row, tile_col, data in tiles:
        # cells of this tile, in grid indices
        ti0, tj0 = tile_row * tile_size, tile_col * tile_size
        r0, r1 = max(i0, ti0), min(i1, ti0 + tile_size)
        c0, c1 = max(j0, tj0), min(j1, tj0 + tile_size)
        if r0 >= r1 or c0 >= c1:
            continue
        tile = decode_tile(data, tile_size)
        out[r0 - i0 : r1 - i0, c0 - j0 : c1 - j0] = tile[
            r0 - ti0 : r1 - ti0, c0 - tj0 : c1 - tj0
        ]
    return out
