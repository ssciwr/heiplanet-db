import numpy as np
import pytest

from heiplanet_db import tiles


@pytest.fixture(autouse=True)
def clean_db():
    # overrides the autouse fixture of conftest: these tests need no database
    yield


@pytest.fixture
def grid():
    # 0.5 degree grid, 7 rows (-1.5 .. 1.5) and 9 columns (10 .. 14), tiles of 4
    return tiles.GridDef(lat0=-1.5, lon0=10.0, step=0.5, n_lat=7, n_lon=9, tile_size=4)


def test_grid_def_validation():
    with pytest.raises(ValueError, match="step"):
        tiles.GridDef(lat0=0, lon0=0, step=0, n_lat=1, n_lon=1)
    with pytest.raises(ValueError, match="row and one column"):
        tiles.GridDef(lat0=0, lon0=0, step=1, n_lat=0, n_lon=1)
    with pytest.raises(ValueError, match="tile size"):
        tiles.GridDef(lat0=0, lon0=0, step=1, n_lat=1, n_lon=1, tile_size=0)


def test_grid_def_properties(grid):
    assert grid.n_tile_rows == 2
    assert grid.n_tile_cols == 3
    np.testing.assert_allclose(grid.latitudes(), np.arange(-1.5, 1.6, 0.5))
    np.testing.assert_allclose(grid.longitudes(2, 4), [11.0, 11.5])


def test_grid_def_from_coords():
    lats = np.arange(-1.5, 1.6, 0.5)
    lons = np.arange(10.0, 14.1, 0.5)
    grid = tiles.grid_def_from_coords(lats, lons, tile_size=4)
    assert grid == tiles.GridDef(-1.5, 10.0, 0.5, 7, 9, 4)
    # a single row needs an explicit step
    with pytest.raises(ValueError, match="explicit step"):
        tiles.grid_def_from_coords(np.array([0.0]), lons)
    single = tiles.grid_def_from_coords(np.array([0.0]), lons, step=0.5)
    assert single.n_lat == 1 and single.step == 0.5
    with pytest.raises(ValueError, match="does not match"):
        tiles.grid_def_from_coords(lats, lons, step=1.0)
    with pytest.raises(ValueError, match="ascending"):
        tiles.grid_def_from_coords(lats[::-1], lons)
    with pytest.raises(ValueError, match="regularly"):
        tiles.grid_def_from_coords(np.array([0.0, 0.5, 1.5]), lons)
    with pytest.raises(ValueError, match="differ"):
        tiles.grid_def_from_coords(lats, np.arange(10.0, 14.1, 1.0))
    with pytest.raises(ValueError, match="Empty"):
        tiles.grid_def_from_coords(np.array([]), lons)


def test_grid_def_from_float32_coords():
    # coordinates read from NetCDF files are often float32
    lats = np.arange(-60, 60, 0.1).astype(np.float32)
    lons = np.arange(-180, 180, 0.1).astype(np.float32)
    grid = tiles.grid_def_from_coords(lats, lons, step=0.1)
    assert grid.n_lat == len(lats)
    assert tiles.cell_index(grid, 59.9, 179.9) == (len(lats) - 1, len(lons) - 1)


def test_cell_index(grid):
    assert tiles.cell_index(grid, -1.5, 10.0) == (0, 0)
    assert tiles.cell_index(grid, 1.5, 14.0) == (6, 8)
    assert tiles.cell_index(grid, 0.0, 12.5) == (3, 5)
    # rounding noise is accepted
    assert tiles.cell_index(grid, 0.00001, 12.49999) == (3, 5)
    # between grid points or off the grid
    assert tiles.cell_index(grid, 0.25, 12.5) is None
    assert tiles.cell_index(grid, 2.0, 12.5) is None
    assert tiles.cell_index(grid, 0.0, 9.5) is None


def test_area_to_index_range(grid):
    assert tiles.area_to_index_range(grid, None) == (0, 7, 0, 9)
    # boundaries are included: (north, west, south, east)
    assert tiles.area_to_index_range(grid, (0.5, 11.0, -0.5, 12.0)) == (2, 5, 2, 5)
    # boundaries between grid points
    assert tiles.area_to_index_range(grid, (0.7, 11.2, -0.7, 12.2)) == (2, 5, 3, 5)
    # larger than the grid
    assert tiles.area_to_index_range(grid, (90, -180, -90, 180)) == (0, 7, 0, 9)
    # no cell inside
    assert tiles.area_to_index_range(grid, (0.4, 11.0, 0.1, 12.0)) is None
    assert tiles.area_to_index_range(grid, (10, 20, 5, 30)) is None
    # a single cell
    assert tiles.area_to_index_range(grid, (0, 12.5, 0, 12.5)) == (3, 4, 5, 6)
    # non-finite bounds
    for bad in (np.inf, -np.inf, np.nan):
        with pytest.raises(ValueError, match="finite"):
            tiles.area_to_index_range(grid, (bad, 11.0, -0.5, 12.0))


def test_shortest_float64():
    values = np.array(
        [[0.3, 293.15, 0.0, -1.25e-5], [np.nan, np.inf, 3.4e38, 1e-30]],
        dtype=np.float32,
    )
    result = tiles.shortest_float64(values)
    assert result.dtype == np.float64 and result.shape == values.shape
    assert result[0].tolist() == [0.3, 293.15, 0.0, -1.25e-5]
    assert np.isnan(result[1, 0]) and result[1, 1] == np.inf
    # always the same float32, and the shortest decimal for random values
    rng = np.random.default_rng(0)
    random = (rng.standard_normal(10_000) * 10.0 ** rng.integers(-8, 8, 10_000)).astype(
        np.float32
    )
    shortest = tiles.shortest_float64(random)
    np.testing.assert_array_equal(shortest.astype(np.float32), random)
    np.testing.assert_array_equal(shortest, random.astype(str).astype(np.float64))
    # scalars
    assert float(tiles.shortest_float64(np.float32(0.3))) == 0.3


def test_tile_range(grid):
    assert tiles.tile_range(grid, 0, 7, 0, 9) == (0, 2, 0, 3)
    assert tiles.tile_range(grid, 2, 4, 2, 4) == (0, 1, 0, 1)
    assert tiles.tile_range(grid, 3, 5, 3, 9) == (0, 2, 0, 3)
    assert tiles.tile_range(grid, 4, 5, 8, 9) == (1, 2, 2, 3)


def test_encode_decode_tile():
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    values[1, 2] = np.nan
    data = tiles.encode_tile(values)
    assert len(data) == 16 * 4
    # little endian float32
    assert data[4:8] == np.array(1.0, dtype="<f4").tobytes()
    decoded = tiles.decode_tile(data, 4)
    np.testing.assert_array_equal(decoded, values.astype(np.float32))
    with pytest.raises(ValueError, match="expected 9"):
        tiles.decode_tile(data, 3)


def test_array_to_tiles_and_back(grid):
    rng = np.random.default_rng(0)
    values = rng.random((grid.n_lat, grid.n_lon))
    values[values < 0.3] = 0.0
    # tile (0, 2) holds columns 8 only, rows 0..3: make it all NaN
    values[0:4, 8] = np.nan
    result = list(tiles.array_to_tiles(values, grid.tile_size))
    keys = [(r, c) for r, c, _ in result]
    assert keys == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    assert all(len(d) == 16 * 4 for _, _, d in result)

    full = tiles.tiles_to_array(result, grid.tile_size, 0, grid.n_lat, 0, grid.n_lon)
    assert full.dtype == np.float32
    np.testing.assert_array_equal(full, values.astype(np.float32))

    # cropping, across tile borders
    part = tiles.tiles_to_array(result, grid.tile_size, 2, 6, 3, 9)
    np.testing.assert_array_equal(part, values[2:6, 3:9].astype(np.float32))

    # missing tiles are NaN, tiles outside the range are ignored
    some = [t for t in result if (t[0], t[1]) == (1, 1)]
    part = tiles.tiles_to_array(some, grid.tile_size, 3, 6, 3, 6)
    expected = np.full((3, 3), np.nan, dtype=np.float32)
    expected[1:, 1:] = values[4:6, 4:6]
    np.testing.assert_array_equal(part, expected)


def test_array_to_tiles_padding():
    values = np.ones((3, 5))
    result = list(tiles.array_to_tiles(values, 4))
    assert [(r, c) for r, c, _ in result] == [(0, 0), (0, 1)]
    edge = tiles.decode_tile(result[1][2], 4)
    np.testing.assert_array_equal(edge[:3, 0], 1.0)
    assert np.isnan(edge[3, :]).all() and np.isnan(edge[:, 1:]).all()


def test_array_to_tiles_all_nan_and_errors():
    assert list(tiles.array_to_tiles(np.full((5, 5), np.nan), 4)) == []
    with pytest.raises(ValueError, match="2D"):
        list(tiles.array_to_tiles(np.ones((2, 2, 2)), 4))
