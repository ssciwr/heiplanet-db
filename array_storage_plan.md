# Plan: store gridded values as compressed tiles instead of one row per value

Status: proposal, 2026-11-23. Follows the storage quick wins on branch
`include-pmodel` (narrower column types, no PostGIS point, constraints rebuilt
after the bulk load, binary COPY loader).

## Why

Measured on the local deployment (`onehealth_db`, 60 months, R0 and
`adults_log10`, 9 resolutions), before the quick wins:

| Object | Size | Rows |
|---|---|---|
| `var_value` table + unique index | 10.0 GB | 102.6 M |
| `grid_point` + indexes | 1.5 GB | 6.5 M |
| `grid_point_resolution` + index | 0.55 GB | 8.5 M |
| `var_coverage` + index | 0.36 GB | 5.1 M |
| whole database | 12 GB | |

Each value costs about 100 bytes (52 in the table, 48 in the index), of which 8
are the value. The quick wins bring this to about 85 bytes. The remaining cost
is the row-per-value layout itself: a 24-byte tuple header, a 4-byte line
pointer and an index entry for every single number.

The data, however, is dense rasters, and every query reads rasters: one month,
one variable, one resolution, a rectangular area.

## Measured size of the tiled layout

One month of real data from the local DB, cut into 100 × 100 cell tiles,
stored as float32 `bytea` with lz4 compression, tiles that are entirely NaN
skipped:

| Data (one month) | Tiles | Compressed | Today (rows × ~100 B) |
|---|---|---|---|
| R0, 0.1° | 421 | 2.3 MB | ~89 MB |
| adults_log10, 0.1° | 421 | 2.5 MB | ~61 MB |
| adults_log10, 0.2° | 138 | 0.7 MB | ~16 MB |

That is about 0.9 bytes per grid cell, zeros and land cells included: the many
zeros and floor values compress very well. Extrapolated to the 60 months of the
local DB, all resolutions included: **about 0.4 GB instead of about 12 GB**
(roughly 30× smaller), and `grid_point`, `grid_point_resolution` and
`var_coverage` are no longer needed.

## Design

### Tables

```sql
-- one regular grid per resolution; points are implicit
CREATE TABLE grid_def (
    resolution_id smallint PRIMARY KEY REFERENCES resolution_group (id),
    lat0 double precision NOT NULL,   -- latitude of row 0
    lon0 double precision NOT NULL,   -- longitude of column 0
    step double precision NOT NULL,   -- degrees
    n_lat integer NOT NULL,
    n_lon integer NOT NULL,
    tile_size smallint NOT NULL       -- cells per tile side, e.g. 100
);

CREATE TABLE var_tile (
    var_id smallint NOT NULL REFERENCES var_type (id) ON DELETE CASCADE,
    resolution_id smallint NOT NULL REFERENCES resolution_group (id),
    time_id integer NOT NULL REFERENCES time_point (id) ON DELETE CASCADE,
    tile_row smallint NOT NULL,
    tile_col smallint NOT NULL,
    data bytea COMPRESSION lz4 NOT NULL,  -- float32 little-endian, row-major
    PRIMARY KEY (var_id, resolution_id, time_id, tile_row, tile_col)
);
```

- Cell `(i, j)` of a grid is at `lat0 + i * step`, `lon0 + j * step`. It lies
  in tile `(i // tile_size, j // tile_size)`. Edge tiles are padded with NaN.
- **NaN means "no data"** (ocean, outside the model domain), **0 means 0**.
  This replaces `var_coverage`; compression makes the zeros almost free.
- `tile_size = 100` cells: 40 kB raw per tile, about 5 kB compressed. A global
  0.1° month is about 420 land tiles; a global 5° month is a single tile.
- Every variable is stored at every resolution, including "subsample"
  variables (strided at load time). This costs about +35 % over the finest
  resolution and keeps every query the same: read the tiles of one resolution.
  `var_type.aggregation` still decides how coarse values are computed
  (stride or area-weighted mean, as now).
- Optional later step: int16 quantisation with a scale and offset per variable
  would halve the raw size. It is not needed for the size targets above.

### Loading

For each (variable, resolution, time step): build the full 2D array as now
(`block_mean` / stride, transform, mask), cut it into tiles, skip all-NaN
tiles, and send the tiles with one binary `COPY` (`copy_rows` already exists).
No grid point IDs and no `grid_id_map` are needed. The 6.5 M-entry dict
(likely over 1 GB of RAM) and the insert of 6.5 M grid points disappear.
Loading time should be dominated by reading the NetCDF files.

### Queries (API unchanged)

- `get_var_values_cartesian(time, resolution, area, var)`: convert the area to a
  range of rows/columns and tiles, fetch those tiles in a single query, decode
  them with `np.frombuffer`, crop, and build the response. The response format
  `{"latitude, longitude, var_value": [...]}` stays the same. Cells that are NaN
  are left out, exactly like grid points without a value and without coverage
  today.
- `get_var_value(point)`: one tile fetch and an array index.
- `get_var_values_cartesian_for_download`: assemble the xarray dataset directly
  from the tiles of the finest resolution. No per-value ORM objects (today it
  loads every row as an ORM object, which is the most memory-hungry request).
- NUTS tables and endpoints are unchanged. `get_grid_ids_in_nuts` is not used
  by the API; it can be rewritten to return a cell mask of the regions instead
  of grid point IDs, or removed.
- A SQL view (`var_value_v`) that unnests tiles into
  `(var, resolution, time, latitude, longitude, value)` rows keeps ad-hoc SQL
  inspection possible. It is for debugging only, not for the API.

## Steps

1. **Grid and tile helpers** (pure numpy, unit tests): cell ↔ coordinate
   conversion, area → tile range, array → tiles, tiles → array with cropping.
2. **Schema**: `grid_def`, `var_tile`, the debug view. These live alongside the
   current tables; nothing is dropped yet.
3. **Loader**: `insert_var_tiles(engine, ds, var, ...)` for both aggregations,
   reusing `_block_mean_plan`, `block_mean`, `apply_transform` and the mask
   handling of `insert_var_values_by_resolution`. Add a config switch
   `storage: tiles | rows` in `production.py` (default `rows` at first).
4. **Read path**: tile versions of the three query functions, selected by the
   same switch.
5. **Equivalence check**: load the 2014–2018 config both ways in one test
   database. Compare the responses of `/cartesian` (several areas, all
   resolutions, both variables), single-point lookups and the NetCDF download.
   They must be equal within float32 precision. Record the database size, the
   load time, the response times and the peak API memory.
6. **Switch**: default to `tiles`, redeploy with a fresh load, and keep the old
   path for one release.
7. **Clean-up**: remove `var_value`, `var_coverage`, `grid_point_resolution`,
   `grid_point` (if nothing else uses it) and the row-based code and tests.

Steps 1–4 are the bulk of the work. Step 5 is what makes the switch safe.

## Risks and open points

- **Single-value reads** decompress a whole tile (about 5 kB) for one number.
  This is negligible for an interactive map, but worth checking if point
  time series are added later. A time series then reads one tile per month.
  If that becomes a hot path, a `(var, resolution, tile, time-range)` layout
  could be considered.
- **Large areas at 0.1°**: the response is still a JSON list of up to millions
  of triples, now built from arrays. The database stops being the bottleneck;
  JSON encoding and the transfer then dominate. Returning compact arrays
  (`lat0`, `step`, a values array) would need a frontend change. It is out of
  scope here but becomes easy once the data is tiled.
- **Grid alignment**: all gridded variables must share one grid per resolution.
  This is already enforced by `_check_same_grid`; `grid_def` makes it explicit.
- **Tile size** (100 is a first guess): to be measured in step 5. Smaller tiles
  mean less waste at area edges and more rows; larger tiles mean fewer rows and
  more over-reading.
