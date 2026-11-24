import os
import zipfile
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

import dotenv
import numpy as np
import pandas as pd
import pooch
import xarray as xr
import yaml
from sqlalchemy import engine, text

from heiplanet_db import postgresql_database as db

# config "type" values and what kind of input they describe;
# "R0" and "R0_nuts" are kept as aliases for older configs
GRID_TYPES = {"grid", "R0"}
NUTS_TYPES = {"nuts", "R0_nuts"}
DEFINITION_TYPES = {"definition"}
AGGREGATIONS = {"subsample", "mean"}


@dataclass
class GridVar:
    """A gridded variable to load, possibly spread over several files."""

    name: str
    paths: list[Path]
    aggregation: str = "subsample"
    mask_by: str | None = None
    # name of the variable in the file(s), if it differs from `name`
    source_name: str | None = None
    # transform applied before storing (after aggregating), e.g. "log10"
    transform: str | None = None
    transform_floor: float = 1.0

    @property
    def file_variable(self) -> str:
        return self.source_name or self.name


@dataclass
class NutsVar:
    """A variable aggregated over NUTS regions."""

    name: str
    path: Path
    source_name: str | None = None
    transform: str | None = None
    transform_floor: float = 1.0

    @property
    def file_variable(self) -> str:
        return self.source_name or self.name


def read_production_config(dict_path: str | Traversable | Path | None = None) -> dict:
    """
    Read configuration of the production database.

    Args:
        dict_path (str|Traversable): Path to the configuration dictionary.
            Defaults to None, which uses the default path.
    Returns:
        dict: Dict with configuration details for the
            production database.
    """
    if dict_path is None:
        dict_path = (
            resources.files("heiplanet_db") / "data" / "production_config_small.yml"
        )
    # check if the file exists
    if isinstance(dict_path, str):
        dict_path = Path(dict_path)
    if not dict_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {dict_path}")
    # read the configuration file
    with dict_path.open("r") as file:
        production_dict = yaml.safe_load(file)
    return production_dict


def get_production_data(url: str, filename: str, filehash: str, outputdir: Path) -> int:
    """
    Fetch data that is fed into the production database.

    Args:
    url (str): URL to fetch the data from.
        filename (str): Name of the file to be fetched.
        filehash (str): SHA256SUM hash of the file to verify integrity.
        outputdir (Path): Directory where the file will be saved.
    Returns:
        completion_code (int): Status code indicating the success or
            failure of the operation.
    """
    try:
        file = pooch.retrieve(
            url=url,
            known_hash=filehash,
            fname=filename,
            path=outputdir,
            progressbar=True,
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise RuntimeError(f"Failed to fetch data from {url}") from e
    print(f"Data fetched and saved to {file}")
    return 0


def create_directories(dir: str) -> int:
    """
    Create directories if they do not exist.

    Args:
        dir (str): String of the directory to create/use.
    """
    output_dir = Path(dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return 0


def get_engine(drop_tables: bool = False) -> engine.Engine:
    """
    Get or initialize the database engine.

    Args:
        drop_tables (bool): If True, drop all existing tables and recreate them.
            If False, only create tables if they don't exist. Defaults to False.

    Returns:
        engine.Engine: SQLAlchemy engine object.
    """
    # get the db url from dotenv
    dotenv.load_dotenv()
    db_url = os.environ.get("DB_URL")
    try:
        engine = db.initialize_database(db_url, replace=drop_tables)
    except Exception as e:
        raise ValueError(
            f"Could not initialize engine, please check \
                         your db_url {db_url}"
        ) from e
    return engine


def insert_data(engine: engine.Engine, shapefiles_folder_path: Path) -> int:
    # check that the folder exists
    if not shapefiles_folder_path.is_dir():
        raise ValueError(
            f"Shapefile folder path {shapefiles_folder_path} does not exist."
        )
    # add NUTS definition data
    db.insert_nuts_def(engine, shapefiles_folder_path)
    return 0


def get_var_types_from_config(config: dict) -> list:
    """Get the variable types from the configuration file and
    place them in a dictionary.

    The first entry for a variable name sets its unit and description. The
    aggregation is taken from whichever entry sets it (default "subsample").
    """
    var_types: dict[str, dict] = {}
    for data in config:
        for var_name in data["var_name"]:
            aggregation = var_name.get("aggregation")
            if aggregation is not None and aggregation not in AGGREGATIONS:
                raise ValueError(
                    f"Unknown aggregation {aggregation} for {var_name['name']}; "
                    f"use one of {sorted(AGGREGATIONS)}."
                )
            if var_name["name"] not in var_types:
                var_types[var_name["name"]] = {
                    "name": var_name["name"],
                    "unit": var_name["unit"],
                    "description": var_name["description"],
                    "aggregation": aggregation or "subsample",
                }
            elif aggregation is not None:
                var_types[var_name["name"]]["aggregation"] = aggregation
    return list(var_types.values())


def check_paths(paths: list[Path | None]) -> None:
    """Check that the paths are not None."""
    for path in paths:
        if path is None:
            raise ValueError(
                "One of the paths is None, please check your configuration."
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"File not found at {path}. Please check your configuration."
            )


def open_grid_dataset(grid_var: GridVar) -> xr.Dataset:
    """Open the file(s) of a gridded variable as one lazy dataset.

    Args:
        grid_var (GridVar): Variable to open.

    Returns:
        xr.Dataset: Dataset with only this variable, named `grid_var.name`, as
            (time, latitude, longitude). Values are not transformed yet.
    """
    check_paths(grid_var.paths)
    if len(grid_var.paths) == 1:
        ds = xr.open_dataset(grid_var.paths[0], chunks={})
    else:
        ds = xr.open_mfdataset(grid_var.paths, combine="by_coords", chunks={})
    source = grid_var.file_variable
    if source not in ds:
        raise ValueError(f"Variable {source} not found in {grid_var.paths}.")
    ds = ds[[source]]
    if source != grid_var.name:
        ds = ds.rename({source: grid_var.name})
    # some model outputs are written as (longitude, latitude, time)
    return ds.transpose("time", "latitude", "longitude", ...)


def _check_same_grid(datasets: dict[str, xr.Dataset]) -> xr.Dataset:
    """Make sure all gridded datasets share one grid and return the first."""
    names = list(datasets)
    reference = datasets[names[0]]
    for name in names[1:]:
        ds = datasets[name]
        if not (
            np.array_equal(ds.latitude.values, reference.latitude.values)
            and np.array_equal(ds.longitude.values, reference.longitude.values)
        ):
            raise ValueError(
                f"{name} is on a different grid than {names[0]}; all gridded "
                "variables must share one grid."
            )
    return reference


def insert_grid_var_values(
    engine: engine.Engine, grid_vars: list[GridVar], storage: str = "rows"
) -> int:
    """Insert grid points, time points and values of all gridded variables.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        grid_vars (list[GridVar]): Gridded variables to insert.
        storage (str): "rows" (one row per value in `var_value`) or "tiles"
            (compressed tiles in `var_tile`, no grid points). Defaults to "rows".
    """
    if storage not in db.STORAGES:
        raise ValueError(
            f"Unknown storage {storage}; use one of {sorted(db.STORAGES)}."
        )
    names = {gv.name for gv in grid_vars}
    for gv in grid_vars:
        if gv.aggregation not in AGGREGATIONS:
            raise ValueError(f"Unknown aggregation {gv.aggregation} for {gv.name}.")
        db.apply_transform(np.array([1.0]), gv.transform, gv.transform_floor)
        if gv.mask_by is not None and gv.mask_by not in names:
            raise ValueError(
                f"{gv.name} is masked by {gv.mask_by}, which is not a gridded "
                "variable in the configuration."
            )
    datasets = {gv.name: open_grid_dataset(gv) for gv in grid_vars}
    reference = _check_same_grid(datasets)
    if storage == "tiles":
        _insert_grid_var_tiles(engine, grid_vars, datasets)
        return 0

    # add grid points
    grid_point_session = db.create_session(engine)
    db.insert_grid_points(
        grid_point_session,
        latitudes=reference.latitude.to_numpy(),
        longitudes=reference.longitude.to_numpy(),
    )
    grid_point_session.commit()
    grid_point_session.close()
    # add time points of all variables (monthly data)
    _insert_time_points(engine, datasets)
    # get id maps for grid, time, variable types and resolutions
    print("Getting id maps for grid, time, and variable types.")
    id_map_session = db.create_session(engine)
    grid_id_map, time_id_map, var_type_id_map = db.get_id_maps(id_map_session)
    resolution_id_map = db.get_resolution_id_map(id_map_session)
    id_map_session.close()

    for gv in grid_vars:
        print(f"Inserting values of {gv.name} ({gv.aggregation}) into the database.")
        ds = datasets[gv.name]
        if gv.aggregation == "subsample":
            # rechunk the dataset
            ds = ds.chunk({"time": 1, "latitude": 180, "longitude": 360})
            ds = db.apply_transform(ds, gv.transform, gv.transform_floor)
            db.insert_var_values(
                engine, ds, gv.name, grid_id_map, time_id_map, var_type_id_map
            )
        else:
            mask = None
            if gv.mask_by is not None:
                mask = datasets[gv.mask_by][gv.mask_by].notnull()
            db.insert_var_values_by_resolution(
                engine,
                ds,
                gv.name,
                grid_id_map,
                time_id_map,
                var_type_id_map,
                resolution_id_map,
                mask=mask,
                transform=gv.transform,
                transform_floor=gv.transform_floor,
            )
    # assign resolution groups to grid points
    print("Assigning resolution groups to grid points.")
    resolution_session = db.create_session(engine)
    db.assign_grid_resolution_group_to_grid_point(resolution_session)
    resolution_session.close()
    print("Variable values inserted and resolution groups assigned.")
    return 0


def _insert_time_points(engine: engine.Engine, datasets: dict[str, xr.Dataset]):
    """Insert the (monthly) time points of all datasets that are not stored yet.

    Skipping stored ones allows loading the same data as rows and as tiles
    into one database, e.g. to compare the two.
    """
    session = db.create_session(engine)
    stored = {(tp.year, tp.month, tp.day) for tp in session.query(db.TimePoint)}
    times = np.unique(np.concatenate([ds.time.to_numpy() for ds in datasets.values()]))
    new = [t for t in times if pd.Timestamp(t).timetuple()[:3] not in stored]
    if new:
        db.insert_time_points(
            session, time_point_data=[(np.array(new, dtype=times.dtype), False)]
        )
    session.commit()
    session.close()


def _insert_grid_var_tiles(
    engine: engine.Engine, grid_vars: list[GridVar], datasets: dict[str, xr.Dataset]
) -> None:
    """Insert time points and the values of all gridded variables as tiles."""
    _insert_time_points(engine, datasets)
    session = db.create_session(engine)
    _, time_id_map, var_type_id_map = db.get_id_maps(session)
    resolution_id_map = db.get_resolution_id_map(session)
    session.close()
    for gv in grid_vars:
        print(f"Inserting tiles of {gv.name} ({gv.aggregation}) into the database.")
        mask = None
        if gv.mask_by is not None:
            mask = datasets[gv.mask_by][gv.mask_by].notnull()
        db.insert_var_tiles(
            engine,
            datasets[gv.name],
            gv.name,
            time_id_map,
            var_type_id_map,
            resolution_id_map,
            aggregation=gv.aggregation,
            mask=mask,
            transform=gv.transform,
            transform_floor=gv.transform_floor,
        )
    print("Variable tiles inserted.")


def insert_nuts_var_values(engine: engine.Engine, nuts_vars: list[NutsVar]) -> int:
    """Insert the values of all NUTS variables.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        nuts_vars (list[NutsVar]): NUTS variables to insert.
    """
    check_paths([nv.path for nv in nuts_vars])
    id_map_session = db.create_session(engine)
    _, time_id_map, var_type_id_map = db.get_id_maps(id_map_session)
    id_map_session.close()
    datasets: dict[Path, xr.Dataset] = {}
    for nv in nuts_vars:
        db.apply_transform(np.array([1.0]), nv.transform, nv.transform_floor)
    for nv in nuts_vars:
        if nv.path not in datasets:
            datasets[nv.path] = xr.open_dataset(nv.path, chunks={})
        source = nv.file_variable
        if source not in datasets[nv.path]:
            raise ValueError(f"Variable {source} not found in {nv.path}.")
        da = db.apply_transform(
            datasets[nv.path][source], nv.transform, nv.transform_floor
        )
        db.insert_var_value_nuts(
            engine,
            da.to_dataset(name=nv.name),
            var_name=nv.name,
            time_id_map=time_id_map,
            var_id_map=var_type_id_map,
        )
    return 0


def reset_production_data(engine: engine.Engine) -> None:
    """
    Truncate production data tables while keeping the existing schema.

    This makes reruns of the production load idempotent when tables are kept
    (i.e. when drop_tables=False), avoiding uniqueness constraint violations
    from bulk inserts into already-populated tables.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                TRUNCATE TABLE
                    var_tile,
                    grid_def,
                    grid_point_resolution,
                    var_value_nuts,
                    var_coverage,
                    var_value,
                    grid_point,
                    time_point,
                    var_type,
                    resolution_group,
                    nuts_def
                RESTART IDENTITY CASCADE;
                """
            )
        )
    print("Production data tables truncated (schema preserved).")


def get_data_files(data: dict, config: dict, data_level: str) -> None:
    if "local" in data["host"]:
        # if the host is local, we can use the local path
        data["url"] = str(Path(data["url"]).resolve())
        print(f"Using local path {data['url']} for {data['filename']}")
    elif "heibox" in data["host"]:
        # if the host is heibox, we need to use the heibox URL
        get_production_data(
            url=data["url"],
            filename=data["filename"],
            filehash=data["filehash"],
            outputdir=Path(config["datalake"][data_level]),
        )
    else:
        raise ValueError(f"Unknown host {data['host']} for data {data['filename']}")


def _as_list(value) -> list:
    """Wrap a single config value in a list; None becomes an empty list."""
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def fetch_files(data: dict, config: dict, data_level: str) -> list[Path]:
    """Make sure all files of a config entry are available.

    `filename` may be a single name or a list. `url` and `filehash` must then
    be lists of the same length.

    For `host: local`, files are read from `url` if it is given (a path to the
    file, or to the folder holding the files when `filename` is a list), and
    otherwise from the data lake. Local files are not copied.

    Args:
        data (dict): Config entry from `data_to_fetch`.
        config (dict): Full production configuration.
        data_level (str): Data lake level the files belong to.

    Returns:
        list[Path]: Paths of the files in the data lake.
    """
    filenames = _as_list(data.get("filename"))
    urls = _as_list(data.get("url"))
    if "local" in data.get("host", ""):
        return _local_files(filenames, urls, Path(config["datalake"][data_level]))
    filehashes = _as_list(data.get("filehash"))
    _check_lengths(filenames, url=urls, filehash=filehashes)
    return [
        _fetch_file(
            data,
            config,
            data_level,
            filename,
            url=_entry_for_file(urls, i, len(filenames)),
            filehash=_entry_for_file(filehashes, i, len(filenames)),
        )
        for i, filename in enumerate(filenames)
    ]


def _check_lengths(filenames: list, **lists: list) -> None:
    """With several files, each given list must have one entry per file."""
    for key, values in lists.items():
        if len(filenames) > 1 and values and len(values) != len(filenames):
            raise ValueError(f"{key} must list one entry per file for {filenames}.")


def _entry_for_file(values: list, i: int, n_files: int):
    """The entry of a per-file config list for file `i` (None if not given)."""
    if not values:
        return None
    return values[i] if n_files > 1 else values[0]


def _local_files(filenames: list, urls: list, datalake_dir: Path) -> list[Path]:
    """Paths of local files: from `urls` (files or one folder), else the data lake."""
    if len(urls) == 1 and Path(urls[0]).is_dir():
        paths = [(Path(urls[0]) / f).resolve() for f in filenames]
    elif urls:
        _check_lengths(filenames, url=urls)
        paths = [Path(u).resolve() for u in urls]
    else:
        paths = [(datalake_dir / f).resolve() for f in filenames]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Local data files not found: {missing}")
    print(f"Using local files {[str(p) for p in paths]}.")
    return paths


def _fetch_file(
    data: dict,
    config: dict,
    data_level: str,
    filename: str,
    url: str | None,
    filehash: str | None,
) -> Path:
    """Download one file into the data lake unless it is already there."""
    path = (Path(config["datalake"][data_level]) / filename).resolve()
    if path.is_file():
        print(
            f"File {filename} already exists in the data lake "
            f"at level {data_level}, skipping download."
        )
    else:
        entry = dict(data)
        entry["filename"] = filename
        if url is not None:
            entry["url"] = url
        if filehash is not None:
            entry["filehash"] = filehash
        get_data_files(entry, config, data_level)
    return path


def _resolve_config_path(config_path: str | None) -> str | None:
    """Prefer an explicit path, then $CONFIG_FILE, then the container mount.

    None means the package default config.
    """
    if config_path is not None:
        return config_path
    env_config = os.getenv("CONFIG_FILE")
    if env_config:
        return env_config
    if Path("/heiplanet_db/production.yaml").is_file():
        return "/heiplanet_db/production.yaml"
    return None


def _collect_var(
    var: dict,
    paths: list[Path],
    grid_vars: list[GridVar],
    nuts_vars: list[NutsVar],
) -> Path | None:
    """Register one `var_name` entry of the config.

    Gridded and NUTS variables are appended to `grid_vars` / `nuts_vars`. For a
    NUTS definition, the shapefile archive is extracted.

    Returns:
        Path | None: The extracted shapefile folder for a definition, else None.
    """
    var_type = var.get("type")
    if var_type in DEFINITION_TYPES:
        # extract file and set the path to the NUTS shapefiles
        shapefile_path = paths[0]
        shapefile_folder_path = shapefile_path.with_suffix("")
        with zipfile.ZipFile(shapefile_path, "r") as zip_ref:
            print(f"Extracting zip archive to {shapefile_folder_path}.")
            zip_ref.extractall(shapefile_folder_path)
        return shapefile_folder_path
    if var_type in GRID_TYPES:
        grid_vars.append(
            GridVar(
                name=var["name"],
                paths=paths,
                aggregation=var.get("aggregation", "subsample"),
                mask_by=var.get("mask_by"),
                source_name=var.get("source_variable"),
                transform=var.get("transform"),
                transform_floor=float(var.get("transform_floor", 1.0)),
            )
        )
        print(f"Gridded data for {var['name']}: {[str(p) for p in paths]}")
        return None
    if var_type in NUTS_TYPES:
        if len(paths) != 1:
            raise ValueError(f"NUTS data for {var['name']} must be a single file.")
        nuts_vars.append(
            NutsVar(
                name=var["name"],
                path=paths[0],
                source_name=var.get("source_variable"),
                transform=var.get("transform"),
                transform_floor=float(var.get("transform_floor", 1.0)),
            )
        )
        print(f"NUTS data for {var['name']}: {paths[0]}")
        return None
    raise ValueError(
        f"Unknown type {var_type} for {var.get('name')} in the configuration."
    )


def main(drop_tables: bool = False, config_path: str | None = None) -> None:
    """
    Main function to set up the production database and data lake.
    This function reads the production configuration, creates the necessary
    directories, and fetches the data from the configured sources.
    It is intended to be run as a script.

    Args:
        drop_tables (bool): If True, drop all existing tables before inserting data.
            If False, keep the existing schema but truncate and reload all
            production data tables so the load is idempotent. Defaults to False.
            Set to True only when initializing a fresh database and you also
            want tables to be recreated.
        config_path (str | None): Path to the production configuration file.
            If None, uses the CONFIG_FILE environment variable or the default
            container path `/heiplanet_db/production.yaml` if available; otherwise
            falls back to the built-in default config. Defaults to None.
    """
    # set up production database and data lake using the provided config
    config = read_production_config(_resolve_config_path(config_path))
    # "rows" or "tiles", see insert_grid_var_values; checked before any download
    storage = config.get("storage", "rows")
    if storage not in db.STORAGES:
        raise ValueError(
            f"Unknown storage {storage}; use one of {sorted(db.STORAGES)}."
        )
    shapefile_folder_path = None
    grid_vars: list[GridVar] = []
    nuts_vars: list[NutsVar] = []
    # create the data lake structure if it does not exist
    for dir_name in config["datalake"]:
        create_directories(config["datalake"][dir_name])

    # fetch the data from the configured sources
    for data in config["data_to_fetch"]:
        # set the data level, default to bronze if not specified
        data_level = data["var_name"][0].get("level", "bronze")
        paths = fetch_files(data, config, data_level)
        for var in data["var_name"]:
            shapefile_folder_path = (
                _collect_var(var, paths, grid_vars, nuts_vars) or shapefile_folder_path
            )
    # now the files are in the silver stage, feed into database
    if not shapefile_folder_path:
        raise ValueError(
            "Shapefile path could not be generated from the data to fetch."
        )
    engine = get_engine(drop_tables=drop_tables)
    # When we keep tables (drop_tables=False) we still need to clear existing
    # data to avoid uniqueness violations from bulk inserts on reruns.
    if not drop_tables:
        reset_production_data(engine)
    # insert the NUTS shape data
    insert_data(engine=engine, shapefiles_folder_path=shapefile_folder_path)

    # insert the cartesian variables data
    var_type_session = db.create_session(engine)
    var_types = get_var_types_from_config(config=config["data_to_fetch"])
    db.insert_var_types(var_type_session, var_types)
    var_type_session.close()

    # insert the resolution groups
    resolution_session = db.create_session(engine)
    db.insert_resolution_groups(resolution_session)
    resolution_session.close()

    # load data with optimization
    load_data_with_optimization(
        engine,
        grid_vars=grid_vars,
        nuts_vars=nuts_vars,
        storage=storage,
    )


def _restore_after_bulk_load(engine: engine.Engine, after_error: bool) -> None:
    """Re-enable autovacuum and restore the constraints after a bulk load.

    Autovacuum is re-enabled first, so it is back on even if the constraints
    cannot be restored. Restoring them fails if the loaded data has duplicate
    rows or unknown IDs. After a successful load that error is raised. After a
    failed load it is only reported, so that it does not hide the error that
    stopped the load; the constraints then stay missing until the data is
    fixed and `db.add_bulk_load_constraints` is run.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        after_error (bool): Whether the load itself failed.
    """
    print("DEBUG: Re-enabling autovacuum on all tables...")
    with engine.begin() as conn:
        for table in ("grid_point", "time_point", "var_value", "var_value_nuts"):
            conn.execute(text(f"ALTER TABLE {table} SET (autovacuum_enabled = true);"))
    print("DEBUG: Autovacuum re-enabled on all tables.")

    print("Restoring the constraints of var_value and var_coverage...")
    try:
        db.add_bulk_load_constraints(engine)
    except Exception as e:
        if not after_error:
            raise
        print(
            "Warning: could not restore the constraints of var_value and "
            f"var_coverage after the failed load: {e}"
        )


def load_data_with_optimization(
    engine: engine.Engine,
    grid_vars: list[GridVar] | None = None,
    nuts_vars: list[NutsVar] | None = None,
    storage: str = "rows",
) -> None:
    """
    Load data into the database with autovacuum disabled for performance.
    Gridded variables are stored as `storage` ("rows" or "tiles"), see
    `insert_grid_var_values`.
    The unique and foreign key constraints of var_value and var_coverage are
    dropped during the load and created again afterwards, which is much faster
    and gives a smaller index. Ensures autovacuum and the constraints are
    restored even if an error occurs.
    """
    try:
        db.drop_bulk_load_constraints(engine)
        # Disable autovacuum globally for all tables during bulk load
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE grid_point SET (autovacuum_enabled = false);")
            )
            conn.execute(
                text("ALTER TABLE time_point SET (autovacuum_enabled = false);")
            )
            conn.execute(
                text("ALTER TABLE var_value SET (autovacuum_enabled = false);")
            )
            conn.execute(
                text("ALTER TABLE var_value_nuts SET (autovacuum_enabled = false);")
            )

        # insert the data
        if grid_vars:
            insert_grid_var_values(engine, grid_vars, storage=storage)
            print("DEBUG: insert_grid_var_values completed.")
        # insert the nuts variables data
        if nuts_vars:
            insert_nuts_var_values(engine, nuts_vars)
            print("DEBUG: insert_nuts_var_values completed.")
        print("DEBUG: All variable value inserts done.")
    except BaseException:
        # restore what we can, but let the original error propagate
        _restore_after_bulk_load(engine, after_error=True)
        raise
    _restore_after_bulk_load(engine, after_error=False)

    # Run VACUUM in autocommit mode (outside transaction)
    print("DEBUG: Running VACUUM ANALYZE...")
    with engine.connect() as conn:
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text("VACUUM ANALYZE;"))
    print("DEBUG: VACUUM ANALYZE completed.")
    print("Database vacuumed and ready.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Set up production database and data lake for HEIPlanet."
    )
    parser.add_argument(
        "--drop-tables",
        action="store_true",
        default=False,
        help="Drop all existing tables before inserting data (use only for fresh initialization).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the production configuration file. If not provided, uses the default config.",
    )

    args = parser.parse_args()
    main(drop_tables=args.drop_tables, config_path=args.config)
