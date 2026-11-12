import os
import zipfile
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

import dotenv
import numpy as np
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


@dataclass
class NutsVar:
    """A variable aggregated over NUTS regions."""

    name: str
    path: Path


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
        xr.Dataset: Dataset with only this variable, as (time, latitude, longitude).
    """
    check_paths(grid_var.paths)
    if len(grid_var.paths) == 1:
        ds = xr.open_dataset(grid_var.paths[0], chunks={})
    else:
        ds = xr.open_mfdataset(grid_var.paths, combine="by_coords", chunks={})
    if grid_var.name not in ds:
        raise ValueError(f"Variable {grid_var.name} not found in {grid_var.paths}.")
    # some model outputs are written as (longitude, latitude, time)
    return ds[[grid_var.name]].transpose("time", "latitude", "longitude", ...)


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


def insert_grid_var_values(engine: engine.Engine, grid_vars: list[GridVar]) -> int:
    """Insert grid points, time points and values of all gridded variables.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        grid_vars (list[GridVar]): Gridded variables to insert.
    """
    names = {gv.name for gv in grid_vars}
    for gv in grid_vars:
        if gv.aggregation not in AGGREGATIONS:
            raise ValueError(f"Unknown aggregation {gv.aggregation} for {gv.name}.")
        if gv.mask_by is not None and gv.mask_by not in names:
            raise ValueError(
                f"{gv.name} is masked by {gv.mask_by}, which is not a gridded "
                "variable in the configuration."
            )
    datasets = {gv.name: open_grid_dataset(gv) for gv in grid_vars}
    reference = _check_same_grid(datasets)

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
    time_point_session = db.create_session(engine)
    db.insert_time_points(
        time_point_session,
        time_point_data=[(ds.time.to_numpy(), False) for ds in datasets.values()],
    )
    time_point_session.commit()
    time_point_session.close()
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
            )
    # assign resolution groups to grid points
    print("Assigning resolution groups to grid points.")
    resolution_session = db.create_session(engine)
    db.assign_grid_resolution_group_to_grid_point(resolution_session)
    resolution_session.close()
    print("Variable values inserted and resolution groups assigned.")
    return 0


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
        if nv.path not in datasets:
            datasets[nv.path] = xr.open_dataset(nv.path, chunks={})
        db.insert_var_value_nuts(
            engine,
            datasets[nv.path],
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
                    grid_point_resolution,
                    var_value_nuts,
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
    filehashes = _as_list(data.get("filehash"))

    def _check_lengths(*items: tuple[str, list]) -> None:
        for key, values in items:
            if len(filenames) > 1 and values and len(values) != len(filenames):
                raise ValueError(f"{key} must list one entry per file for {filenames}.")

    if "local" in data.get("host", ""):
        if len(urls) == 1 and Path(urls[0]).is_dir():
            paths = [(Path(urls[0]) / f).resolve() for f in filenames]
        elif urls:
            _check_lengths(("url", urls))
            paths = [Path(u).resolve() for u in urls]
        else:
            paths = [
                (Path(config["datalake"][data_level]) / f).resolve() for f in filenames
            ]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Local data files not found: {missing}")
        print(f"Using local files {[str(p) for p in paths]}.")
        return paths

    _check_lengths(("url", urls), ("filehash", filehashes))

    paths = []
    for i, filename in enumerate(filenames):
        path = (Path(config["datalake"][data_level]) / filename).resolve()
        if path.is_file():
            print(
                f"File {filename} already exists in the data lake "
                f"at level {data_level}, skipping download."
            )
        else:
            entry = dict(data)
            entry["filename"] = filename
            if urls:
                entry["url"] = urls[i] if len(filenames) > 1 else urls[0]
            if filehashes:
                entry["filehash"] = (
                    filehashes[i] if len(filenames) > 1 else filehashes[0]
                )
            get_data_files(entry, config, data_level)
        paths.append(path)
    return paths


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
    # determine configuration path, preferring an explicit argument, then env var,
    # then the default container mount path; finally fall back to package default
    if config_path is None:
        env_config = os.getenv("CONFIG_FILE")
        if env_config:
            config_path = env_config
        elif Path("/heiplanet_db/production.yaml").is_file():
            config_path = "/heiplanet_db/production.yaml"
    # set up production database and data lake using the provided config
    config = read_production_config(config_path)
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
            var_type = var.get("type")
            if var_type in DEFINITION_TYPES:
                # extract file and set the path to the NUTS shapefiles
                shapefile_path = paths[0]
                # make sure the shapefile folder is unzipped
                shapefile_folder_path = shapefile_path.with_suffix("")
                with zipfile.ZipFile(shapefile_path, "r") as zip_ref:
                    print(f"Extracting zip archive to {shapefile_folder_path}.")
                    zip_ref.extractall(shapefile_folder_path)
            elif var_type in GRID_TYPES:
                grid_vars.append(
                    GridVar(
                        name=var["name"],
                        paths=paths,
                        aggregation=var.get("aggregation", "subsample"),
                        mask_by=var.get("mask_by"),
                    )
                )
                print(f"Gridded data for {var['name']}: {[str(p) for p in paths]}")
            elif var_type in NUTS_TYPES:
                if len(paths) != 1:
                    raise ValueError(
                        f"NUTS data for {var['name']} must be a single file."
                    )
                nuts_vars.append(NutsVar(name=var["name"], path=paths[0]))
                print(f"NUTS data for {var['name']}: {paths[0]}")
            else:
                raise ValueError(
                    f"Unknown type {var_type} for {var.get('name')} in the "
                    "configuration."
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
    load_data_with_optimization(engine, grid_vars=grid_vars, nuts_vars=nuts_vars)


def load_data_with_optimization(
    engine: engine.Engine,
    grid_vars: list[GridVar] | None = None,
    nuts_vars: list[NutsVar] | None = None,
) -> None:
    """
    Load data into the database with autovacuum disabled for performance.
    Ensures autovacuum is re-enabled even if an error occurs.
    """
    try:
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
            insert_grid_var_values(engine, grid_vars)
            print("DEBUG: insert_grid_var_values completed.")
        # insert the nuts variables data
        if nuts_vars:
            insert_nuts_var_values(engine, nuts_vars)
            print("DEBUG: insert_nuts_var_values completed.")
        print("DEBUG: All variable value inserts done, entering finally block.")

    finally:
        # Re-enable autovacuum and run VACUUM ANALYZE on all tables
        # This block is executed even if an error occurs in the try block
        print("DEBUG: Re-enabling autovacuum on all tables...")
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE grid_point SET (autovacuum_enabled = true);")
            )
            conn.execute(
                text("ALTER TABLE time_point SET (autovacuum_enabled = true);")
            )
            conn.execute(text("ALTER TABLE var_value SET (autovacuum_enabled = true);"))
            conn.execute(
                text("ALTER TABLE var_value_nuts SET (autovacuum_enabled = true);")
            )
        print("DEBUG: Autovacuum re-enabled on all tables.")

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
