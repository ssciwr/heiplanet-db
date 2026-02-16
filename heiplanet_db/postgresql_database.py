from sqlalchemy import (
    cast,
    create_engine,
    text,
    Float,
    String,
    Integer,
    Numeric,
    BigInteger,
    Index,
    ForeignKey,
    UniqueConstraint,
    ForeignKeyConstraint,
    engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.exc import SQLAlchemyError
from geoalchemy2 import Geometry, WKBElement
from sqlalchemy.orm.session import sessionmaker, Session
import geopandas as gpd
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import time
import os
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Type, Tuple, List
from fastapi import HTTPException
import json
import gc

CRS = 4326
STR_POINT = "SRID={};POINT({} {})"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 10000))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))
VAR_TIME_CHUNK = int(os.environ.get("VAR_TIME_CHUNK", 6))
GRID_LAT_CHUNK = int(os.environ.get("GRID_LAT_CHUNK", 45))
GRID_LON_CHUNK = int(os.environ.get("GRID_LON_CHUNK", 90))
ROUND_DIGITS = int(os.environ.get("ROUND_DIGITS", 4))
TIMEOUT = int(os.environ.get("TIMEOUT", 300))


def _q(x: float) -> float:
    return round(float(x), ROUND_DIGITS)


# Base declarative class
class Base(DeclarativeBase):
    """
    Base class for all models in the database."""

    pass


class NutsDef(Base):
    """
    NUTS definition table."""

    __tablename__ = "nuts_def"

    nuts_id: Mapped[String] = mapped_column(String(), primary_key=True)
    levl_code: Mapped[int] = mapped_column(Integer(), nullable=True)
    cntr_code: Mapped[String] = mapped_column(String(), nullable=True)
    name_latn: Mapped[String] = mapped_column(String(), nullable=True)
    nuts_name: Mapped[String] = mapped_column(String(), nullable=True)
    mount_type: Mapped[Float] = mapped_column(Float(), nullable=True)
    urbn_type: Mapped[Float] = mapped_column(Float(), nullable=True)
    coast_type: Mapped[Float] = mapped_column(Float(), nullable=True)
    geometry: Mapped[WKBElement] = mapped_column(
        Geometry(geometry_type="MULTIPOLYGON", srid=CRS)
    )


class GridPoint(Base):
    """
    Grid point table for storing latitude and longitude coordinates."""

    __tablename__ = "grid_point"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    latitude: Mapped[float] = mapped_column(Numeric(precision=8, scale=4))
    longitude: Mapped[float] = mapped_column(Numeric(precision=8, scale=4))

    # Geometry column for PostGIS
    point: Mapped[Geometry] = mapped_column(Geometry("POINT", srid=CRS), nullable=True)

    __table_args__ = (
        Index("idx_point_gridpoint", "point", postgresql_using="gist"),
        UniqueConstraint("latitude", "longitude", name="uq_lat_lon"),
    )

    def __init__(self, latitude, longitude, **kw):
        super().__init__(**kw)
        self.latitude = latitude
        self.longitude = longitude
        # add value of point automatically,
        # only works when using the constructor, i.e. session.add()
        self.point = func.ST_GeomFromText(
            STR_POINT.format(str(CRS), self.longitude, self.latitude)
        )


class TimePoint(Base):
    """
    Time point table for storing year, month, and day."""

    __tablename__ = "time_point"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    year: Mapped[int] = mapped_column(Integer())
    month: Mapped[int] = mapped_column(Integer())
    day: Mapped[int] = mapped_column(Integer())

    __table_args__ = (
        UniqueConstraint("year", "month", "day", name="uq_year_month_day"),
    )


class VarType(Base):
    """
    Variable type table for storing variable metadata."""

    __tablename__ = "var_type"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    name: Mapped[String] = mapped_column(String())
    unit: Mapped[String] = mapped_column(String())
    description: Mapped[String] = mapped_column(String(), nullable=True)

    __table_args__ = (UniqueConstraint("name", name="uq_var_name"),)


class VarValue(Base):
    """
    Variable value table for storing variable values at specific
    grid points and time points.
    """

    __tablename__ = "var_value"

    id: Mapped[int] = mapped_column(BigInteger(), primary_key=True, autoincrement=True)
    grid_id: Mapped[int] = mapped_column(Integer(), ForeignKey("grid_point.id"))
    time_id: Mapped[int] = mapped_column(Integer(), ForeignKey("time_point.id"))
    var_id: Mapped[int] = mapped_column(Integer(), ForeignKey("var_type.id"))
    # not sure if we should put resolution id here, since the same grid point
    # can belong to different resolution groups
    # so we have a separate many-to-many relationship table between grid points and resolution groups
    # however, this makes lookup somewhat more complex and maybe slower
    value: Mapped[float] = mapped_column(Float())

    __table_args__ = (
        UniqueConstraint("time_id", "grid_id", "var_id", name="uq_time_grid_var"),
        ForeignKeyConstraint(
            ["grid_id"],
            ["grid_point.id"],
            name="fk_grid_id",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["time_id"],
            ["time_point.id"],
            name="fk_time_id",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["var_id"],
            ["var_type.id"],
            name="fk_var_id",
            ondelete="CASCADE",
        ),
    )


class VarValueNuts(Base):
    """
    Variable value table for storing variable values at specific
    NUTS regions and time points.
    """

    __tablename__ = "var_value_nuts"

    id: Mapped[int] = mapped_column(BigInteger(), primary_key=True, autoincrement=True)
    nuts_id: Mapped[String] = mapped_column(String(), ForeignKey("nuts_def.nuts_id"))
    time_id: Mapped[int] = mapped_column(Integer(), ForeignKey("time_point.id"))
    var_id: Mapped[int] = mapped_column(Integer(), ForeignKey("var_type.id"))
    value: Mapped[float] = mapped_column(Float())

    __table_args__ = (
        UniqueConstraint("time_id", "nuts_id", "var_id", name="uq_time_nuts_var"),
        ForeignKeyConstraint(
            ["nuts_id"],
            ["nuts_def.nuts_id"],
            name="fk_nuts_id",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["time_id"],
            ["time_point.id"],
            name="fk_time_id_nuts",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["var_id"],
            ["var_type.id"],
            name="fk_var_id_nuts",
            ondelete="CASCADE",
        ),
    )


class ResolutionGroup(Base):
    """Resolution group for the different grid resolutions."""

    # create different grid resolution groups
    # 0.1, 0.25, 0.5 resolution degree resolution
    # ideally, we want values at 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5
    __tablename__ = "resolution_group"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    resolution: Mapped[float] = mapped_column(
        Numeric(precision=4, scale=2), unique=True
    )
    description: Mapped[str] = mapped_column(String(), nullable=True)


class GridPointResolution(Base):
    """Many-to-many relationship between GridPoint and ResolutionGroup"""

    __tablename__ = "grid_point_resolution"

    grid_id: Mapped[int] = mapped_column(
        Integer(), ForeignKey("grid_point.id"), primary_key=True
    )
    resolution_id: Mapped[int] = mapped_column(
        Integer(), ForeignKey("resolution_group.id"), primary_key=True
    )


def install_postgis(engine: engine.Engine):
    """
    Install PostGIS extension on the database.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
    """
    # use begin() to commit the transaction when operation is successful
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        print("PostGIS extension installed.")


def create_session(engine: engine.Engine) -> Session:
    """
    Create a new session for the database.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.

    Returns:
        Session: SQLAlchemy session object.
    """
    session_class = sessionmaker(bind=engine)
    return session_class()


def create_tables(engine: engine.Engine):
    """
    Create all tables in the database.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
    """
    Base.metadata.create_all(engine)
    print("All tables created.")


def create_or_replace_tables(engine: engine.Engine):
    """
    Create or replace tables in the database.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
    """
    Base.metadata.drop_all(engine)
    print("All tables dropped.")
    create_tables(engine)


def initialize_database(db_url: str, replace: bool = False):
    """
    Initialize the database by creating the engine and tables, and installing PostGIS.
    If replace is True, it will drop and recreate the tables.

    Args:
        db_url (str): Database URL for SQLAlchemy.
        replace (bool): Whether to drop and recreate the tables. Defaults to False.
    """
    # create engine
    engine = create_engine(db_url)  # remove echo=True to show just errors in terminal

    # install PostGIS extension
    install_postgis(engine)

    # create or replace tables
    if replace:
        create_or_replace_tables(engine)
    else:
        create_tables(engine)

    print("Database initialized successfully.")

    return engine


def insert_nuts_def(engine: engine.Engine, shapefiles_path: Path):
    """
    Insert NUTS definition data into the database.
    The shapefiles are downloaded from the Eurostat website.
    More details for downloading NUTS shapefiles can be found in
    [our data page](https://ssciwr.github.io/heiplanet-db/data/#eurostats-nuts-definition)

    Five shapefiles are involved in the process:
    - `.shp`: geometry data (e.g. polygons)
    - `.shx`: shape index data
    - `.dbf`: attribute data (e.g. names, codes)
    - `.prj`: projection data (i.e. CRS)
    - `.cpg`: character encoding data

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        shapefiles_path (Path): Path to the NUTS shapefiles.
    """
    nuts_data = gpd.GeoDataFrame.from_file(shapefiles_path)
    # rename columns to match the database schema
    nuts_data = nuts_data.rename(
        columns={
            "NUTS_ID": "nuts_id",
            "LEVL_CODE": "levl_code",
            "CNTR_CODE": "cntr_code",
            "NAME_LATN": "name_latn",
            "NUTS_NAME": "nuts_name",
            "MOUNT_TYPE": "mount_type",
            "URBN_TYPE": "urbn_type",
            "COAST_TYPE": "coast_type",
        }
    )

    # clean up the data first if nuts_def table already exists
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE nuts_def RESTART IDENTITY CASCADE"))

    # insert the data into the nuts_def table
    # here we do not use replace for if_exists because
    # the table var_value_nuts has a foreign key constraint
    # to nuts_def, so append would be safer
    nuts_data.to_postgis(NutsDef.__tablename__, engine, if_exists="append", index=False)
    print("NUTS definition data inserted.")


def add_data_list(session: Session, data_list: list):
    """
    Add a list of data instances to the database.

    Args:
        session (Session): SQLAlchemy session object.
        data_list (list): List of data instances to add.
    """
    try:
        session.add_all(data_list)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Error inserting data: {e}")


def add_data_list_bulk(session: Session, data_dict_list: list, class_type: Type[Base]):
    """
    Add a list of data to the database in bulk.

    Args:
        session (Session): SQLAlchemy session object.
        data_dict_list (list): List of dictionaries containing data to insert.
        class_type (Type[Base]): SQLAlchemy mapped class to insert data into.
    """
    try:
        session.bulk_insert_mappings(class_type, data_dict_list)
        session.commit()
        session.expire_all()  # clear identity map
    except SQLAlchemyError as e:
        session.rollback()
        raise RuntimeError(f"Error inserting data: {e}")


def insert_grid_points(session: Session, latitudes: np.ndarray, longitudes: np.ndarray):
    """
    Insert grid points into the database.

    Args:
        session (Session): SQLAlchemy session object.
        latitudes (np.ndarray): Array of latitudes.
        longitudes (np.ndarray): Array of longitudes.
    """
    total_batches = math.ceil(len(latitudes) / GRID_LAT_CHUNK) * math.ceil(
        len(longitudes) / GRID_LON_CHUNK
    )
    batch_count = 0
    for lat_start in range(0, len(latitudes), GRID_LAT_CHUNK):
        lat_end = min(lat_start + GRID_LAT_CHUNK, len(latitudes))
        lat_slice = latitudes[lat_start:lat_end]

        for lon_start in range(0, len(longitudes), GRID_LON_CHUNK):
            lon_end = min(lon_start + GRID_LON_CHUNK, len(longitudes))
            lon_slice = longitudes[lon_start:lon_end]

            # Build batch
            grid_points = [
                {
                    "latitude": _q(lat),
                    "longitude": _q(lon),
                    "point": STR_POINT.format(str(CRS), _q(lon), _q(lat)),
                }
                for lat in lat_slice
                for lon in lon_slice
            ]

            # Insert batch
            add_data_list_bulk(session, grid_points, GridPoint)
            session.expire_all()  # clear identity map
            batch_count += 1
            print(f"Grid points batch {batch_count}/{total_batches} inserted.")

    print("All grid points inserted.")


def insert_resolution_groups(
    session: Session,
    resolutions: np.ndarray = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]),
    descriptions: list[str] | None = None,
) -> None:
    """Create the resolution groups.

    There are different degrees resolution that can be requested:
    0.1 degree, 0.2/0.5/1.0/1.5/2.0/2.5/3.0/5.0 degrees.
    We are currently using 0.2 degree resolution and not 0.25 degree resolution here,
    since this is a subset of 0.1 degree resolution grid points. Otherwise we would
    need to create additional grid points for 0.25 degree resolution through interpolation,
    which has not been defined as to which interpolation method can be used for this.

    Args:
        session (Session): SQLAlchemy session object.
        resolutions (np.ndarray): Array of resolutions to insert. Defaults to
            0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0 degree resolution.
        descriptions (list[str]|None): List of descriptions for each resolution.
            If None, default descriptions will be used.
    """
    if descriptions is None:
        descriptions = [f"{res} degree resolution" for res in resolutions]
    # create list of dictionaries for bulk insert
    resolution_groups = [
        {
            "resolution": float(resolution),
            "description": description,
        }
        for resolution, description in zip(resolutions, descriptions)
    ]
    add_data_list_bulk(session, resolution_groups, ResolutionGroup)
    print("Resolution groups inserted:", resolutions)


def assign_grid_resolution_group_to_grid_point(session: Session) -> None:
    """
    Assign the grid resolution group to each grid point,
    creating the many-to-many relationship between resolutions and
    grid points.

    Args:
        session (Session): SQLAlchemy session object.
    """
    # we assume that by multiplying the resolution, and the
    # lat/long values by 10, we can use the modulo operator to
    # map the grid points to the resolution groups
    # also we are using exact equality check since the way it is stored on postgresql
    # should be exact and not numerically imprecise
    grid_point_resolutions = []
    # get the resolution groups from the database
    resolution_groups = session.query(ResolutionGroup).all()
    if not resolution_groups:
        raise ValueError("No resolution groups found in the database.")
    for resolution_group in resolution_groups:
        resolution = float(resolution_group.resolution) * 10
        grid_points = (
            session.query(GridPoint)
            .filter(
                (cast(func.round(GridPoint.latitude * 10), Integer) % resolution == 0)
                & (
                    cast(func.round(GridPoint.longitude * 10), Integer) % resolution
                    == 0
                )
            )
            .all()
        )
        if not grid_points:
            raise ValueError(
                f"No matching grid points for {resolution / 10} degree resolution found."
            )
        grid_point_resolutions.extend(
            [
                {
                    "grid_id": grid_point.id,
                    "resolution_id": resolution_group.id,
                }
                for grid_point in grid_points
            ]
        )
    add_data_list_bulk(session, grid_point_resolutions, GridPointResolution)
    print("Grid point resolutions assigned.")


def extract_time_point(
    time_point: np.datetime64,
) -> tuple[int, int, int, int, int, int]:
    """
    Extract year, month, and day from a numpy datetime64 object.

    Args:
        time_point (np.datetime64): Numpy datetime64 object representing a time point.

    Returns:
        tuple: A tuple containing year, month, day, hour, minute, second.
    """
    if isinstance(time_point, np.datetime64):
        time_stamp = pd.Timestamp(time_point)
        return (
            time_stamp.year,
            time_stamp.month,
            time_stamp.day,
            time_stamp.hour,
            time_stamp.minute,
            time_stamp.second,
        )
    else:
        raise ValueError("Invalid time point format.")


def get_unique_time_points(
    time_point_data: list[Tuple[np.ndarray, bool]],
) -> np.ndarray:
    """Get the unique of time points.

    Args:
        time_point_data: List of tuples containing time point data, and the yearly flag.
            If flag is True, the time point needs to be converted to monthly.

    Returns:
        np.ndarray: Unique of (sorted) time points as a numpy array.
    """
    time_points = []
    for tpd, yearly in time_point_data:
        if not yearly:
            # assume it's monthly TODO
            time_points.append(tpd)
        else:
            # convert to monthly for the whole range
            if np.datetime64(tpd[0]) > np.datetime64(tpd[-1]):
                # sort before converting
                tpd = np.sort(tpd)

            start_of_year = pd.Timestamp(
                year=extract_time_point(np.datetime64(tpd[0]))[0], month=1, day=1
            )
            end_of_year = pd.Timestamp(
                year=extract_time_point(np.datetime64(tpd[-1]))[0], month=12, day=1
            )
            time_points.append(
                pd.date_range(start=start_of_year, end=end_of_year, freq="MS").values
            )

    if not time_points:
        return np.array([])

    concatenated = np.concatenate(time_points)
    unique_time_points = np.unique(concatenated)
    return np.sort(unique_time_points)


def insert_time_points(
    session: Session, time_point_data: list[Tuple[np.ndarray, bool]]
):
    """Insert time points into the database.

    Args:
        session (Session): SQLAlchemy session object.
        time_point_data (list[(np.ndarray, bool)]): List of tuples containing
            time point data, and its flag.
            If flag is True, the time point needs to be converted to monthly.
    """
    time_point_values = []
    # get the overlap of the time points
    time_points = get_unique_time_points(time_point_data)

    # extract year, month, day from the time points
    for time_point in time_points:
        year, month, day, _, _, _ = extract_time_point(time_point)
        if year is not None and month is not None and day is not None:
            time_point_values.append(
                {
                    "year": year,
                    "month": month,
                    "day": day,
                }
            )

    add_data_list_bulk(session, time_point_values, TimePoint)
    print("Time points inserted.")


def insert_var_types(session: Session, var_types: list[dict]):
    """
    Insert variable types into the database.

    Args:
        session (Session): SQLAlchemy session object.
        var_types (list[dict]): List of dictionaries containing variable type data.
    """
    add_data_list_bulk(session, var_types, VarType)
    print("Variable types inserted.")


def get_id_maps(session: Session) -> tuple[dict, dict, dict]:
    """
    Get ID maps for grid points, time points, and variable types.
    Uses streaming to avoid loading all data into memory at once.

    Args:
        session (Session): SQLAlchemy session object.

    Returns:
        tuple: A tuple containing three dictionaries:\n
            - grid_id_map: Mapping of (latitude, longitude) to grid point ID.\n
            - time_id_map: Mapping of datetime64 to time point ID.\n
            - var_id_map: Mapping of variable name to variable type ID.
    """
    grid_id_map = {}
    time_id_map = {}
    var_id_map = {}

    # Stream grid points in batches
    grid_point_batch_size = 100000
    for offset in range(0, session.query(GridPoint).count(), grid_point_batch_size):
        grid_points = (
            session.query(GridPoint.id, GridPoint.latitude, GridPoint.longitude)
            .order_by(GridPoint.id)
            .offset(offset)
            .limit(grid_point_batch_size)
            .all()
        )

        for grid_id, lat, lon in grid_points:
            grid_id_map[(_q(lat), _q(lon))] = grid_id

        session.expire_all()  # Clear session cache

    # Stream time points in batches
    time_point_batch_size = 10000
    for offset in range(0, session.query(TimePoint).count(), time_point_batch_size):
        time_points = (
            session.query(TimePoint)
            .order_by(TimePoint.id)
            .offset(offset)
            .limit(time_point_batch_size)
            .all()
        )

        for row in time_points:
            time_id_map[
                np.datetime64(pd.to_datetime(f"{row.year}-{row.month}-{row.day}"), "ns")
            ] = row.id

        session.expire_all()

    # Stream variable types in batches
    var_type_batch_size = 1000
    for offset in range(0, session.query(VarType).count(), var_type_batch_size):
        var_types = (
            session.query(VarType)
            .order_by(VarType.id)
            .offset(offset)
            .limit(var_type_batch_size)
            .all()
        )

        for row in var_types:
            var_id_map[row.name] = row.id

        session.expire_all()

    return grid_id_map, time_id_map, var_id_map


def convert_yearly_to_monthly(ds: xr.Dataset) -> xr.Dataset:
    """Convert yearly data to monthly data.

    Args:
        ds (xr.Dataset): xarray dataset with yearly data.

    Returns:
        xr.Dataset: xarray dataset with monthly data.
    """
    if ds.time.values[0] > ds.time.values[-1]:
        # sort the time points
        ds = ds.sortby("time")

    # create monthly time points
    s_y, s_m, _, s_h, s_mi, s_s = extract_time_point(ds.time.values[0])
    e_y, _, _, e_h, e_mi, e_s = extract_time_point(ds.time.values[-1])
    new_time_points = pd.date_range(
        start=pd.Timestamp(
            year=s_y, month=s_m, day=1, hour=s_h, minute=s_mi, second=s_s
        ),
        end=pd.Timestamp(year=e_y, month=12, day=1, hour=e_h, minute=e_mi, second=e_s),
        freq="MS",
    )

    # reindex dataset with new time points
    return ds.reindex(time=new_time_points, method="ffill")


def insert_batch(batch: list[VarValue], engine: engine.Engine, VarClass: Type[Base]):
    session = create_session(engine)
    try:
        add_data_list_bulk(session, batch, VarClass)
        session.commit()
    finally:
        session.close()


def insert_var_values(
    engine: engine.Engine,
    ds: xr.Dataset,
    var_name: str,
    grid_id_map: dict,
    time_id_map: dict,
    var_id_map: dict,
    to_monthly: bool = False,
) -> tuple[float, float]:
    """Insert variable values into the database in streaming chunks."""

    if to_monthly:
        print(f"Converting {var_name} data from yearly to monthly...")
        ds = convert_yearly_to_monthly(ds)
    t_yearly_to_monthly = time.time()

    print(f"Prepare inserting {var_name} values...")

    var_id = var_id_map.get(var_name)
    if var_id is None:
        raise ValueError(f"Variable {var_name} not found in var_type table.")

    print(f"Start inserting {var_name} values in streaming chunks...")
    t_start_insert = time.time()

    generate_threaded_inserts(
        VAR_TIME_CHUNK,
        GRID_LAT_CHUNK,
        GRID_LON_CHUNK,
        ds,
        var_name,
        grid_id_map,
        time_id_map,
        var_id,
        engine,
    )

    return t_yearly_to_monthly, t_start_insert


def generate_threaded_inserts(
    t_chunk: int,
    lat_chunk: int,
    lon_chunk: int,
    ds: xr.Dataset,
    var_name: str,
    grid_id_map: dict,
    time_id_map: dict,
    var_id: int,
    engine: engine.Engine,
) -> int:
    """Generate threaded inserts for variable values.

    Args:
        t_chunk (int): Time chunk size.
        lat_chunk (int): Latitude chunk size.
        lon_chunk (int): Longitude chunk size.
        ds (xr.Dataset): xarray dataset.
        var_name (str): Variable name.
        grid_id_map (dict): Grid ID map.
        time_id_map (dict): Time ID map.
        var_id (int): Variable ID.
        engine (engine.Engine): SQLAlchemy engine object.
    Returns:
        int: Total number of variable values inserted.
    """
    futures = []
    total_values_inserted = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # the dimension is (time, latitude, longitude)
        for t_idx in range(0, ds.sizes["time"], t_chunk):
            for lat_idx in range(0, ds.sizes["latitude"], lat_chunk):
                for lon_idx in range(0, ds.sizes["longitude"], lon_chunk):
                    t_end = min(t_idx + t_chunk, ds.sizes["time"])
                    lat_end = min(lat_idx + lat_chunk, ds.sizes["latitude"])
                    lon_end = min(lon_idx + lon_chunk, ds.sizes["longitude"])

                    ds_chunk = ds.isel(
                        time=slice(t_idx, t_end),
                        latitude=slice(lat_idx, lat_end),
                        longitude=slice(lon_idx, lon_end),
                    )
                    var_data = ds_chunk[var_name].load()
                    # drop all empty latitude rows
                    var_data = var_data.dropna(dim="latitude", how="all")
                    # skip inserting if the chunk is empty
                    if var_data.size == 0:
                        continue

                    # make sure time has the correct format for db insertion
                    # so that values can be requested correctly later
                    times = var_data.time.values.astype("datetime64[ns]")
                    lats = var_data.latitude.values
                    lons = var_data.longitude.values
                    values = var_data.values

                    # skip insertion if chunk is somehow corrupted
                    # I think this case should not happen
                    # maybe this is where the values go missing
                    if values.ndim != 3:
                        continue
                    var_values = get_var_values_mapping(
                        times, time_id_map, grid_id_map, var_id, lats, lons, values
                    )
                    if var_values:
                        for i in range(0, len(var_values), BATCH_SIZE // 2):
                            batch = var_values[i : i + BATCH_SIZE // 2]
                            future = executor.submit(
                                insert_batch, batch, engine, VarValue
                            )
                            futures.append(future)
                            total_values_inserted += len(batch)

                    del var_data, var_values, times, lats, lons, values
                    gc.collect()

        # Wait for all futures to complete with timeout
        print(f"Waiting for {len(futures)} batches to complete...")
        for future in tqdm(as_completed(futures, timeout=TIMEOUT), total=len(futures)):
            try:
                future.result()  # Ensure exceptions are raised
            except Exception as e:
                print(f"Error in batch: {e}")
                raise

    print(f"Values of {var_name} inserted. Total: {total_values_inserted}")
    return total_values_inserted


def get_var_values_mapping(
    times: np.ndarray,
    time_id_map: dict,
    grid_id_map: dict,
    var_id: int,
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
) -> list[dict]:
    """Get variable values mapping for database insertion.

    Args:
        times (np.ndarray): Array of time points.
        time_id_map (dict): Mapping of datetime64 to time point ID.
        grid_id_map (dict): Mapping of (latitude, longitude) to grid point ID.
        var_id (int): Variable type ID.
        lats (np.ndarray): Array of latitudes.
        lons (np.ndarray): Array of longitudes.
        values (np.ndarray): Array of variable values.
    Returns:
        list[dict]: List of variable values mapping for database insertion.
    """
    var_values = []
    # Loop order: time, lat, lon to match xarray dimension order (time, latitude, longitude)
    for t_i in range(len(times)):
        t_val = times[t_i]
        ts = pd.Timestamp(t_val)
        time_key = np.datetime64(pd.to_datetime(f"{ts.year}-{ts.month}-{ts.day}"), "ns")
        time_id = time_id_map.get(time_key)
        if time_id is None:
            print(
                f"Missing time_id for {time_key}. Available keys sample: {list(time_id_map.keys())[:5]}"
            )
            continue
        for lat_i in range(len(lats)):
            lat_val = _q(lats[lat_i])
            for lon_i in range(len(lons)):
                lon_val = _q(lons[lon_i])
                val = float(values[t_i, lat_i, lon_i])  # Correct indexing order
                if np.isnan(val):
                    continue
                grid_id = grid_id_map.get((lat_val, lon_val))
                if grid_id is None:
                    print(f"Missing grid_id for ({lat_val}, {lon_val})")

                if grid_id is not None:
                    var_values.append(
                        {
                            "grid_id": int(grid_id),
                            "time_id": int(time_id),
                            "var_id": int(var_id),
                            "value": float(val),
                        }
                    )
    return var_values


def get_var_value(
    session: Session,
    var_name: str,
    lat: float,
    lon: float,
    year: int,
    month: int,
    day: int,
) -> float | int | str | None:
    """Get variable value from the database.

    Args:
        session (Session): SQLAlchemy session object.
        var_name (str): Name of the variable to retrieve.
        lat (float): Latitude of the grid point.
        lon (float): Longitude of the grid point.
        year (int): Year of the time point.
        month (int): Month of the time point.
        day (int): Day of the time point.

    Returns:
        float | int | str | None: Value of the variable at
            the specified grid point and time point.
    """
    if day != 1:
        print(
            "The current database only supports monthly data."
            "Retieving data for the first day of the month..."
        )
        day = 1

    lat = _q(lat)
    lon = _q(lon)

    result = (
        session.query(VarValue)
        .join(GridPoint, VarValue.grid_id == GridPoint.id)
        .join(TimePoint, VarValue.time_id == TimePoint.id)
        .join(VarType, VarValue.var_id == VarType.id)
        .filter(
            GridPoint.latitude == lat,
            GridPoint.longitude == lon,
            TimePoint.year == year,
            TimePoint.month == month,
            TimePoint.day == day,
            VarType.name == var_name,
        )
        .first()
    )
    return result.value if result else None


def get_var_value_nuts(
    session: Session,
    var_name: str,
    nuts_region: str,
    year: int,
    month: int,
    day: int,
) -> float | int | str | None:
    """Get variable value from the database.

    Args:
        session (Session): SQLAlchemy session object.
        var_name (str): Name of the variable to retrieve.
        nuts_region (str): NUTS region code.
        year (int): Year of the time point.
        month (int): Month of the time point.
        day (int): Day of the time point.

    Returns:
        float | int | str | None: Value of the variable at
            the specified grid point and time point.
    """
    if day != 1:
        print(
            "The current database only supports monthly data."
            "Retieving data for the first day of the month..."
        )
        day = 1
    print(f"Retrieving {var_name} for {nuts_region} at {year}-{month}-{day}...")
    result = (
        session.query(VarValueNuts)
        .join(NutsDef, VarValueNuts.nuts_id == NutsDef.nuts_id)
        .join(TimePoint, VarValueNuts.time_id == TimePoint.id)
        .join(VarType, VarValueNuts.var_id == VarType.id)
        .filter(
            NutsDef.nuts_id == nuts_region,
            TimePoint.year == year,
            TimePoint.month == month,
            TimePoint.day == day,
            VarType.name == var_name,
        )
        .first()
    )
    return result.value if result else None


def get_time_points(
    session: Session,
    start_time_point: Tuple[int, int],
    end_time_point: Tuple[int, int] | None = None,
) -> List[TimePoint]:
    """Get time points from the database that fall within a specified range.

    Args:
        session (Session): SQLAlchemy session object.
        start_time_point (Tuple[int, int]): Start time point as (year, month).
        end_time_point (Tuple[int, int] | None): End time point as (year, month).
            If None, only the start time point is used.

    Returns:
        List[TimePoint]: List of TimePoint objects within the specified range.
    """
    if end_time_point is None:
        end_time_point = start_time_point

    return (
        session.query(TimePoint)
        .filter(
            (TimePoint.year > start_time_point[0])
            | (
                (TimePoint.year == start_time_point[0])
                & (TimePoint.month >= start_time_point[1])
            ),
            (TimePoint.year < end_time_point[0])
            | (
                (TimePoint.year == end_time_point[0])
                & (TimePoint.month <= end_time_point[1])
            ),
        )
        .all()
    )


def get_resolution_id(session: Session, resolution: float) -> int | None:
    """Get the resolution ID for a given resolution value.

    Args:
        session (Session): SQLAlchemy session object.
        resolution (float): Resolution value to look up.

    Returns:
        int | None: Resolution ID if found, None otherwise.
    """
    resolution_group = (
        session.query(ResolutionGroup)
        .filter(ResolutionGroup.resolution == resolution)
        .first()
    )
    return resolution_group.id if resolution_group else None


def get_grid_ids_by_resolution(session: Session, resolution_id: int) -> List[int]:
    """Get all grid point IDs for a specific resolution.

    Args:
        session (Session): SQLAlchemy session object.
        resolution_id (int): Resolution ID to filter by.

    Returns:
        List[int]: List of grid point IDs belonging to the specified resolution.
    """
    grid_point_resolutions = (
        session.query(GridPointResolution.grid_id)
        .filter(GridPointResolution.resolution_id == resolution_id)
        .all()
    )
    return [gpr.grid_id for gpr in grid_point_resolutions]


def get_grid_points(
    session: Session,
    area: None | Tuple[float, float, float, float] = None,
    resolution_id: int | None = None,
) -> List[GridPoint]:
    """Get grid points from the database that fall within a specified area.
    Args:
        session (Session): SQLAlchemy session object.
        area (None | Tuple[float, float, float, float]):
            Area as (North, West, South, East).
            If None, all grid points are returned.
        resolution_id (int | None): Resolution ID of the grid points. Defaults to None.
    Returns:
        List[GridPoint]: List of GridPoint objects within the specified area.
    """

    query = session.query(GridPoint)

    # Filter by resolution if specified
    if resolution_id is not None:
        query = query.join(
            GridPointResolution, GridPoint.id == GridPointResolution.grid_id
        ).filter(GridPointResolution.resolution_id == resolution_id)

    # Filter by area if specified
    if area is not None:
        north, west, south, east = area
        query = query.filter(
            GridPoint.latitude <= north,
            GridPoint.latitude >= south,
            GridPoint.longitude >= west,
            GridPoint.longitude <= east,
        )

    return query.all()


def get_var_types(
    session: Session,
    var_names: None | List[str] = None,
) -> List[VarType]:
    """Get variable types from the database with names specified in a list.

    Args:
        session (Session): SQLAlchemy session object.
        var_names (None | List[str]): List of variable names to filter by.
            If None, all variable types are returned.

    Returns:
        List[VarType]: List of VarType objects with the specified names.
    """
    if var_names is None:
        return session.query(VarType).all()

    return session.query(VarType).filter(VarType.name.in_(var_names)).all()


def sort_grid_points_get_ids(
    grid_points: List[GridPoint],
) -> tuple[dict, list[float], list[float]]:
    # Sort and deduplicate latitudes and longitudes
    latitudes = sorted({_q(gp.latitude) for gp in grid_points})
    longitudes = sorted({_q(gp.longitude) for gp in grid_points})

    # Create fast index maps for latitude and longitude
    lat_to_index = {lat: i for i, lat in enumerate(latitudes)}
    lon_to_index = {lon: i for i, lon in enumerate(longitudes)}

    # Map grid_id to (lat_index, lon_index)
    grid_ids = {
        gp.id: (lat_to_index[_q(gp.latitude)], lon_to_index[_q(gp.longitude)])
        for gp in grid_points
    }
    return grid_ids, latitudes, longitudes


def get_var_values_cartesian(
    session: Session,
    time_point: Tuple[int, int],
    grid_resolution: float = 0.1,
    area: None | Tuple[float, float, float, float] = None,
    var_name: None | str = None,
) -> dict:
    """Get variable values for a cartesian map.

    Args:
        session (Session): SQLAlchemy session object.
        time_point (Tuple[int, int]): Date point as (year, month).
        grid_resolution (float): Resolution of the grid points. Defaults to 0.1 degree.
        area (None | Tuple[float, float, float, float]):
            Area as (North, West, South, East).
            If None, all grid points are used.
        var_name (None | str): Variable name for which values should be returned.
         If None, the default model values will be returned.

    Returns:
        dict: a dict with (latitude, longitude, var_value) for the requested date.
    """
    # get the time points and their ids
    date_object = (
        session.query(TimePoint)
        .filter((TimePoint.year == time_point[0]) & (TimePoint.month == time_point[1]))
        .first()
    )

    if not date_object:
        print("No time point found for the specified date.")
        raise HTTPException(status_code=400, detail="Missing data for requested date.")

    # get time id
    time_id = date_object.id

    # get the resolution id for the targeted resolution
    resolution_id = get_resolution_id(session=session, resolution=grid_resolution)
    if not resolution_id:
        print("No resolution id found for requested resolution.")
        raise HTTPException(
            status_code=400, detail="No id found for specified resolution."
        )

    # get the grid points and their ids
    grid_points = get_grid_points(session, area, resolution_id)

    if not grid_points:
        print(
            "No grid points found in the specified area or with the specified resolution."
        )
        raise HTTPException(
            status_code=400,
            detail="No grid points found in specified area or with the specified resolution.",
        )
    # get grid ids for lookup
    grid_ids = [grid_point.id for grid_point in grid_points]

    # get the var type
    if not var_name:
        var_name = "t2m"  # default variable name
    var_type = session.query(VarType).filter(VarType.name == var_name).first()
    if not var_type:
        print("No variable type found for the specified name.")
        raise HTTPException(
            status_code=400, detail="Missing variable type for requested time point."
        )

    # get the variable type id
    var_id = var_type.id

    # now query all variable values with their latitude and longitude for this time point
    # get variable values for each grid point and time point
    # Query with JOIN to get lat, lon, and value directly
    values = (
        session.query(GridPoint.latitude, GridPoint.longitude, VarValue.value)
        .join(GridPoint, VarValue.grid_id == GridPoint.id)
        .filter(
            VarValue.grid_id.in_(grid_ids),
            VarValue.time_id == time_id,
            VarValue.var_id == var_id,
        )
        .order_by(GridPoint.latitude, GridPoint.longitude)  # Ensure consistent ordering
        .all()
    )
    # Convert directly to list of tuples
    values_list = [(np.float64(lat), np.float64(lon), val) for lat, lon, val in values]

    mydict = {"latitude, longitude, var_value": values_list}
    return mydict


def get_var_values_cartesian_for_download(
    session: Session,
    start_time_point: Tuple[int, int],
    end_time_point: Tuple[int, int] | None = None,
    area: None | Tuple[float, float, float, float] = None,
    var_names: None | List[str] = None,
    netcdf_file: str = "cartesian_grid_data_heiplanet.nc",
) -> dict:
    """Get variable values for a cartesian map.

    Args:
        session (Session): SQLAlchemy session object.
        start_time_point (Tuple[int, int]): Start time point as (year, month).
        end_time_point (Tuple[int, int] | None): End time point as (year, month).
            If None, only the start time point is used.
        area (None | Tuple[float, float, float, float]):
            Area as (North, West, South, East).
            If None, all grid points are used.
        var_names (None | List[str]): List of variable names to filter by.
            If None, all variable types are used.
        netcdf_file (str): Name of the NetCDF file to save the dataset.

    Returns:
        dict: a dict with (time, latitude, longitude, var_value) keys.
            time or var_value is empty if no data is found.
    """
    # get the time points and their ids
    time_points = get_time_points(session, start_time_point, end_time_point)

    if not time_points:
        print("No time points found in the specified range.")
        raise HTTPException(
            status_code=400, detail="Missing data for requested time point."
        )

    # create a list of time points and their ids
    time_values = [
        np.datetime64(pd.Timestamp(year=tp.year, month=tp.month, day=1), "ns")
        for tp in time_points
    ]
    time_ids = {tp.id: tidx for tidx, tp in enumerate(time_points)}

    # get the grid points and their ids
    grid_points = get_grid_points(session, area)

    if not grid_points:
        print("No grid points found in the specified area.")
        raise HTTPException(
            status_code=400, detail="No grid points found in specified area."
        )

    # Sort and deduplicate latitudes and longitudes
    grid_ids, latitudes, longitudes = sort_grid_points_get_ids(grid_points)

    # Force netCDF-safe coordinate dtypes
    time_values = np.asarray(time_values, dtype="datetime64[ns]")
    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)

    # get variable types and their ids
    var_types = get_var_types(session, var_names)
    if not var_types:
        print("No variable types found in the specified names.")
        raise HTTPException(
            status_code=400, detail="Missing variable type for requested time point."
        )

    # create an empty dataset
    ds = xr.Dataset(
        coords={
            "time": ("time", time_values),
            "latitude": ("latitude", latitudes),
            "longitude": ("longitude", longitudes),
        }
    )

    # get variable values for each grid point and time point
    for vt in var_types:
        var_name = vt.name
        values = (
            session.query(VarValue)
            .filter(
                VarValue.grid_id.in_(grid_ids.keys()),
                VarValue.time_id.in_(time_ids.keys()),
                VarValue.var_id == vt.id,
            )
            .all()
        )

        # dummy values array
        values_array = np.full(
            (len(time_values), len(latitudes), len(longitudes)),
            np.nan,
            dtype=np.float64,
        )

        # fill the values array with the variable values
        for vv in values:
            grid_index = grid_ids[vv.grid_id]
            lat_index, lon_index = grid_index
            time_index = time_ids[vv.time_id]
            values_array[time_index, lat_index, lon_index] = vv.value

        # add data to the dataset
        ds[var_name] = (("time", "latitude", "longitude"), values_array)

    # add variable attributes
    for var_type in var_types:
        ds[var_type.name].attrs["unit"] = var_type.unit
        ds[var_type.name].attrs["description"] = var_type.description
    # add global attributes
    ds.attrs["source"] = "OneHealth Database"
    ds.attrs["created_at"] = pd.Timestamp.now().isoformat()
    ds.attrs["description"] = "Variable values for a cartesian map from the database."

    # save the dataset to a NetCDF file
    ds.to_netcdf(netcdf_file)
    print(f"Dataset saved to {netcdf_file}")

    return {"response": "Dataset created successfully.", "netcdf_file": netcdf_file}


def get_nuts_regions(
    engine: engine.Engine,
) -> gpd.GeoDataFrame:
    """Get NUTS regions from the database.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with NUTS region attributes and geometries.
    """
    return gpd.read_postgis("SELECT * FROM nuts_def", engine, geom_col="geometry")


def get_nuts_regions_geojson(
    engine: engine.Engine, grid_resolution: str | None = None
) -> dict:
    """
    Return NUTS regions as GeoJSON, optionally filtered by resolution.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        grid_resolution (str | None, optional): NUTS resolution level to filter by.
            Must be one of 'NUTS0', 'NUTS1', 'NUTS2', or 'NUTS3'. If None, all levels are returned.

    Returns:
        dict: GeoJSON representation of the NUTS regions.
    """
    nuts_regions = get_nuts_regions(engine)
    if nuts_regions.empty:
        raise HTTPException(
            status_code=404, detail="No NUTS regions found in the database."
        )

    if grid_resolution:
        level_map = {
            "NUTS0": 0,
            "NUTS1": 1,
            "NUTS2": 2,
            "NUTS3": 3,
        }
        if grid_resolution not in level_map:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid grid resolution. Use one of "
                    "'NUTS0', 'NUTS1', 'NUTS2', 'NUTS3'."
                ),
            )
        nuts_regions = nuts_regions[
            nuts_regions["levl_code"] == level_map[grid_resolution]
        ]
        if nuts_regions.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No NUTS regions found for resolution {grid_resolution}.",
            )

    return json.loads(nuts_regions.to_json())


def get_grid_ids_in_nuts(
    engine: engine.Engine,
    nuts_regions: gpd.GeoDataFrame,
) -> List[int]:
    """Get grid point IDs that are within the NUTS regions.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        nuts_regions (gpd.GeoDataFrame): GeoDataFrame with NUTS region geometries.

    Returns:
        List[int]: List of grid point IDs that intersect with the NUTS regions.
    """
    if nuts_regions.empty:
        return []

    sql = """
    SELECT id, point as geometry
    FROM grid_point
    """

    # turn the grid points into a GeoDataFrame
    grid_points_gdf = gpd.read_postgis(
        sql,
        engine,
        geom_col="geometry",
        crs=f"EPSG:{str(CRS)}",
    )

    # filter grid points that intersect with NUTS regions
    filtered_grid_points_gdf = gpd.sjoin(
        grid_points_gdf, nuts_regions, how="inner", predicate="intersects"
    )

    return sorted(set(filtered_grid_points_gdf["id"].tolist()))


def filter_nuts_ids_for_resolution(nuts_ids: List[str], resolution: str) -> List[str]:
    """Filter NUTS IDs based on the specified resolution.

    Args:
        nuts_ids (List[str]): List of NUTS IDs to filter.
        resolution (str): Desired NUTS resolution ("NUTS0", "NUTS1", "NUTS2", "NUTS3").

    Returns:
        List[str]: Filtered list of NUTS IDs matching the specified resolution.
    """
    if resolution not in {"NUTS0", "NUTS1", "NUTS2", "NUTS3"}:
        raise ValueError(
            "Invalid resolution. Must be one of 'NUTS0', 'NUTS1', 'NUTS2', 'NUTS3'."
        )

    level_map = {
        "NUTS0": 2,
        "NUTS1": 3,
        "NUTS2": 4,
        "NUTS3": 5,
    }
    level = level_map[resolution]

    filtered_nuts_ids = [nid for nid in nuts_ids if len(nid) == level]
    return filtered_nuts_ids


def get_var_values_nuts(
    session: Session,
    time_point: Tuple[int, int],
    var_name: None | str = None,
    grid_resolution: str = "NUTS2",
) -> dict:
    """Get variable values for all two-digit NUTS regions.

    Args:
        session (Session): SQLAlchemy session object.
        time_point (Tuple[int, int]): Date point as (year, month).
        var_name (None | str): Variable name for which values should be returned.
            If None, the default model values will be returned.
        grid_resolution (str): Grid resolution, by default "NUTS2" is returned.

    Returns:
        dict: A dict with (NUTS_id: var_value) for the requested date and variable type.
            The NUTS id is the two-digit nuts abbreviation for the regions.
    """

    # get the time point and its id
    date_object = (
        session.query(TimePoint)
        .filter((TimePoint.year == time_point[0]) & (TimePoint.month == time_point[1]))
        .first()
    )

    if not date_object:
        print("No time point found for the specified date.")
        raise HTTPException(status_code=400, detail="Missing data for requested date.")

    # get time id
    time_id = date_object.id
    # get the var type
    if not var_name:
        var_name = "t2m"  # default variable name
    var_type = session.query(VarType).filter(VarType.name == var_name).first()
    if not var_type:
        print("No variable type found for the specified name.")
        raise HTTPException(
            status_code=400, detail="Missing variable type for requested time point."
        )

    # get the variable type id
    var_id = var_type.id

    # now query all variable values with their nuts_id for this time point
    values = (
        session.query(VarValueNuts)
        .filter(VarValueNuts.time_id == time_id, VarValueNuts.var_id == var_id)
        .all()
    )
    # now get all the nuts ids associated with the values
    nuts_ids = [v.nuts_id for v in values]
    if not nuts_ids:
        print("No NUTS id's found in the database.")
        raise HTTPException(
            status_code=400, detail="No NUTS ids found in the database."
        )
    # filter the nuts ids to the desired resolution
    # this could also be done based on the NutsDef table "levl_code"
    nuts_ids = filter_nuts_ids_for_resolution(nuts_ids, grid_resolution)
    if not nuts_ids:
        print(f"No {grid_resolution} NUTS id's found in the database.")
        raise HTTPException(
            status_code=400,
            detail=f"No {grid_resolution} NUTS ids found in the database.",
        )
    # create a dict with NUTS_id: var_value
    mydict = {nuts_id: v.value for v, nuts_id in zip(values, nuts_ids)}
    return mydict


def insert_var_value_nuts(
    engine: engine.Engine,
    ds: xr.Dataset,
    var_name: str,
    time_id_map: dict,
    var_id_map: dict,
) -> float:
    """Insert variable values for NUTS regions into the database.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        ds (xr.Dataset): xarray dataset with dimensions (time, NUTS_ID).
        var_name (str): Name of the variable to insert.
        time_id_map (dict): Mapping of time points to IDs.
        var_id_map (dict): Mapping of variable names to variable type IDs.

    Returns:
        float: The time taken to insert the variable values.
    """
    # get the variable id
    var_id = var_id_map.get(var_name)
    if var_id is None:
        raise ValueError(f"Variable {var_name} not found in var_type table.")

    # values of the variable
    var_data = (
        ds[var_name].dropna(dim="NUTS_ID", how="all").load()
    )  # load data into memory

    # using stack() from xarray to vectorize the data
    stacked_var_data = var_data.stack(points=("time", "NUTS_ID"))
    stacked_var_data = stacked_var_data.dropna("points")

    # get values of each dim
    time_vals = stacked_var_data["time"].values.astype("datetime64[ns]")
    nuts_ids = stacked_var_data["NUTS_ID"].values

    # create vectorized mapping
    # normalize time before mapping as the time in isimip is 12:00:00
    # TODO: find an optimal way to do this
    get_time_id = np.vectorize(
        lambda t: time_id_map.get(np.datetime64(pd.Timestamp(t).normalize(), "ns"))
    )

    time_ids = get_time_id(time_vals)
    values = stacked_var_data.values.astype(float)

    # create a mask for valid values
    masks = ~np.isnan(values)

    # create bulk data for insertion
    var_values = [
        {
            "nuts_id": str(nuts_id),
            "time_id": int(time_id),
            "var_id": int(var_id),
            "value": float(value),
        }
        for nuts_id, time_id, value, mask in zip(nuts_ids, time_ids, values, masks)
        if mask and (nuts_id is not None) and (time_id is not None)
    ]

    print(f"Start inserting {var_name} values for NUTS in parallel...")
    t_start_insert = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i in range(0, len(var_values), BATCH_SIZE):
            e_batch = i + BATCH_SIZE
            batch = var_values[i:e_batch]
            futures.append(executor.submit(insert_batch, batch, engine, VarValueNuts))

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    print(f"Values of {var_name} inserted into VarValueNuts.")
    return t_start_insert
