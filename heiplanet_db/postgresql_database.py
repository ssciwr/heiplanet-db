import gc
import io
import json
import os
import struct
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from fastapi import HTTPException
from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import (
    DDL,
    BigInteger,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    LargeBinary,
    Numeric,
    SmallInteger,
    String,
    UniqueConstraint,
    create_engine,
    engine,
    event,
    text,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.orm.session import Session, sessionmaker

from heiplanet_db import tiles

CRS = 4326
TIME_POINT_ID = "time_point.id"
VAR_TYPE_ID = "var_type.id"
RESOLUTION_GROUP_ID = "resolution_group.id"
NUTS_DEF_ID = "nuts_def.nuts_id"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10000"))
VAR_TIME_CHUNK = int(os.environ.get("VAR_TIME_CHUNK", "6"))
ROUND_DIGITS = int(os.environ.get("ROUND_DIGITS", "4"))


def _normalize_time_key(t_val: np.datetime64) -> np.datetime64:
    """Normalize a datetime64 value to the key of its time point.

    The database is monthly: every time point is the first day of a month, and
    all values within a month belong to that time point.

    Args:
        t_val: datetime64 value (may include a day and time of day).

    Returns:
        datetime64[ns] of midnight on the first day of the month.
    """
    return np.datetime64(t_val, "M").astype("datetime64[ns]")


def month_starts(times: np.ndarray) -> np.ndarray:
    """Sorted unique first days of the months of `times`, as datetime64[ns]."""
    times = np.asarray(times)
    if times.size == 0:
        return np.array([], dtype="datetime64[ns]")
    return np.unique(times.astype("datetime64[M]")).astype("datetime64[ns]")


# Base declarative class
class Base(DeclarativeBase):
    """
    Base class for all models in the database."""


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
    # how values at coarser grid resolutions are obtained:
    # "subsample": coarse grid points take the value of the native grid point
    #   at the same place (suitable for smooth, intensive quantities such as R0)
    # "mean": coarse grid points carry the area-weighted mean over the fine
    #   cells in their window
    # values of every resolution are stored as tiles, see `VarTile`
    aggregation: Mapped[String] = mapped_column(
        String(), nullable=False, default="subsample", server_default="subsample"
    )

    __table_args__ = (UniqueConstraint("name", name="uq_var_name"),)


class VarValueNuts(Base):
    """
    Variable value table for storing variable values at specific
    NUTS regions and time points.
    """

    __tablename__ = "var_value_nuts"

    id: Mapped[int] = mapped_column(BigInteger(), primary_key=True, autoincrement=True)
    nuts_id: Mapped[String] = mapped_column(String(), ForeignKey(NUTS_DEF_ID))
    time_id: Mapped[int] = mapped_column(Integer(), ForeignKey(TIME_POINT_ID))
    var_id: Mapped[int] = mapped_column(Integer(), ForeignKey(VAR_TYPE_ID))
    value: Mapped[float] = mapped_column(Float())

    __table_args__ = (
        UniqueConstraint("time_id", "nuts_id", "var_id", name="uq_time_nuts_var"),
        ForeignKeyConstraint(
            ["nuts_id"],
            [NUTS_DEF_ID],
            name="fk_nuts_id",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["time_id"],
            [TIME_POINT_ID],
            name="fk_time_id_nuts",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["var_id"],
            [VAR_TYPE_ID],
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


class GridDefinition(Base):
    """The regular grid of one resolution group.

    Grid points are implicit: cell (i, j) lies at
    (lat0 + i * step, lon0 + j * step), both axes ascending. See
    `heiplanet_db.tiles.GridDef`.
    """

    __tablename__ = "grid_def"

    resolution_id: Mapped[int] = mapped_column(
        SmallInteger(),
        ForeignKey(RESOLUTION_GROUP_ID, ondelete="CASCADE"),
        primary_key=True,
    )
    lat0: Mapped[float] = mapped_column(Float(), nullable=False)
    lon0: Mapped[float] = mapped_column(Float(), nullable=False)
    step: Mapped[float] = mapped_column(Float(), nullable=False)
    n_lat: Mapped[int] = mapped_column(Integer(), nullable=False)
    n_lon: Mapped[int] = mapped_column(Integer(), nullable=False)
    tile_size: Mapped[int] = mapped_column(SmallInteger(), nullable=False)

    def to_grid(self) -> tiles.GridDef:
        return tiles.GridDef(
            lat0=self.lat0,
            lon0=self.lon0,
            step=self.step,
            n_lat=self.n_lat,
            n_lon=self.n_lon,
            tile_size=self.tile_size,
        )


class VarTile(Base):
    """Values of a variable on one tile of a grid at one time point.

    `data` holds tile_size × tile_size float32 values (little endian,
    row-major, see `heiplanet_db.tiles`), NaN where there is no data. Tiles
    without any data are not stored. PostgreSQL compresses the column with lz4.
    """

    __tablename__ = "var_tile"

    var_id: Mapped[int] = mapped_column(
        SmallInteger(), ForeignKey(VAR_TYPE_ID, ondelete="CASCADE"), primary_key=True
    )
    resolution_id: Mapped[int] = mapped_column(
        SmallInteger(),
        ForeignKey("grid_def.resolution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    time_id: Mapped[int] = mapped_column(
        Integer(), ForeignKey(TIME_POINT_ID, ondelete="CASCADE"), primary_key=True
    )
    tile_row: Mapped[int] = mapped_column(SmallInteger(), primary_key=True)
    tile_col: Mapped[int] = mapped_column(SmallInteger(), primary_key=True)
    data: Mapped[bytes] = mapped_column(LargeBinary(), nullable=False)


# Decodes value k of a tile (float32, little endian) in SQL, for the debug
# view below. Slow, but only meant for ad-hoc inspection.
_TILE_VALUE_FUNCTION = """
CREATE OR REPLACE FUNCTION tile_value(data bytea, k integer) RETURNS real
LANGUAGE sql IMMUTABLE STRICT AS $$
    SELECT CASE
        WHEN e = 255 AND m = 0 THEN
            (CASE WHEN s = 1 THEN '-Infinity' ELSE 'Infinity' END)::real
        WHEN e = 255 THEN 'NaN'::real
        WHEN e = 0 THEN ((1 - 2 * s) * m * 2.0::float8 ^ (-149))::real
        ELSE ((1 - 2 * s) * (1 + m / 8388608.0::float8)
              * 2.0::float8 ^ (e - 127))::real
    END
    FROM (
        SELECT (b >> 31) & 1 AS s, (b >> 23) & 255 AS e, b & 8388607 AS m
        FROM (
            -- | and << have the same precedence in PostgreSQL
            SELECT get_byte(data, 4 * k)::bigint
                | (get_byte(data, 4 * k + 1)::bigint << 8)
                | (get_byte(data, 4 * k + 2)::bigint << 16)
                | (get_byte(data, 4 * k + 3)::bigint << 24) AS b
        ) AS raw
    ) AS parts
$$;
"""

# One row per stored value. For debugging only.
# (DDL formats the statement with %, hence the %% for the modulo operator.)
_VAR_VALUE_VIEW = """
CREATE OR REPLACE VIEW var_value_v AS
SELECT t.var_id, t.resolution_id, t.time_id,
       round((g.lat0 + i * g.step)::numeric, 4) AS latitude,
       round((g.lon0 + j * g.step)::numeric, 4) AS longitude,
       tile_value(t.data, c.k) AS value
FROM var_tile t
JOIN grid_def g USING (resolution_id)
CROSS JOIN LATERAL generate_series(0, g.tile_size * g.tile_size - 1) AS c(k)
CROSS JOIN LATERAL (
    SELECT t.tile_row * g.tile_size + c.k / g.tile_size AS i,
           t.tile_col * g.tile_size + c.k %% g.tile_size AS j
) AS cell
WHERE i < g.n_lat AND j < g.n_lon AND tile_value(t.data, c.k) <> 'NaN'::real;
"""

event.listen(
    VarTile.__table__,
    "after_create",
    DDL("ALTER TABLE var_tile ALTER COLUMN data SET COMPRESSION lz4"),
)
event.listen(VarTile.__table__, "after_create", DDL(_TILE_VALUE_FUNCTION))
event.listen(VarTile.__table__, "after_create", DDL(_VAR_VALUE_VIEW))
event.listen(VarTile.__table__, "before_drop", DDL("DROP VIEW IF EXISTS var_value_v"))


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


# Gridded values used to be stored as one row per value in these tables. They
# are now stored as tiles (`var_tile`, `grid_def`).
LEGACY_TABLES = ("var_value", "var_coverage", "grid_point_resolution", "grid_point")


def drop_legacy_tables(engine: engine.Engine) -> None:
    """Drop the tables of the old row-per-value storage, if they exist.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
    """
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {', '.join(LEGACY_TABLES)} CASCADE;"))


def create_or_replace_tables(engine: engine.Engine):
    """
    Create or replace tables in the database.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
    """
    # they reference time_point and var_type, so they would block drop_all
    drop_legacy_tables(engine)
    Base.metadata.drop_all(engine)
    print("All tables dropped.")
    create_tables(engine)


def upgrade_schema(engine: engine.Engine):
    """
    Bring tables created by an older version of this package up to date.

    `create_all` only creates missing tables; it does not add columns to
    existing ones or remove old tables. All statements are idempotent.

    The tables of the old row-per-value storage are dropped (see
    `LEGACY_TABLES`); their data has to be loaded again as tiles.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE var_type ADD COLUMN IF NOT EXISTS aggregation VARCHAR "
                "NOT NULL DEFAULT 'subsample';"
            )
        )
    drop_legacy_tables(engine)
    print("Database schema up to date.")


def normalize_db_url(db_url: str) -> str:
    """Use the psycopg2 driver for plain ``postgresql://`` URLs.

    SQLAlchemy >= 2.1 uses psycopg 3 for plain ``postgresql://`` URLs. psycopg 3
    sends query parameters separately and allows at most 65535 of them, which
    large ``IN (...)`` filters (e.g. long lists of IDs)
    exceed. psycopg2 inlines them, so keep using it unless a driver is named.

    Args:
        db_url (str): Database URL.

    Returns:
        str: The URL with an explicit driver.
    """
    for scheme in ("postgresql://", "postgres://"):
        if db_url.startswith(scheme):
            return "postgresql+psycopg2://" + db_url[len(scheme) :]
    return db_url


def initialize_database(db_url: str, replace: bool = False, upgrade: bool = True):
    """
    Initialize the database by creating the engine and tables, and installing PostGIS.
    If replace is True, it will drop and recreate the tables.

    Args:
        db_url (str): Database URL for SQLAlchemy.
        replace (bool): Whether to drop and recreate the tables. Defaults to False.
        upgrade (bool): Whether to bring existing tables up to date with
            `upgrade_schema` (only if replace is False). Defaults to True.
            The API passes False: `upgrade_schema` drops the tables of the
            previous storage and takes exclusive locks, so only the
            production runner upgrades the schema.
    """
    # create engine
    engine = create_engine(
        normalize_db_url(db_url)
    )  # remove echo=True to show just errors in terminal

    # install PostGIS extension
    install_postgis(engine)

    # create or replace tables
    if replace:
        create_or_replace_tables(engine)
    else:
        create_tables(engine)
        if upgrade:
            upgrade_schema(engine)

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


def add_data_list_bulk(session: Session, data_dict_list: list, class_type: type[Base]):
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


def insert_resolution_groups(
    session: Session,
    resolutions: np.ndarray | None = None,
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
    if resolutions is None:
        resolutions = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0])
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
        raise TypeError("Invalid time point format.")


def get_unique_time_points(
    time_point_data: list[tuple[np.ndarray, bool]],
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
            # the database is monthly: one time point per month, on its first day
            time_points.append(month_starts(tpd))
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

    return month_starts(np.concatenate(time_points))


def insert_time_points(
    session: Session, time_point_data: list[tuple[np.ndarray, bool]]
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


def get_id_maps(session: Session) -> tuple[dict, dict]:
    """
    Get ID maps for time points and variable types.

    Args:
        session (Session): SQLAlchemy session object.

    Returns:
        tuple: A tuple containing two dictionaries:\n
            - time_id_map: Mapping of the first day of a month (datetime64,
              see `_normalize_time_key`) to time point ID.\n
            - var_id_map: Mapping of variable name to variable type ID.
    """
    time_id_map = {}
    # earliest day first, so a month with several (legacy) time points maps
    # to the same one that the API selects
    for time_id, year, month in session.query(
        TimePoint.id, TimePoint.year, TimePoint.month
    ).order_by(TimePoint.year, TimePoint.month, TimePoint.day, TimePoint.id):
        key = np.datetime64(f"{year:04d}-{month:02d}", "M").astype("datetime64[ns]")
        time_id_map.setdefault(key, time_id)
    var_id_map = {
        name: var_id for var_id, name in session.query(VarType.id, VarType.name)
    }
    return time_id_map, var_id_map


_PGCOPY_HEADER = b"PGCOPY\n\xff\r\n\x00" + struct.pack("!ii", 0, 0)
_PGCOPY_TRAILER = struct.pack("!h", -1)


CopyColumn = tuple[str, np.ndarray | list[np.ndarray] | None]


def _column_length(values: np.ndarray | list[np.ndarray]) -> int:
    if isinstance(values, list):
        return sum(len(part) for part in values)
    return len(values)


def _pgcopy_rows(columns: list[CopyColumn]) -> np.ndarray:
    """Encode rows in the binary row format of PostgreSQL's COPY.

    The values are written straight into one structured array, so the data is
    held only once more in memory.

    Args:
        columns (list[CopyColumn]): One (dtype, values) pair per column, in
            table column order. `dtype` is a big-endian numpy type matching the
            column exactly (">i4" for integer, ">i2" for smallint, ">f4" for
            real), or a void type ("V<n>") for a bytea of fixed length n.
            `values` is a 1D array with one entry per row, a list of such
            arrays that are concatenated, or None for a column that is NULL in
            every row.

    Returns:
        np.ndarray: Structured array whose bytes are the COPY rows, without
            header and trailer.
    """
    n_rows = next((_column_length(v) for _, v in columns if v is not None), 0)
    fields = [("n_fields", ">i2")]
    for k, (dtype, values) in enumerate(columns):
        fields.append((f"len{k}", ">i4"))
        if values is not None:
            fields.append((f"val{k}", dtype))
    rows = np.empty(n_rows, dtype=np.dtype(fields))
    rows["n_fields"] = len(columns)
    for k, (dtype, values) in enumerate(columns):
        if values is None:
            rows[f"len{k}"] = -1
            continue
        if _column_length(values) != n_rows:
            raise ValueError("All COPY columns must have the same length.")
        rows[f"len{k}"] = np.dtype(dtype).itemsize
        if not isinstance(values, list):
            rows[f"val{k}"] = values
            continue
        start = 0
        for part in values:
            rows[f"val{k}"][start : start + len(part)] = part
            start += len(part)
    return rows


def _pgcopy_binary(columns: list[CopyColumn]) -> bytes:
    """The complete COPY data of `columns`, see `_pgcopy_rows`.

    Returns:
        bytes: The COPY data, including header and trailer.
    """
    return _PGCOPY_HEADER + _pgcopy_rows(columns).tobytes() + _PGCOPY_TRAILER


class _BufferReader(io.RawIOBase):
    """Read-only file over several buffers, without joining them."""

    def __init__(self, *buffers):
        self._parts = [memoryview(b).cast("B") for b in buffers]
        self._index = 0
        self._offset = 0

    def readable(self) -> bool:
        return True

    def readinto(self, target) -> int:
        target = memoryview(target).cast("B")
        written = 0
        while written < len(target) and self._index < len(self._parts):
            part = self._parts[self._index]
            n = min(len(target) - written, len(part) - self._offset)
            target[written : written + n] = part[self._offset : self._offset + n]
            written += n
            self._offset += n
            if self._offset == len(part):
                self._index += 1
                self._offset = 0
        return written


def copy_rows(
    engine: engine.Engine,
    table: str,
    column_names: list[str],
    columns: list[CopyColumn],
) -> int:
    """Insert rows with a binary COPY, in one transaction.

    Much faster than INSERT statements, and no Python object is created per
    row. The encoded rows are streamed to the server, not copied into one
    bytes object. See `_pgcopy_rows` for `columns`.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        table (str): Table name.
        column_names (list[str]): Names of the columns in `columns`.
        columns (list[CopyColumn]): Column data.

    Returns:
        int: Number of rows inserted.
    """
    rows = _pgcopy_rows(columns)
    n_rows = len(rows)
    if n_rows == 0:
        return 0
    connection = engine.raw_connection()
    try:
        with connection.cursor() as cursor:
            cursor.copy_expert(
                f"COPY {table} ({', '.join(column_names)}) "
                "FROM STDIN WITH (FORMAT BINARY)",
                _BufferReader(_PGCOPY_HEADER, rows, _PGCOPY_TRAILER),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return n_rows


def time_id_array(times: np.ndarray, time_id_map: dict) -> np.ndarray:
    """Look up the time point IDs of `times`.

    Args:
        times (np.ndarray): datetime64 values.
        time_id_map (dict): Mapping of datetime64 to time point ID.

    Returns:
        np.ndarray: int64 array of time point IDs.
    """
    ids = []
    for t_val in times:
        time_key = _normalize_time_key(t_val)
        time_id = time_id_map.get(time_key)
        if time_id is None:
            raise ValueError(f"Time point {time_key} is not in the database.")
        ids.append(time_id)
    return np.array(ids, dtype=np.int64)


def _to_time_lat_lon(ds: xr.Dataset, var_name: str) -> xr.Dataset:
    """Return a dataset with only `var_name`, ordered as (time, latitude, longitude).

    Args:
        ds (xr.Dataset): Dataset containing `var_name`.
        var_name (str): Name of the variable to keep.

    Returns:
        xr.Dataset: Dataset with dims ordered (time, latitude, longitude).
    """
    if var_name not in ds:
        raise ValueError(f"Variable {var_name} not found in the dataset.")
    required = ("time", "latitude", "longitude")
    missing = [d for d in required if d not in ds[var_name].dims]
    if missing:
        raise ValueError(
            f"{var_name} must have dims {required}; got {ds[var_name].dims}."
        )
    return ds[[var_name]].transpose(*required, ...)


TRANSFORMS = {"log10"}


def apply_transform(values, transform: str | None, floor: float = 1.0):
    """Transform values for storage, e.g. to log scale for display.

    For "log10", values below `floor` are raised to `floor` first, so the
    result starts at log10(floor). NaN stays NaN. Works on numpy arrays and
    (lazily) on xarray objects.

    Args:
        values: numpy array or xarray object.
        transform (str | None): None (no change) or "log10".
        floor (float): Lower limit applied before the logarithm. Must be > 0.

    Returns:
        The transformed values, same type as `values`.
    """
    if transform is None:
        return values
    if transform not in TRANSFORMS:
        raise ValueError(
            f"Unknown transform {transform}; use one of {sorted(TRANSFORMS)}."
        )
    # also rejects NaN
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("The floor for a log10 transform must be > 0.")
    # np.maximum keeps NaN (np.fmax would replace it by the floor)
    return np.log10(np.maximum(values, floor))


def get_resolution_id_map(session: Session) -> dict[float, int]:
    """Get a mapping of resolution (degrees) to resolution group ID.

    Args:
        session (Session): SQLAlchemy session object.

    Returns:
        dict[float, int]: Mapping of resolution in degrees to resolution group ID.
    """
    return {float(rg.resolution): rg.id for rg in session.query(ResolutionGroup).all()}


def _grid_step(coords: np.ndarray) -> float:
    """Return the (positive) spacing of a regular 1D coordinate array."""
    if len(coords) < 2:
        raise ValueError("Need at least two coordinate values to get the grid step.")
    steps = np.abs(np.diff(coords.astype(np.float64)))
    step = float(np.round(steps[0], ROUND_DIGITS))
    if not np.allclose(steps, step, atol=10 ** (-ROUND_DIGITS)):
        raise ValueError("Coordinates are not regularly spaced.")
    return step


def _coarse_indices(coords: np.ndarray, resolution: float) -> np.ndarray:
    """Indices of the coordinates that are multiples of `resolution`.

    These are the grid points of that resolution group (see `GridDefinition`).
    """
    resolution = float(resolution)
    ratio = coords.astype(np.float64) / resolution
    offset = np.abs(ratio - np.round(ratio)) * resolution
    return np.flatnonzero(offset <= tiles.COORD_ATOL)


def _window_matrix(
    n: int, centres: np.ndarray, factor: int, periodic: bool
) -> np.ndarray:
    """Build the 1D weight matrix that maps fine cells to coarse cells.

    Each coarse cell is centred on a fine grid point and spans `factor` fine
    cells. For odd factors this is `factor` whole cells. For even factors the
    window boundary cuts through the two outermost cells, so the window holds
    `factor + 1` cells and the two outermost ones get weight 0.5. With this
    rule the coarse cells tile the fine grid without gaps or overlap.

    Args:
        n (int): Number of fine cells along this axis.
        centres (np.ndarray): Fine-grid indices of the coarse cell centres.
        factor (int): Coarsening factor (coarse step / fine step).
        periodic (bool): Wrap around at the ends (global longitude axis).
            Otherwise the window is cut off at the ends (latitude axis).

    Returns:
        np.ndarray: Matrix of shape (len(centres), n).
    """
    if factor % 2:
        half = (factor - 1) // 2
        weights = np.ones(factor)
    else:
        half = factor // 2
        weights = np.ones(factor + 1)
        weights[0] = weights[-1] = 0.5
    matrix = np.zeros((len(centres), n))
    rows = np.arange(len(centres))
    for offset, weight in zip(range(-half, half + 1), weights):
        idx = centres + offset
        if periodic:
            np.add.at(matrix, (rows, idx % n), weight)
        else:
            valid = (idx >= 0) & (idx < n)
            np.add.at(matrix, (rows[valid], idx[valid]), weight)
    return matrix


def _block_mean_plan(
    lats: np.ndarray, lons: np.ndarray, resolution: float
) -> dict | None:
    """Prepare the weights to aggregate a fine grid to `resolution`.

    Args:
        lats (np.ndarray): Fine-grid latitudes.
        lons (np.ndarray): Fine-grid longitudes.
        resolution (float): Target resolution in degrees.

    Returns:
        dict | None: Plan with the coarse coordinates, the coarsening factor and
            the weight matrices. None if `resolution` is not an integer multiple
            of the fine grid step.
    """
    step = _grid_step(lats)
    if not np.isclose(_grid_step(lons), step):
        raise ValueError("Latitude and longitude spacing differ.")
    ratio = float(resolution) / step
    factor = round(ratio)
    if factor < 1 or not np.isclose(ratio, factor):
        return None
    if factor == 1:
        # native resolution: values are stored as they are
        return {"factor": 1, "latitude": lats, "longitude": lons}
    lat_idx = _coarse_indices(lats, resolution)
    lon_idx = _coarse_indices(lons, resolution)
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        return None
    plan = {
        "factor": factor,
        "latitude": lats[lat_idx],
        "longitude": lons[lon_idx],
    }
    periodic = np.isclose(len(lons) * step, 360.0)
    # cell areas on a regular lat/lon grid scale with cos(latitude)
    plan["area_weight"] = np.clip(np.cos(np.deg2rad(lats.astype(np.float64))), 0, None)
    plan["lat_matrix"] = _window_matrix(len(lats), lat_idx, factor, periodic=False)
    plan["lon_matrix_t"] = _window_matrix(
        len(lons), lon_idx, factor, periodic=periodic
    ).T
    return plan


def block_mean(values: np.ndarray, plan: dict) -> np.ndarray:
    """Area-weighted mean of a (time, lat, lon) array over coarse cells.

    NaN cells are left out of the mean. A coarse cell with no valid fine cell
    is NaN.

    Args:
        values (np.ndarray): Fine-grid values with shape (time, lat, lon).
        plan (dict): Plan from `_block_mean_plan`.

    Returns:
        np.ndarray: Coarse values with shape (time, coarse lat, coarse lon).
    """
    if plan["factor"] == 1:
        raise ValueError("block_mean is not needed at the native resolution.")
    area = plan["area_weight"][:, None]
    out = np.empty(
        (values.shape[0], len(plan["latitude"]), len(plan["longitude"])),
        dtype=np.float64,
    )
    for t in range(values.shape[0]):
        fine = values[t].astype(np.float64)
        valid = np.isfinite(fine)
        stacked = np.stack([np.where(valid, fine * area, 0.0), valid * area])
        summed = plan["lat_matrix"] @ stacked @ plan["lon_matrix_t"]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t] = np.where(summed[1] > 0, summed[0] / summed[1], np.nan)
    return out


def _check_grid_usable(
    lats: np.ndarray,
    lons: np.ndarray,
    resolution_id_map: dict[float, int],
    var_name: str,
) -> None:
    """Reject grids that the frontend cannot use.

    The frontend requests values at one of the resolution groups, on grid
    points that are multiples of that resolution. Data on another grid could
    never be shown, so it is rejected before anything is stored.

    Raises:
        ValueError: With the reason and how to fix the data.
    """
    resolutions = ", ".join(f"{r:g}" for r in sorted(resolution_id_map))
    try:
        step = _grid_step(lats)
        lon_step = _grid_step(lons)
    except ValueError as e:
        raise ValueError(
            f"The grid of {var_name} cannot be loaded: {e} Gridded variables "
            "need a regular latitude/longitude grid with at least two points "
            "per axis."
        ) from e
    if not np.isclose(lon_step, step, rtol=0, atol=tiles.COORD_ATOL):
        raise ValueError(
            f"The grid of {var_name} cannot be loaded: its latitude step "
            f"({step:g}°) and longitude step ({lon_step:g}°) differ. Regrid the "
            "data to the same step on both axes."
        )
    if not any(
        np.isclose(r, step, rtol=0, atol=tiles.COORD_ATOL) for r in resolution_id_map
    ):
        raise ValueError(
            f"The grid step of {var_name} ({step:g}°) is not one of the "
            f"resolution groups ({resolutions}°). The frontend only requests "
            "these resolutions, so values on this grid could not be shown. "
            "Regrid the data to one of them (e.g. 0.1°)."
        )
    for axis, coords in (("latitude", lats), ("longitude", lons)):
        ratio = coords.astype(np.float64) / step
        offset = np.abs(ratio - np.round(ratio)) * step
        if (offset > tiles.COORD_ATOL).any():
            bad = float(coords[np.argmax(offset)])
            raise ValueError(
                f"The grid of {var_name} cannot be loaded: its points are not "
                f"multiples of its step {step:g}° (e.g. {axis} {bad:g}). The "
                "frontend requests grid points at multiples of the resolution, "
                "so values on this grid could not be shown. Shift or regrid the "
                f"data so that its coordinates are multiples of {step:g}°."
            )


def _resolution_plans(
    lats: np.ndarray,
    lons: np.ndarray,
    resolution_id_map: dict[float, int],
    var_name: str,
) -> list[tuple[float, int, dict]]:
    """Aggregation plans for the resolution groups that fit the data grid.

    Args:
        lats (np.ndarray): Latitudes of the data.
        lons (np.ndarray): Longitudes of the data.
        resolution_id_map (dict[float, int]): Mapping of resolution to group ID.
        var_name (str): Variable name, for messages.

    Returns:
        list[tuple[float, int, dict]]: (resolution, resolution_id, plan), finest
            first. The first one is always the native resolution.

    Raises:
        ValueError: If the grid cannot be served to the frontend: it is not
            regular, its step is not one of the resolution groups, or its
            points are not multiples of the step.
    """
    _check_grid_usable(lats, lons, resolution_id_map, var_name)
    step = _grid_step(lats)
    plans = []
    for resolution, resolution_id in sorted(resolution_id_map.items()):
        plan = _block_mean_plan(lats, lons, resolution)
        if plan is None:
            ratio = float(resolution) / step
            if round(ratio) < 1 or not np.isclose(ratio, round(ratio)):
                reason = f"not a whole multiple of the data grid step {step:g}°"
            else:
                reason = (
                    "the data covers no grid point of this resolution "
                    "(its area is smaller than one cell)"
                )
            print(f"Skipping {resolution} degree resolution for {var_name}: {reason}.")
            continue
        plans.append((resolution, resolution_id, plan))
    if not plans or plans[0][2]["factor"] != 1:
        raise ValueError(
            f"No resolution group matches the grid step of {var_name}; "
            "add it to the resolution groups."
        )
    return plans


def _align_mask(
    mask: xr.DataArray | None, data: xr.DataArray, var_name: str
) -> xr.DataArray | None:
    """Order a mask as (time, latitude, longitude) and check that it covers `data`."""
    if mask is None:
        return None
    mask = mask.transpose("time", "latitude", "longitude")
    if not (
        np.array_equal(mask.latitude.values, data.latitude.values)
        and np.array_equal(mask.longitude.values, data.longitude.values)
    ):
        raise ValueError(f"The mask for {var_name} is on a different grid.")
    if not np.isin(data.time.values, mask.time.values).all():
        raise ValueError(f"The mask for {var_name} does not cover all time points.")
    return mask


def _masked_chunks(data: xr.DataArray, mask: xr.DataArray | None):
    """Yield (first time index, times, float64 values) per chunk of time steps.

    Values where `mask` is False are set to NaN. See `_align_mask` for `mask`.
    """
    n_time = data.sizes["time"]
    for t0 in range(0, n_time, VAR_TIME_CHUNK):
        chunk = data.isel(time=slice(t0, min(t0 + VAR_TIME_CHUNK, n_time)))
        times = chunk.time.values
        values = chunk.values.astype(np.float64)
        if mask is not None:
            mask_values = mask.sel(time=times).values.astype(bool)
            values = np.where(mask_values, values, np.nan)
        yield t0, times, values


def _ascending(
    values: np.ndarray, lats: np.ndarray, lons: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip a (..., lat, lon) array so that both axes are ascending."""
    if len(lats) > 1 and lats[0] > lats[-1]:
        values, lats = values[..., ::-1, :], lats[::-1]
    if len(lons) > 1 and lons[0] > lons[-1]:
        values, lons = values[..., ::-1], lons[::-1]
    return values, lats, lons


def _plan_grid(plan: dict, resolution: float, tile_size: int) -> tiles.GridDef:
    """The grid of a resolution plan, both axes ascending."""
    lats = np.sort(plan["latitude"].astype(np.float64))
    lons = np.sort(plan["longitude"].astype(np.float64))
    return tiles.grid_def_from_coords(
        lats, lons, step=float(resolution), tile_size=tile_size
    )


def ensure_grid_defs(engine: engine.Engine, grids: dict[int, tiles.GridDef]) -> None:
    """Store the grid of each resolution group, or check the stored one.

    All gridded variables must share one grid per resolution group. A stored
    grid that no tile uses yet (e.g. left over from a load that failed) is
    replaced.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        grids (dict[int, tiles.GridDef]): Grid per resolution group ID.
    """
    with engine.begin() as conn:
        for resolution_id, grid in grids.items():
            stored = conn.execute(
                text(
                    "SELECT lat0, lon0, step, n_lat, n_lon, tile_size "
                    "FROM grid_def WHERE resolution_id = :r"
                ),
                {"r": int(resolution_id)},
            ).first()
            if stored is None:
                conn.execute(
                    text(
                        "INSERT INTO grid_def "
                        "(resolution_id, lat0, lon0, step, n_lat, n_lon, tile_size) "
                        "VALUES (:r, :lat0, :lon0, :step, :n_lat, :n_lon, :ts)"
                    ),
                    {
                        "r": int(resolution_id),
                        "lat0": grid.lat0,
                        "lon0": grid.lon0,
                        "step": grid.step,
                        "n_lat": grid.n_lat,
                        "n_lon": grid.n_lon,
                        "ts": grid.tile_size,
                    },
                )
                continue
            stored_grid = tiles.GridDef(*stored)
            same = np.allclose(
                [stored_grid.lat0, stored_grid.lon0, stored_grid.step],
                [grid.lat0, grid.lon0, grid.step],
                rtol=0,
                atol=tiles.COORD_ATOL,
            ) and (stored_grid.n_lat, stored_grid.n_lon, stored_grid.tile_size) == (
                grid.n_lat,
                grid.n_lon,
                grid.tile_size,
            )
            if same:
                continue
            in_use = conn.execute(
                text("SELECT 1 FROM var_tile WHERE resolution_id = :r LIMIT 1"),
                {"r": int(resolution_id)},
            ).first()
            if in_use is None:
                print(
                    f"Replacing the unused grid of resolution group "
                    f"{resolution_id}: {stored_grid} -> {grid}."
                )
                conn.execute(
                    text(
                        "UPDATE grid_def SET lat0 = :lat0, lon0 = :lon0, "
                        "step = :step, n_lat = :n_lat, n_lon = :n_lon, "
                        "tile_size = :ts WHERE resolution_id = :r"
                    ),
                    {
                        "r": int(resolution_id),
                        "lat0": grid.lat0,
                        "lon0": grid.lon0,
                        "step": grid.step,
                        "n_lat": grid.n_lat,
                        "n_lon": grid.n_lon,
                        "ts": grid.tile_size,
                    },
                )
            else:
                raise ValueError(
                    f"The grid of resolution group {resolution_id} differs from "
                    f"the stored one: {grid} vs. {stored_grid}. All gridded "
                    "variables must share one grid."
                )


def _check_not_loaded(
    engine: engine.Engine, var_id: int, time_ids: np.ndarray, var_name: str
) -> None:
    """Raise if `var_id` already has tiles at any of `time_ids`."""
    with engine.connect() as conn:
        loaded = conn.execute(
            text(
                "SELECT DISTINCT tp.year, tp.month FROM var_tile v "
                "JOIN time_point tp ON tp.id = v.time_id "
                "WHERE v.var_id = :v AND v.time_id = ANY(:t) "
                "ORDER BY tp.year, tp.month"
            ),
            {"v": int(var_id), "t": [int(t) for t in time_ids]},
        ).all()
    if loaded:
        months = ", ".join(f"{y}-{m:02d}" for y, m in loaded[:5])
        more = f" and {len(loaded) - 5} more" if len(loaded) > 5 else ""
        raise ValueError(
            f"{var_name} is already loaded for {months}{more}. Remove its tiles "
            "first (or reset the production tables) to load it again, and make "
            "sure no two files or config entries cover the same months."
        )


def _delete_var_tiles(engine: engine.Engine, var_id: int, time_ids: np.ndarray):
    """Delete the tiles of `var_id` at `time_ids`, at all resolutions."""
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM var_tile WHERE var_id = :v AND time_id = ANY(:t)"),
            {"v": int(var_id), "t": [int(t) for t in time_ids]},
        )


def copy_var_tiles(
    engine: engine.Engine,
    var_id: int,
    resolution_id: int,
    time_ids: np.ndarray,
    values: np.ndarray,
    tile_size: int,
) -> int:
    """Cut (time, lat, lon) values into tiles and store them with a binary COPY.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        var_id (int): Variable type ID.
        resolution_id (int): Resolution group ID.
        time_ids (np.ndarray): Time point ID per entry of the first axis.
        values (np.ndarray): Values on the full grid of the resolution group,
            both axes ascending, NaN where there is no data.
        tile_size (int): Cells per tile side.

    Returns:
        int: Number of tiles stored.
    """
    # all tiles have the same size, so they fit a fixed-size numpy type
    blob_dtype = np.dtype(f"V{tile_size * tile_size * tiles.TILE_DTYPE.itemsize}")
    t_ids, rows, cols, blocks = [], [], [], []
    for time_id, values_t in zip(time_ids, values):
        tile_rows, tile_cols, blocks_t = tiles.array_to_tile_blocks(values_t, tile_size)
        if len(tile_rows) == 0:
            continue
        t_ids.append(np.full(len(tile_rows), time_id))
        rows.append(tile_rows)
        cols.append(tile_cols)
        # a view: each tile becomes one fixed-size bytea value
        blocks.append(blocks_t.reshape(len(tile_rows), -1).view(blob_dtype)[:, 0])
    n_tiles = sum(len(r) for r in rows)
    if n_tiles == 0:
        return 0
    return copy_rows(
        engine,
        "var_tile",
        ["var_id", "resolution_id", "time_id", "tile_row", "tile_col", "data"],
        [
            (">i2", np.full(n_tiles, var_id)),
            (">i2", np.full(n_tiles, resolution_id)),
            (">i4", t_ids),
            (">i2", rows),
            (">i2", cols),
            (blob_dtype.str, blocks),
        ],
    )


def insert_var_tiles(
    engine: engine.Engine,
    ds: xr.Dataset,
    var_name: str,
    time_id_map: dict,
    var_id_map: dict,
    resolution_id_map: dict[float, int],
    aggregation: str = "subsample",
    mask: xr.DataArray | None = None,
    transform: str | None = None,
    transform_floor: float = 1.0,
    tile_size: int = tiles.DEFAULT_TILE_SIZE,
) -> int:
    """Insert a gridded variable as tiles, at every fitting resolution group.

    For every resolution group that is an integer multiple of the data's grid
    step, the values are computed on that group's grid and stored as tiles
    (see `VarTile`): by taking every n-th grid point (aggregation
    "subsample") or the area-weighted mean over each coarse cell (aggregation
    "mean"). The grid of each group is stored in `grid_def` (or checked
    against the stored one).

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        ds (xr.Dataset): Dataset with `var_name` over (time, latitude, longitude),
            in any dimension order.
        var_name (str): Name of the variable to insert.
        time_id_map (dict): Mapping of datetime64 to time point ID.
        var_id_map (dict): Mapping of variable name to variable type ID.
        resolution_id_map (dict[float, int]): Mapping of resolution to group ID.
        aggregation (str): "subsample" or "mean". Defaults to "subsample".
        mask (xr.DataArray | None): Boolean (time, latitude, longitude) array on
            the same grid. Cells where it is False are set to NaN first, e.g.
            to drop ocean cells. Defaults to None.
        transform (str | None): Transform applied after aggregating, see
            `apply_transform`. Means are always taken over the original values.
        transform_floor (float): Floor for the transform. Defaults to 1.
        tile_size (int): Cells per tile side.

    Returns:
        int: Total number of tiles inserted over all resolutions.
    """
    if aggregation not in ("subsample", "mean"):
        raise ValueError(f"Unknown aggregation {aggregation} for {var_name}.")
    # fail early, before any data is read
    apply_transform(np.array([1.0]), transform, transform_floor)
    var_id = var_id_map.get(var_name)
    if var_id is None:
        raise ValueError(f"Variable {var_name} not found in var_type table.")
    data = _to_time_lat_lon(ds, var_name)[var_name]
    lats = data.latitude.values
    lons = data.longitude.values

    plans = _resolution_plans(lats, lons, resolution_id_map, var_name)
    mask = _align_mask(mask, data, var_name)
    # check every time point before anything is stored
    all_time_ids = time_id_array(data.time.values, time_id_map)
    if len(np.unique(all_time_ids)) != len(all_time_ids):
        raise ValueError(
            f"{var_name} has several time steps in one month. The database "
            "stores one value per month; aggregate the data to monthly first."
        )
    _check_not_loaded(engine, var_id, all_time_ids, var_name)
    grids = {
        resolution_id: _plan_grid(plan, resolution, tile_size)
        for resolution, resolution_id, plan in plans
    }
    ensure_grid_defs(engine, grids)
    strides = {
        resolution_id: (
            _coarse_indices(lats, resolution),
            _coarse_indices(lons, resolution),
        )
        for resolution, resolution_id, plan in plans
        if plan["factor"] > 1
    }

    total = 0
    try:
        for t0, times, values in _masked_chunks(data, mask):
            time_ids = all_time_ids[t0 : t0 + len(times)]
            for resolution, resolution_id, plan in plans:
                if plan["factor"] == 1:
                    coarse = values
                elif aggregation == "mean":
                    coarse = block_mean(values, plan)
                else:
                    lat_idx, lon_idx = strides[resolution_id]
                    coarse = values[:, lat_idx][:, :, lon_idx]
                coarse = apply_transform(coarse, transform, transform_floor)
                coarse, _, _ = _ascending(coarse, plan["latitude"], plan["longitude"])
                total += copy_var_tiles(
                    engine, var_id, resolution_id, time_ids, coarse, tile_size
                )
                print(
                    f"{var_name}: inserted {resolution} degree tiles for time "
                    f"indices {t0}:{t0 + len(times)}."
                )
            del values
            gc.collect()
    except BaseException:
        # the chunks are committed one by one: remove what this call stored, so
        # that the variable is not left half-loaded and can be loaded again
        _delete_var_tiles(engine, var_id, all_time_ids)
        print(f"Loading {var_name} failed; its tiles stored so far were removed.")
        raise

    print(f"Tiles of {var_name} inserted for all resolutions. Total: {total}")
    return total


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
    start_time_point: tuple[int, int],
    end_time_point: tuple[int, int] | None = None,
) -> list[TimePoint]:
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


def get_var_types(
    session: Session,
    var_names: None | list[str] = None,
) -> list[VarType]:
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


def get_grid_def(session: Session, resolution_id: int) -> tiles.GridDef | None:
    """The stored grid of a resolution group, None if there is none."""
    row = session.get(GridDefinition, resolution_id)
    return None if row is None else row.to_grid()


def _finest_resolution_id(session: Session, var_id: int) -> int | None:
    """Finest resolution group that has tiles of a variable."""
    row = session.execute(
        text(
            """
            SELECT g.resolution_id
            FROM grid_def g
            JOIN resolution_group r ON r.id = g.resolution_id
            WHERE EXISTS (
                SELECT 1 FROM var_tile t
                WHERE t.var_id = :v AND t.resolution_id = g.resolution_id
            )
            ORDER BY r.resolution
            LIMIT 1
            """
        ),
        {"v": int(var_id)},
    ).first()
    return None if row is None else row[0]


def read_tile_values(
    session: Session,
    var_id: int,
    resolution_id: int,
    time_ids: list[int],
    grid: tiles.GridDef,
    index_range: tuple[int, int, int, int],
) -> np.ndarray:
    """Values of a variable on a block of cells, from the tiles, in one query.

    Args:
        session (Session): SQLAlchemy session object.
        var_id (int): Variable type ID.
        resolution_id (int): Resolution group ID.
        time_ids (list[int]): Time point IDs.
        grid (tiles.GridDef): Grid of the resolution group.
        index_range (tuple[int, int, int, int]): Cells
            (i_start, i_stop, j_start, j_stop), see `tiles.area_to_index_range`.

    Returns:
        np.ndarray: float32 array (time, lat, lon), both axes ascending, NaN
            where there is no data.
    """
    i0, i1, j0, j1 = index_range
    r0, r1, c0, c1 = tiles.tile_range(grid, i0, i1, j0, j1)
    rows = session.execute(
        text(
            """
            SELECT time_id, tile_row, tile_col, data FROM var_tile
            WHERE var_id = :v AND resolution_id = :r AND time_id = ANY(:t)
                AND tile_row >= :r0 AND tile_row < :r1
                AND tile_col >= :c0 AND tile_col < :c1
            """
        ),
        {
            "v": int(var_id),
            "r": int(resolution_id),
            "t": [int(t) for t in time_ids],
            "r0": r0,
            "r1": r1,
            "c0": c0,
            "c1": c1,
        },
    ).all()
    by_time: dict[int, list] = {int(t): [] for t in time_ids}
    for time_id, tile_row, tile_col, data in rows:
        by_time[time_id].append((tile_row, tile_col, data))
    return np.stack(
        [
            tiles.tiles_to_array(by_time[int(t)], grid.tile_size, i0, i1, j0, j1)
            for t in time_ids
        ]
    )


def get_var_value(
    session: Session,
    var_name: str,
    lat: float,
    lon: float,
    year: int,
    month: int,
    day: int,
    resolution: float | None = None,
) -> float | None:
    """Get the value of a variable at one grid point and time point.

    Args:
        session (Session): SQLAlchemy session object.
        var_name (str): Name of the variable to retrieve.
        lat (float): Latitude of the grid point.
        lon (float): Longitude of the grid point.
        year (int): Year of the time point.
        month (int): Month of the time point.
        day (int): Day of the time point. Only monthly data is stored, so the
            first day of the month is always used.
        resolution (float | None): Grid resolution in degrees, only used for
            variables with aggregation "mean". If None, the finest resolution
            with data is used. "subsample" values are the same at every
            resolution, so the finest one is always used for them.

    Returns:
        float | None: The value, or None if the variable, time point or grid
            point does not exist or there is no data at that point.
    """
    if day != 1:
        print(
            "The current database only supports monthly data."
            "Retieving data for the first day of the month..."
        )
        day = 1
    var_type = session.query(VarType).filter(VarType.name == var_name).first()
    if var_type is None:
        return None
    if resolution is None or var_type.aggregation == "subsample":
        resolution_id = _finest_resolution_id(session, var_type.id)
    else:
        resolution_id = get_resolution_id(session, resolution)
    if resolution_id is None:
        return None
    grid = get_grid_def(session, resolution_id)
    time_point = (
        session.query(TimePoint.id)
        .filter(TimePoint.year == year, TimePoint.month == month, TimePoint.day == day)
        .first()
    )
    if grid is None or time_point is None:
        return None
    cell = tiles.cell_index(grid, lat, lon)
    if cell is None:
        return None
    i, j = cell
    value = read_tile_values(
        session, var_type.id, resolution_id, [time_point.id], grid, (i, i + 1, j, j + 1)
    )[0, 0, 0]
    return None if np.isnan(value) else float(tiles.shortest_float64(value))


def _area_to_index_range(
    grid: tiles.GridDef, area: tuple[float, float, float, float] | None
) -> tuple[int, int, int, int] | None:
    """`tiles.area_to_index_range`, rejecting invalid areas with HTTP 400."""
    try:
        return tiles.area_to_index_range(grid, area)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def get_var_values_cartesian(
    session: Session,
    time_point: tuple[int, int],
    grid_resolution: float = 0.1,
    area: None | tuple[float, float, float, float] = None,
    var_name: None | str = None,
) -> dict:
    """Get variable values for a cartesian map.

    Args:
        session (Session): SQLAlchemy session object.
        time_point (Tuple[int, int]): Date point as (year, month).
        grid_resolution (float): Resolution of the grid points. Defaults to 0.1 degree.
        area (None | Tuple[float, float, float, float]):
            Area as (North, West, South, East), boundaries included.
            If None, all grid points are used.
        var_name (None | str): Variable name for which values should be returned.
         If None, the default model values will be returned.

    Returns:
        dict: a dict with (latitude, longitude, var_value) for the requested
            date, ordered by latitude and then longitude. Grid points without
            data (NaN, e.g. ocean) are left out.
    """
    date_object = (
        session.query(TimePoint)
        .filter((TimePoint.year == time_point[0]) & (TimePoint.month == time_point[1]))
        .order_by(TimePoint.day, TimePoint.id)
        .first()
    )
    if not date_object:
        print("No time point found for the specified date.")
        raise HTTPException(status_code=400, detail="Missing data for requested date.")

    resolution_id = get_resolution_id(session=session, resolution=grid_resolution)
    if not resolution_id:
        print("No resolution id found for requested resolution.")
        raise HTTPException(
            status_code=400, detail="No id found for specified resolution."
        )

    grid = get_grid_def(session, resolution_id)
    index_range = None if grid is None else _area_to_index_range(grid, area)
    if index_range is None:
        print(
            "No grid points found in the specified area or with the specified resolution."
        )
        raise HTTPException(
            status_code=400,
            detail="No grid points found in specified area or with the specified resolution.",
        )

    if not var_name:
        var_name = "t2m"  # default variable name
    var_type = session.query(VarType).filter(VarType.name == var_name).first()
    if not var_type:
        print("No variable type found for the specified name.")
        raise HTTPException(
            status_code=400, detail="Missing variable type for requested time point."
        )

    values = read_tile_values(
        session, var_type.id, resolution_id, [date_object.id], grid, index_range
    )[0]
    i0, i1, j0, j1 = index_range
    lat_idx, lon_idx = np.nonzero(~np.isnan(values))
    # ordered by latitude, then longitude
    lats = np.round(grid.latitudes(i0, i1), ROUND_DIGITS)[lat_idx]
    lons = np.round(grid.longitudes(j0, j1), ROUND_DIGITS)[lon_idx]
    values_list = list(
        zip(
            lats.tolist(),
            lons.tolist(),
            tiles.shortest_float64(values[lat_idx, lon_idx]).tolist(),
        )
    )
    return {"latitude, longitude, var_value": values_list}


def get_var_values_cartesian_for_download(
    session: Session,
    start_time_point: tuple[int, int],
    end_time_point: tuple[int, int] | None = None,
    area: None | tuple[float, float, float, float] = None,
    var_names: None | list[str] = None,
    netcdf_file: str = "cartesian_grid_data_heiplanet.nc",
) -> dict:
    """Write variable values for a cartesian map to a NetCDF file.

    Every variable is exported at the finest resolution at which it has data,
    on the grid of the finest stored resolution.

    Args:
        session (Session): SQLAlchemy session object.
        start_time_point (Tuple[int, int]): Start time point as (year, month).
        end_time_point (Tuple[int, int] | None): End time point as (year, month).
            If None, only the start time point is used.
        area (None | Tuple[float, float, float, float]):
            Area as (North, West, South, East), boundaries included.
            If None, all grid points are used.
        var_names (None | List[str]): List of variable names to filter by.
            If None, all variable types are used.
        netcdf_file (str): Name of the NetCDF file to save the dataset.

    Returns:
        dict: a dict with the keys "response" and "netcdf_file".
    """
    time_points = sorted(
        get_time_points(session, start_time_point, end_time_point),
        key=lambda tp: (tp.year, tp.month, tp.day),
    )
    if not time_points:
        print("No time points found in the specified range.")
        raise HTTPException(
            status_code=400, detail="Missing data for requested time point."
        )
    time_values = np.asarray(
        [
            np.datetime64(pd.Timestamp(year=tp.year, month=tp.month, day=1), "ns")
            for tp in time_points
        ],
        dtype="datetime64[ns]",
    )
    time_ids = [tp.id for tp in time_points]

    # the finest stored grid gives the coordinates of the dataset
    finest = (
        session.query(GridDefinition)
        .join(ResolutionGroup, ResolutionGroup.id == GridDefinition.resolution_id)
        .order_by(ResolutionGroup.resolution.asc())
        .first()
    )
    out_grid = None if finest is None else finest.to_grid()
    out_range = None if out_grid is None else _area_to_index_range(out_grid, area)
    if out_range is None:
        print("No grid points found in the specified area.")
        raise HTTPException(
            status_code=400, detail="No grid points found in specified area."
        )
    oi0, oi1, oj0, oj1 = out_range
    latitudes = np.round(out_grid.latitudes(oi0, oi1), ROUND_DIGITS)
    longitudes = np.round(out_grid.longitudes(oj0, oj1), ROUND_DIGITS)

    var_types = get_var_types(session, var_names)
    if not var_types:
        print("No variable types found in the specified names.")
        raise HTTPException(
            status_code=400, detail="Missing variable type for requested time point."
        )

    ds = xr.Dataset(
        coords={
            "time": ("time", time_values),
            "latitude": ("latitude", latitudes),
            "longitude": ("longitude", longitudes),
        }
    )
    for vt in var_types:
        values_array = np.full(
            (len(time_values), len(latitudes), len(longitudes)), np.nan
        )
        resolution_id = _finest_resolution_id(session, vt.id)
        grid = None if resolution_id is None else get_grid_def(session, resolution_id)
        index_range = None if grid is None else _area_to_index_range(grid, area)
        if index_range is not None:
            values = read_tile_values(
                session, vt.id, resolution_id, time_ids, grid, index_range
            )
            i0, i1, j0, j1 = index_range
            # positions of this variable's cells on the output grid
            rows = np.round(
                (grid.latitudes(i0, i1) - latitudes[0]) / out_grid.step
            ).astype(int)
            cols = np.round(
                (grid.longitudes(j0, j1) - longitudes[0]) / out_grid.step
            ).astype(int)
            keep_r = (rows >= 0) & (rows < len(latitudes))
            keep_c = (cols >= 0) & (cols < len(longitudes))
            values_array[:, rows[keep_r][:, None], cols[keep_c][None, :]] = (
                tiles.shortest_float64(values[:, keep_r][:, :, keep_c])
            )
        ds[vt.name] = (("time", "latitude", "longitude"), values_array)
        # NetCDF attributes cannot be None
        for attr in ("unit", "description"):
            if getattr(vt, attr) is not None:
                ds[vt.name].attrs[attr] = getattr(vt, attr)

    ds.attrs["source"] = "OneHealth Database"
    ds.attrs["created_at"] = pd.Timestamp.now().isoformat()
    ds.attrs["description"] = "Variable values for a cartesian map from the database."
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


def filter_nuts_ids_for_resolution(nuts_ids: list[str], resolution: str) -> list[str]:
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
    time_point: tuple[int, int],
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
        .order_by(TimePoint.day, TimePoint.id)
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
    nuts_ids = set(filter_nuts_ids_for_resolution(nuts_ids, grid_resolution))
    if not nuts_ids:
        print(f"No {grid_resolution} NUTS id's found in the database.")
        raise HTTPException(
            status_code=400,
            detail=f"No {grid_resolution} NUTS ids found in the database.",
        )
    # create a dict with NUTS_id: var_value; select the values by their own
    # nuts_id (pairing the filtered ids with the unfiltered values by position
    # would shift values onto the wrong regions)
    mydict = {v.nuts_id: v.value for v in values if v.nuts_id in nuts_ids}
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

    # Stream inserts in small time-chunks to avoid loading/stacking the full
    # (time, NUTS_ID) array and building a huge in-memory list of rows.
    var_data = ds[var_name].dropna(dim="NUTS_ID", how="all")
    required_dims = {"time", "NUTS_ID"}
    missing_dims = required_dims - set(var_data.dims)
    if missing_dims:
        raise ValueError(
            f"{var_name} must include dims ('time', 'NUTS_ID'); got {var_data.dims}."
        )

    # Reject unexpected non-singleton dimensions (e.g., lat/lon) early to avoid
    # shape mismatches that would later lead to IndexError / incorrect inserts.
    extra_dims = [d for d in var_data.dims if d not in required_dims]
    if extra_dims:
        non_singleton = [d for d in extra_dims if int(var_data.sizes.get(d, 0)) != 1]
        if non_singleton:
            raise ValueError(
                f"{var_name} must be 2D over ('time', 'NUTS_ID'); got extra dims "
                f"{tuple(non_singleton)} with sizes "
                f"{ {d: int(var_data.sizes[d]) for d in non_singleton} }."
            )
        # harmless singleton dims can be squeezed away (keeps behavior flexible)
        var_data = var_data.squeeze(dim=extra_dims, drop=True)

    # Some inputs come as (NUTS_ID, time). Normalize to (time, NUTS_ID) so the
    # chunking/insertion logic always treats axis 0 as time.
    var_data = var_data.transpose("time", "NUTS_ID")
    months = var_data["time"].values.astype("datetime64[M]")
    if len(np.unique(months)) != len(months):
        raise ValueError(
            f"{var_name} has several time steps in one month. The database "
            "stores one value per month; aggregate the data to monthly first."
        )

    nuts_ids = var_data["NUTS_ID"].values

    print(
        f"Start inserting {var_name} values for NUTS in streaming time-chunks "
        f"(time_chunk={VAR_TIME_CHUNK}, batch_size={BATCH_SIZE})..."
    )
    t_start_insert = time.time()

    session = create_session(engine)
    total_inserted = 0
    try:
        n_time = int(var_data.sizes["time"])
        for t0 in range(0, n_time, VAR_TIME_CHUNK):
            t1 = min(t0 + VAR_TIME_CHUNK, n_time)
            chunk = var_data.isel(time=slice(t0, t1)).load()

            time_vals = chunk["time"].values.astype("datetime64[ns]")
            time_ids = [time_id_map.get(_normalize_time_key(t)) for t in time_vals]

            values = chunk.values
            if values.ndim != 2:
                raise ValueError(
                    f"{var_name} must be 2D after normalization; got array with "
                    f"ndim={values.ndim} and dims={chunk.dims}."
                )
            if values.shape[1] != len(nuts_ids):
                raise ValueError(
                    f"{var_name} dimension mismatch: NUTS axis length is {values.shape[1]} "
                    f"but got {len(nuts_ids)} NUTS_IDs."
                )
            batch: list[dict] = []

            for ti, time_id in enumerate(time_ids):
                if time_id is None:
                    continue
                row = values[ti]
                valid_idx = np.flatnonzero(~np.isnan(row))
                for nj in valid_idx:
                    nuts_id = nuts_ids[nj]
                    if nuts_id is None:
                        continue
                    batch.append(
                        {
                            "nuts_id": str(nuts_id),
                            "time_id": int(time_id),
                            "var_id": int(var_id),
                            "value": float(row[nj]),
                        }
                    )
                    if len(batch) >= BATCH_SIZE:
                        add_data_list_bulk(session, batch, VarValueNuts)
                        total_inserted += len(batch)
                        batch.clear()

            if batch:
                add_data_list_bulk(session, batch, VarValueNuts)
                total_inserted += len(batch)
                batch.clear()

            del chunk, values
            gc.collect()
            print(
                f"Inserted {total_inserted} {var_name} rows into VarValueNuts "
                f"(processed time indices {t0}:{t1})."
            )
    finally:
        session.close()

    print(f"Values of {var_name} inserted into VarValueNuts. Total: {total_inserted}.")
    return t_start_insert
