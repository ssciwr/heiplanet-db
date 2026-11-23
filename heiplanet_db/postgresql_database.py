import gc
import io
import json
import math
import os
import struct
import time
from pathlib import Path
from typing import ClassVar

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from fastapi import HTTPException
from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import (
    REAL,
    BigInteger,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    Numeric,
    SmallInteger,
    String,
    UniqueConstraint,
    and_,
    create_engine,
    engine,
    func,
    or_,
    text,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.orm.session import Session, sessionmaker
from sqlalchemy.schema import AddConstraint

CRS = 4326
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10000"))
VAR_TIME_CHUNK = int(os.environ.get("VAR_TIME_CHUNK", "6"))
GRID_LAT_CHUNK = int(os.environ.get("GRID_LAT_CHUNK", "45"))
GRID_LON_CHUNK = int(os.environ.get("GRID_LON_CHUNK", "360"))
ROUND_DIGITS = int(os.environ.get("ROUND_DIGITS", "4"))


def _q(value):
    """Round a float coordinate to ROUND_DIGITS precision.

    Ensures consistent rounding behavior for both Python floats and NumPy scalars.
    Returns a Python float for reliable dictionary key lookups.
    """
    # Convert NumPy scalars to Python float first
    if isinstance(value, (np.floating, np.integer)):
        value = float(value)
    return round(value, ROUND_DIGITS)


def _normalize_time_key(t_val: np.datetime64) -> np.datetime64:
    """Normalize a datetime64 value to a date-only key for time_id_map lookup.

    Args:
        t_val: datetime64 value (may include time components).

    Returns:
        datetime64[ns] normalized to midnight of the date.
    """
    ts = pd.Timestamp(t_val)
    return np.datetime64(pd.to_datetime(f"{ts.year}-{ts.month}-{ts.day}"), "ns")


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


class GridPoint(Base):
    """
    Grid point table for storing latitude and longitude coordinates."""

    __tablename__ = "grid_point"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    latitude: Mapped[float] = mapped_column(Numeric(precision=8, scale=4))
    longitude: Mapped[float] = mapped_column(Numeric(precision=8, scale=4))

    # There is no PostGIS point column: no query used it, and with its spatial
    # index it took about 0.9 GB for the global 0.1 degree grid.
    __table_args__ = (UniqueConstraint("latitude", "longitude", name="uq_lat_lon"),)


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
    # "subsample": coarse grid points reuse the value stored at the native
    #   resolution (suitable for smooth, intensive quantities such as R0)
    # "mean": coarse grid points carry the area-weighted mean over the fine
    #   cells in their window, stored separately per resolution
    aggregation: Mapped[String] = mapped_column(
        String(), nullable=False, default="subsample", server_default="subsample"
    )

    __table_args__ = (UniqueConstraint("name", name="uq_var_name"),)


class VarValue(Base):
    """
    Variable value table for storing variable values at specific
    grid points and time points.
    """

    __tablename__ = "var_value"

    # This table holds hundreds of millions of rows, so it has no surrogate
    # key: rows are identified by the unique constraint below. The columns are
    # as narrow as the data allows and ordered so that PostgreSQL needs no
    # alignment padding (int4, int4, int2, int2, float4: 16 bytes of data, 44
    # bytes per row with the tuple header). float4 keeps about 7 significant
    # digits, more than the maps and the downloads need.
    # The foreign keys are declared once, in __table_args__.
    grid_id: Mapped[int] = mapped_column(Integer())
    time_id: Mapped[int] = mapped_column(Integer())
    var_id: Mapped[int] = mapped_column(SmallInteger())
    # NULL for variables with aggregation "subsample" (one value per grid point,
    # shared by all resolutions); set for variables whose value depends on the
    # resolution (aggregation "mean"), one row per resolution group
    resolution_id: Mapped[int | None] = mapped_column(SmallInteger(), nullable=True)
    value: Mapped[float] = mapped_column(REAL())

    __mapper_args__: ClassVar[dict] = {
        # identity for the ORM only; the database enforces it with
        # uq_time_grid_var_res (resolution_id may be NULL)
        "primary_key": [time_id, grid_id, var_id, resolution_id],
    }

    def __init__(self, **kw):
        # part of the ORM identity, so it must be set even when it is NULL
        kw.setdefault("resolution_id", None)
        super().__init__(**kw)

    __table_args__ = (
        UniqueConstraint(
            "time_id",
            "grid_id",
            "var_id",
            "resolution_id",
            name="uq_time_grid_var_res",
            postgresql_nulls_not_distinct=True,
        ),
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
        ForeignKeyConstraint(
            ["resolution_id"],
            ["resolution_group.id"],
            name="fk_resolution_id",
            ondelete="CASCADE",
        ),
    )


class VarCoverage(Base):
    """Grid points where a variable has a value at every time point.

    Values equal to 0 at these grid points are not stored in `var_value`; a
    missing row there means 0. Outside of them, a missing row means that there
    is no data (e.g. ocean cells), and zeros are stored. This keeps the many
    zeros of the model output (e.g. no mosquitoes in winter) out of the
    database. Like `var_value`, `resolution_id` is NULL for variables with
    aggregation "subsample".
    """

    __tablename__ = "var_coverage"

    var_id: Mapped[int] = mapped_column(SmallInteger())
    resolution_id: Mapped[int | None] = mapped_column(SmallInteger(), nullable=True)
    grid_id: Mapped[int] = mapped_column(Integer())

    __mapper_args__: ClassVar[dict] = {"primary_key": [var_id, resolution_id, grid_id]}

    def __init__(self, **kw):
        # part of the ORM identity, so it must be set even when it is NULL
        kw.setdefault("resolution_id", None)
        super().__init__(**kw)

    __table_args__ = (
        UniqueConstraint(
            "var_id",
            "resolution_id",
            "grid_id",
            name="uq_coverage_var_res_grid",
            postgresql_nulls_not_distinct=True,
        ),
        ForeignKeyConstraint(
            ["var_id"], ["var_type.id"], name="fk_coverage_var_id", ondelete="CASCADE"
        ),
        ForeignKeyConstraint(
            ["resolution_id"],
            ["resolution_group.id"],
            name="fk_coverage_resolution_id",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["grid_id"],
            ["grid_point.id"],
            name="fk_coverage_grid_id",
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


def upgrade_schema(engine: engine.Engine):
    """
    Bring tables created by an older version of this package up to date.

    `create_all` only creates missing tables; it does not add columns to
    existing ones. This adds the columns and constraints needed for
    resolution-dependent variables. All statements are idempotent.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
    """
    statements = [
        (
            "ALTER TABLE var_type ADD COLUMN IF NOT EXISTS aggregation VARCHAR "
            "NOT NULL DEFAULT 'subsample';"
        ),
        "ALTER TABLE var_value ADD COLUMN IF NOT EXISTS resolution_id INTEGER;",
        # rows are identified by uq_time_grid_var_res; the surrogate key and its
        # index took about 15 % of the database
        "ALTER TABLE var_value DROP COLUMN IF EXISTS id;",
        # duplicates of fk_grid_id, fk_time_id and fk_var_id
        "ALTER TABLE var_value DROP CONSTRAINT IF EXISTS var_value_grid_id_fkey;",
        "ALTER TABLE var_value DROP CONSTRAINT IF EXISTS var_value_time_id_fkey;",
        "ALTER TABLE var_value DROP CONSTRAINT IF EXISTS var_value_var_id_fkey;",
        "ALTER TABLE var_value DROP CONSTRAINT IF EXISTS uq_time_grid_var;",
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'uq_time_grid_var_res'
            ) THEN
                ALTER TABLE var_value ADD CONSTRAINT uq_time_grid_var_res
                    UNIQUE NULLS NOT DISTINCT
                    (time_id, grid_id, var_id, resolution_id);
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'fk_resolution_id'
            ) THEN
                ALTER TABLE var_value ADD CONSTRAINT fk_resolution_id
                    FOREIGN KEY (resolution_id) REFERENCES resolution_group (id)
                    ON DELETE CASCADE;
            END IF;
        END $$;
        """,
        # unused PostGIS point; dropping it also drops its two GiST indexes
        "ALTER TABLE grid_point DROP COLUMN IF EXISTS point;",
        # narrower types (rewrites the tables once, so only when needed)
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'var_value' AND (
                    (column_name = 'value' AND data_type <> 'real')
                    OR (column_name IN ('var_id', 'resolution_id')
                        AND data_type <> 'smallint')
                )
            ) THEN
                ALTER TABLE var_value
                    ALTER COLUMN var_id TYPE smallint,
                    ALTER COLUMN resolution_id TYPE smallint,
                    ALTER COLUMN value TYPE real;
            END IF;
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'var_coverage'
                    AND column_name IN ('var_id', 'resolution_id')
                    AND data_type <> 'smallint'
            ) THEN
                ALTER TABLE var_coverage
                    ALTER COLUMN var_id TYPE smallint,
                    ALTER COLUMN resolution_id TYPE smallint;
            END IF;
        END $$;
        """,
    ]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
    print("Database schema up to date.")


# tables that receive hundreds of millions of rows in a bulk load
BULK_LOAD_TABLES = ("var_value", "var_coverage")


def _bulk_load_constraints(table_name: str) -> list:
    """Unique and foreign key constraints of a table, as declared in the model."""
    table = Base.metadata.tables[table_name]
    return [
        c
        for c in table.constraints
        if isinstance(c, (UniqueConstraint, ForeignKeyConstraint))
    ]


def drop_bulk_load_constraints(engine: engine.Engine) -> None:
    """Drop the unique and foreign key constraints of `BULK_LOAD_TABLES`.

    With the constraints in place, PostgreSQL checks every row against four
    other tables and keeps a unique index up to date while rows arrive in no
    particular order; a load is then about 25 times slower and the index about
    20 % larger. Restore them with `add_bulk_load_constraints` after the load,
    which validates all rows at once.

    Needs an exclusive lock on the referenced tables, so no other transaction
    may use them meanwhile.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
    """
    with engine.begin() as conn:
        for table_name in BULK_LOAD_TABLES:
            for constraint in _bulk_load_constraints(table_name):
                conn.execute(
                    text(
                        f"ALTER TABLE {table_name} "
                        f"DROP CONSTRAINT IF EXISTS {constraint.name};"
                    )
                )


def add_bulk_load_constraints(engine: engine.Engine) -> None:
    """(Re)create the missing constraints of `BULK_LOAD_TABLES`.

    Fails if the loaded data contains duplicate rows or unknown IDs.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
    """
    with engine.begin() as conn:
        existing = {
            row[0] for row in conn.execute(text("SELECT conname FROM pg_constraint"))
        }
        for table_name in BULK_LOAD_TABLES:
            for constraint in _bulk_load_constraints(table_name):
                if constraint.name not in existing:
                    conn.execute(AddConstraint(constraint))


def normalize_db_url(db_url: str) -> str:
    """Use the psycopg2 driver for plain ``postgresql://`` URLs.

    SQLAlchemy >= 2.1 uses psycopg 3 for plain ``postgresql://`` URLs. psycopg 3
    sends query parameters separately and allows at most 65535 of them, which
    large ``IN (...)`` filters (e.g. all grid points of a global request)
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


def initialize_database(db_url: str, replace: bool = False):
    """
    Initialize the database by creating the engine and tables, and installing PostGIS.
    If replace is True, it will drop and recreate the tables.

    Args:
        db_url (str): Database URL for SQLAlchemy.
        replace (bool): Whether to drop and recreate the tables. Defaults to False.
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


def assign_grid_resolution_group_to_grid_point(session: Session) -> None:
    """
    Assign the grid resolution group to each grid point,
    creating the many-to-many relationship between resolutions and
    grid points.

    Uses raw SQL INSERT...SELECT so the database performs the matching
    and insert directly, avoiding OOM when processing millions of grid points.

    This function is idempotent due to ON CONFLICT DO NOTHING - reruns will
    skip existing assignments without error. If you need to completely refresh
    assignments (e.g., after changing resolution logic), manually TRUNCATE
    the grid_point_resolution table before running this function.

    Args:
        session (Session): SQLAlchemy session object.
    """
    resolution_groups = session.query(ResolutionGroup).all()
    if not resolution_groups:
        raise ValueError("No resolution groups found in the database.")

    insert_sql = text("""
        INSERT INTO grid_point_resolution (grid_id, resolution_id)
        SELECT gp.id, :resolution_id
        FROM grid_point gp
        WHERE (ROUND(gp.latitude::numeric * 10)::integer % :resolution_mod = 0)
          AND (ROUND(gp.longitude::numeric * 10)::integer % :resolution_mod = 0)
        ON CONFLICT DO NOTHING
    """)

    try:
        for resolution_group in resolution_groups:
            resolution_mod = int(float(resolution_group.resolution) * 10)
            result = session.execute(
                insert_sql,
                {
                    "resolution_id": resolution_group.id,
                    "resolution_mod": resolution_mod,
                },
            )
            rowcount = result.rowcount
            if rowcount == 0:
                # Check if assignments already exist for this resolution
                existing_count = session.execute(
                    text("""
                        SELECT COUNT(*) FROM grid_point_resolution
                        WHERE resolution_id = :resolution_id
                    """),
                    {"resolution_id": resolution_group.id},
                ).scalar()
                if existing_count == 0:
                    raise ValueError(
                        f"No matching grid points for {resolution_group.resolution} degree resolution found."
                    )
                else:
                    print(
                        f"Resolution {resolution_group.resolution}° already has {existing_count} assignments (skipped)."
                    )
            else:
                print(
                    f"Assigned {rowcount} grid points to resolution {resolution_group.resolution}°."
                )
        session.commit()
        print("Grid point resolutions assigned.")
    except Exception:
        session.rollback()
        raise


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
            # Convert to Python float first, then apply _q() for consistent rounding
            lat_float = float(lat)
            lon_float = float(lon)
            grid_id_map[(_q(lat_float), _q(lon_float))] = grid_id

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


_PGCOPY_HEADER = b"PGCOPY\n\xff\r\n\x00" + struct.pack("!ii", 0, 0)
_PGCOPY_TRAILER = struct.pack("!h", -1)


def _pgcopy_binary(columns: list[tuple[str, np.ndarray | None]]) -> bytes:
    """Encode rows in the binary format of PostgreSQL's COPY.

    Args:
        columns (list[tuple[str, np.ndarray | None]]): One (dtype, values) pair
            per column, in table column order. `dtype` is a big-endian numpy
            type matching the column exactly (">i4" for integer, ">i2" for
            smallint, ">f4" for real). `values` is a 1D array with one entry
            per row, or None for a column that is NULL in every row.

    Returns:
        bytes: The COPY data, including header and trailer.
    """
    n_rows = next((len(v) for _, v in columns if v is not None), 0)
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
        else:
            if len(values) != n_rows:
                raise ValueError("All COPY columns must have the same length.")
            rows[f"len{k}"] = np.dtype(dtype).itemsize
            rows[f"val{k}"] = values
    return _PGCOPY_HEADER + rows.tobytes() + _PGCOPY_TRAILER


def copy_rows(
    engine: engine.Engine,
    table: str,
    column_names: list[str],
    columns: list[tuple[str, np.ndarray | None]],
) -> int:
    """Insert rows with a binary COPY, in one transaction.

    Much faster than INSERT statements, and no Python object is created per
    row. See `_pgcopy_binary` for `columns`.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        table (str): Table name.
        column_names (list[str]): Names of the columns in `columns`.
        columns (list[tuple[str, np.ndarray | None]]): Column data.

    Returns:
        int: Number of rows inserted.
    """
    data = _pgcopy_binary(columns)
    n_rows = next((len(v) for _, v in columns if v is not None), 0)
    if n_rows == 0:
        return 0
    connection = engine.raw_connection()
    try:
        with connection.cursor() as cursor:
            cursor.copy_expert(
                f"COPY {table} ({', '.join(column_names)}) "
                "FROM STDIN WITH (FORMAT BINARY)",
                io.BytesIO(data),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return n_rows


def grid_id_array(lats: np.ndarray, lons: np.ndarray, grid_id_map: dict) -> np.ndarray:
    """Look up the grid point IDs of a regular (latitude, longitude) grid once.

    Args:
        lats (np.ndarray): Latitudes.
        lons (np.ndarray): Longitudes.
        grid_id_map (dict): Mapping of (latitude, longitude) to grid point ID.

    Returns:
        np.ndarray: int64 array (len(lats), len(lons)) of grid point IDs, -1
            where the grid point is missing from `grid_id_map`.
    """
    ids = np.full((len(lats), len(lons)), -1, dtype=np.int64)
    q_lons = [_q(lon) for lon in lons]
    for i, lat in enumerate(lats):
        q_lat = _q(lat)
        ids[i] = [grid_id_map.get((q_lat, q_lon), -1) for q_lon in q_lons]
    n_missing = int((ids < 0).sum())
    if n_missing:
        print(f"Warning: {n_missing} grid points are not in the database.")
    return ids


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


def copy_var_values(
    engine: engine.Engine,
    values: np.ndarray,
    grid_ids: np.ndarray,
    time_ids: np.ndarray,
    var_id: int,
    resolution_id: int | None = None,
) -> int:
    """Insert the non-NaN values of a (time, latitude, longitude) array.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        values (np.ndarray): Values over (time, latitude, longitude).
        grid_ids (np.ndarray): Grid point IDs over (latitude, longitude), -1
            for grid points that are not in the database (skipped).
        time_ids (np.ndarray): Time point IDs, one per time step.
        var_id (int): Variable type ID.
        resolution_id (int | None): Resolution group of the values, None for
            variables with aggregation "subsample".

    Returns:
        int: Number of values inserted.
    """
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[1:] != grid_ids.shape:
        raise ValueError(
            f"Values of shape {values.shape} do not match the grid {grid_ids.shape}."
        )
    if len(time_ids) != values.shape[0]:
        raise ValueError("One time point ID per time step is needed.")
    total = 0
    known = grid_ids >= 0
    for t in range(values.shape[0]):
        keep = known & np.isfinite(values[t])
        n_rows = int(keep.sum())
        if n_rows == 0:
            continue
        resolution = None
        if resolution_id is not None:
            resolution = np.full(n_rows, resolution_id)
        total += copy_rows(
            engine,
            "var_value",
            ["grid_id", "time_id", "var_id", "resolution_id", "value"],
            [
                (">i4", grid_ids[keep]),
                (">i4", np.full(n_rows, time_ids[t])),
                (">i2", np.full(n_rows, var_id)),
                (">i2", resolution),
                (">f4", values[t][keep]),
            ],
        )
    return total


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
    # the chunk processing indexes values positionally as (time, lat, lon);
    # some model outputs (e.g. the P-Model) are written as (lon, lat, time)
    ds = _to_time_lat_lon(ds, var_name)

    if to_monthly:
        print(f"Converting {var_name} data from yearly to monthly...")
        ds = convert_yearly_to_monthly(ds)
    t_yearly_to_monthly = time.time()

    print(f"Prepare inserting {var_name} values...")

    var_id = var_id_map.get(var_name)
    if var_id is None:
        raise ValueError(f"Variable {var_name} not found in var_type table.")
    _check_no_coverage(engine, var_id, var_name)

    grid_ids = grid_id_array(ds.latitude.values, ds.longitude.values, grid_id_map)

    # grid points with a value at every time point: zeros there are not stored
    covered = ds[var_name].notnull().all("time").compute()
    data = drop_covered_zeros(ds[var_name], covered)
    n_covered = insert_var_coverage(
        engine,
        var_id,
        ds.latitude.values,
        ds.longitude.values,
        covered.values,
        grid_id_map,
        grid_ids=grid_ids,
    )
    print(f"Coverage of {var_name}: {n_covered} grid points.")

    print(f"Start inserting {var_name} values in streaming chunks...")
    t_start_insert = time.time()

    total = 0
    n_time = data.sizes["time"]
    for t0 in range(0, n_time, VAR_TIME_CHUNK):
        chunk = data.isel(time=slice(t0, min(t0 + VAR_TIME_CHUNK, n_time)))
        total += copy_var_values(
            engine,
            chunk.values,
            grid_ids,
            time_id_array(chunk.time.values, time_id_map),
            var_id,
        )
        del chunk
        gc.collect()
    print(f"Values of {var_name} inserted. Total: {total}")
    return t_yearly_to_monthly, t_start_insert


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
    if not floor > 0:
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

    Uses the same rule as `assign_grid_resolution_group_to_grid_point`, so the
    selected points are exactly the grid points of that resolution group.
    """
    tenths = np.round(coords.astype(np.float64) * 10).astype(np.int64)
    return np.flatnonzero(tenths % round(float(resolution) * 10) == 0)


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


def block_valid(valid: np.ndarray, plan: dict) -> np.ndarray:
    """Coarse cells for which `block_mean` gives a value.

    Args:
        valid (np.ndarray): Boolean fine-grid array (lat, lon), True where the
            fine cell has a value.
        plan (dict): Plan from `_block_mean_plan`.

    Returns:
        np.ndarray: Boolean array (coarse lat, coarse lon).
    """
    if plan["factor"] == 1:
        return valid.astype(bool)
    weight = valid * plan["area_weight"][:, None]
    return (plan["lat_matrix"] @ weight @ plan["lon_matrix_t"]) > 0


def drop_covered_zeros(values, covered):
    """Set values that are 0 at covered grid points to NaN, so they are not stored.

    Args:
        values: numpy array or xarray object over (time, latitude, longitude).
        covered: Boolean (latitude, longitude) array, see `VarCoverage`.

    Returns:
        The values with those zeros replaced by NaN, same type as `values`.
    """
    drop = (values == 0) & covered
    if isinstance(values, xr.DataArray):
        return values.where(~drop)
    return np.where(drop, np.nan, values)


def _check_no_coverage(engine: engine.Engine, var_id: int, var_name: str) -> None:
    """Raise if values of a variable were already inserted with coverage.

    The coverage of a variable is computed over all its time points at once, so
    a variable cannot be inserted in several parts.
    """
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM var_coverage WHERE var_id = :v LIMIT 1"),
            {"v": int(var_id)},
        ).first()
    if exists:
        raise ValueError(
            f"Values of {var_name} are already in the database; a variable has "
            "to be inserted in one go (truncate the tables to reload it)."
        )


def insert_var_coverage(
    engine: engine.Engine,
    var_id: int,
    lats: np.ndarray,
    lons: np.ndarray,
    covered: np.ndarray,
    grid_id_map: dict,
    resolution_id: int | None = None,
    grid_ids: np.ndarray | None = None,
) -> int:
    """Store the grid points where a variable has a value at every time point.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        var_id (int): Variable type ID.
        lats (np.ndarray): Latitudes of `covered`.
        lons (np.ndarray): Longitudes of `covered`.
        covered (np.ndarray): Boolean (lat, lon) array.
        grid_id_map (dict): Mapping of (latitude, longitude) to grid point ID.
        resolution_id (int | None): Resolution group, None for variables with
            aggregation "subsample".
        grid_ids (np.ndarray | None): Grid point IDs of (lats, lons) from
            `grid_id_array`, looked up from `grid_id_map` if None.

    Returns:
        int: Number of grid points stored.
    """
    if grid_ids is None:
        grid_ids = grid_id_array(lats, lons, grid_id_map)
    keep = np.asarray(covered, dtype=bool) & (grid_ids >= 0)
    n_rows = int(keep.sum())
    resolution = None if resolution_id is None else np.full(n_rows, resolution_id)
    return copy_rows(
        engine,
        "var_coverage",
        ["var_id", "resolution_id", "grid_id"],
        [
            (">i2", np.full(n_rows, var_id)),
            (">i2", resolution),
            (">i4", grid_ids[keep]),
        ],
    )


def insert_var_values_by_resolution(
    engine: engine.Engine,
    ds: xr.Dataset,
    var_name: str,
    grid_id_map: dict,
    time_id_map: dict,
    var_id_map: dict,
    resolution_id_map: dict[float, int],
    mask: xr.DataArray | None = None,
    transform: str | None = None,
    transform_floor: float = 1.0,
) -> int:
    """Insert a variable whose value depends on the grid resolution.

    For every resolution group that is an integer multiple of the data's grid
    step, the area-weighted mean over each coarse cell is computed and stored
    with that group's `resolution_id`. The native resolution is stored as is.
    Resolutions that are finer than the data, or not a multiple of its step,
    are skipped.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        ds (xr.Dataset): Dataset with `var_name` over (time, latitude, longitude),
            in any dimension order.
        var_name (str): Name of the variable to insert.
        grid_id_map (dict): Mapping of (latitude, longitude) to grid point ID.
        time_id_map (dict): Mapping of datetime64 to time point ID.
        var_id_map (dict): Mapping of variable name to variable type ID.
        resolution_id_map (dict[float, int]): Mapping of resolution to group ID.
        mask (xr.DataArray | None): Boolean (time, latitude, longitude) array on
            the same grid. Fine cells where it is False are set to NaN before
            aggregating, e.g. to drop ocean cells. Defaults to None.
        transform (str | None): Transform applied after aggregating, see
            `apply_transform`. Means are always taken over the original values.
        transform_floor (float): Floor for the transform. Defaults to 1.

    Returns:
        int: Total number of values inserted over all resolutions.
    """
    # fail early, before any data is read
    apply_transform(np.array([1.0]), transform, transform_floor)
    var_id = var_id_map.get(var_name)
    if var_id is None:
        raise ValueError(f"Variable {var_name} not found in var_type table.")
    data = _to_time_lat_lon(ds, var_name)[var_name]
    lats = data.latitude.values
    lons = data.longitude.values

    plans = []
    for resolution, resolution_id in sorted(resolution_id_map.items()):
        plan = _block_mean_plan(lats, lons, resolution)
        if plan is None:
            print(
                f"Skipping {resolution} degree resolution for {var_name}: "
                "not a multiple of the data grid step."
            )
            continue
        plans.append((resolution, resolution_id, plan))
    if not plans or plans[0][2]["factor"] != 1:
        raise ValueError(
            f"No resolution group matches the grid step of {var_name}; "
            "add it to the resolution groups."
        )

    if mask is not None:
        mask = mask.transpose("time", "latitude", "longitude")
        if not (
            np.array_equal(mask.latitude.values, lats)
            and np.array_equal(mask.longitude.values, lons)
        ):
            raise ValueError(f"The mask for {var_name} is on a different grid.")

    _check_no_coverage(engine, var_id, var_name)
    grid_ids = {
        resolution_id: grid_id_array(plan["latitude"], plan["longitude"], grid_id_map)
        for _, resolution_id, plan in plans
    }

    def masked_chunks():
        n_time = data.sizes["time"]
        for t0 in range(0, n_time, VAR_TIME_CHUNK):
            chunk = data.isel(time=slice(t0, min(t0 + VAR_TIME_CHUNK, n_time)))
            times = chunk.time.values
            values = chunk.values.astype(np.float64)
            if mask is not None:
                try:
                    mask_values = mask.sel(time=times).values.astype(bool)
                except KeyError as e:
                    raise ValueError(
                        f"The mask for {var_name} does not cover all time points."
                    ) from e
                values = np.where(mask_values, values, np.nan)
            yield t0, times, values

    # first pass: coarse cells that have a value at every time point
    covered = {
        resolution_id: np.ones(
            (len(plan["latitude"]), len(plan["longitude"])), dtype=bool
        )
        for _, resolution_id, plan in plans
    }
    for _, _, values in masked_chunks():
        for _, resolution_id, plan in plans:
            for t in range(values.shape[0]):
                covered[resolution_id] &= block_valid(np.isfinite(values[t]), plan)
    for resolution, resolution_id, plan in plans:
        n_covered = insert_var_coverage(
            engine,
            var_id,
            plan["latitude"],
            plan["longitude"],
            covered[resolution_id],
            grid_id_map,
            resolution_id=resolution_id,
            grid_ids=grid_ids[resolution_id],
        )
        print(f"Coverage of {var_name} at {resolution} degree: {n_covered} points.")

    # second pass: values, without the zeros at covered grid points
    total = 0
    for t0, times, values in masked_chunks():
        time_ids = time_id_array(times, time_id_map)
        for resolution, resolution_id, plan in plans:
            coarse = values if plan["factor"] == 1 else block_mean(values, plan)
            coarse = apply_transform(coarse, transform, transform_floor)
            coarse = drop_covered_zeros(coarse, covered[resolution_id])
            total += copy_var_values(
                engine,
                coarse,
                grid_ids[resolution_id],
                time_ids,
                var_id,
                resolution_id=resolution_id,
            )
            print(
                f"{var_name}: inserted {resolution} degree values for time "
                f"indices {t0}:{t0 + len(times)}."
            )
        del values
        gc.collect()

    print(f"Values of {var_name} inserted for all resolutions. Total: {total}")
    return total


def get_var_value(
    session: Session,
    var_name: str,
    lat: float,
    lon: float,
    year: int,
    month: int,
    day: int,
    resolution: float | None = None,
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
        resolution (float | None): Grid resolution in degrees, only used for
            variables stored per resolution. If None, the finest available
            resolution is returned. Defaults to None.

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

    query = (
        session.query(VarValue)
        .join(GridPoint, VarValue.grid_id == GridPoint.id)
        .join(TimePoint, VarValue.time_id == TimePoint.id)
        .join(VarType, VarValue.var_id == VarType.id)
        .outerjoin(ResolutionGroup, VarValue.resolution_id == ResolutionGroup.id)
        .filter(
            GridPoint.latitude == lat,
            GridPoint.longitude == lon,
            TimePoint.year == year,
            TimePoint.month == month,
            TimePoint.day == day,
            VarType.name == var_name,
        )
    )
    if resolution is not None:
        query = query.filter(
            (VarValue.resolution_id.is_(None))
            | (ResolutionGroup.resolution == resolution)
        )
    result = query.order_by(ResolutionGroup.resolution.asc().nulls_first()).first()
    if result is not None:
        return result.value

    # no stored value: 0 if the grid point is covered (see VarCoverage)
    time_exists = (
        session.query(TimePoint.id)
        .filter(TimePoint.year == year, TimePoint.month == month, TimePoint.day == day)
        .first()
    )
    if time_exists is None:
        return None
    coverage = (
        session.query(VarCoverage.grid_id)
        .join(GridPoint, VarCoverage.grid_id == GridPoint.id)
        .join(VarType, VarCoverage.var_id == VarType.id)
        .outerjoin(ResolutionGroup, VarCoverage.resolution_id == ResolutionGroup.id)
        .filter(
            GridPoint.latitude == lat,
            GridPoint.longitude == lon,
            VarType.name == var_name,
        )
    )
    if resolution is not None:
        coverage = coverage.filter(
            (VarCoverage.resolution_id.is_(None))
            | (ResolutionGroup.resolution == resolution)
        )
    # like the values: the finest resolution unless one is requested
    covered = coverage.order_by(ResolutionGroup.resolution.asc().nulls_first()).first()
    return 0.0 if covered is not None else None


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


def get_grid_ids_by_resolution(session: Session, resolution_id: int) -> list[int]:
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
    area: None | tuple[float, float, float, float] = None,
    resolution_id: int | None = None,
) -> list[GridPoint]:
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


def sort_grid_points_get_ids(
    grid_points: list[GridPoint],
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

    # grid points of the requested resolution in the requested area
    grid_query = session.query(GridPoint.id).join(
        GridPointResolution,
        and_(
            GridPointResolution.grid_id == GridPoint.id,
            GridPointResolution.resolution_id == resolution_id,
        ),
    )
    if area is not None:
        north, west, south, east = area
        area_filter = and_(
            GridPoint.latitude <= north,
            GridPoint.latitude >= south,
            GridPoint.longitude >= west,
            GridPoint.longitude <= east,
        )
        grid_query = grid_query.filter(area_filter)
    if grid_query.first() is None:
        print(
            "No grid points found in the specified area or with the specified resolution."
        )
        raise HTTPException(
            status_code=400,
            detail="No grid points found in specified area or with the specified resolution.",
        )

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

    # variables that are aggregated per resolution store one row per
    # resolution group; the others store a single row shared by all resolutions
    if var_type.aggregation == "subsample":
        value_resolution = VarValue.resolution_id.is_(None)
        coverage_resolution = VarCoverage.resolution_id.is_(None)
    else:
        value_resolution = VarValue.resolution_id == resolution_id
        coverage_resolution = VarCoverage.resolution_id == resolution_id

    # values of the grid points; at covered grid points a missing row means 0
    # (see VarCoverage), elsewhere it means that there is no data
    query = (
        session.query(
            GridPoint.latitude,
            GridPoint.longitude,
            func.coalesce(VarValue.value, 0.0),
        )
        .join(
            GridPointResolution,
            and_(
                GridPointResolution.grid_id == GridPoint.id,
                GridPointResolution.resolution_id == resolution_id,
            ),
        )
        .outerjoin(
            VarValue,
            and_(
                VarValue.grid_id == GridPoint.id,
                VarValue.time_id == time_id,
                VarValue.var_id == var_id,
                value_resolution,
            ),
        )
        .outerjoin(
            VarCoverage,
            and_(
                VarCoverage.grid_id == GridPoint.id,
                VarCoverage.var_id == var_id,
                coverage_resolution,
            ),
        )
        .filter(or_(VarValue.value.isnot(None), VarCoverage.grid_id.isnot(None)))
    )
    if area is not None:
        query = query.filter(area_filter)
    values = query.order_by(GridPoint.latitude, GridPoint.longitude).all()
    # Convert directly to list of tuples
    values_list = [(np.float64(lat), np.float64(lon), val) for lat, lon, val in values]

    mydict = {"latitude, longitude, var_value": values_list}
    return mydict


def get_var_values_cartesian_for_download(
    session: Session,
    start_time_point: tuple[int, int],
    end_time_point: tuple[int, int] | None = None,
    area: None | tuple[float, float, float, float] = None,
    var_names: None | list[str] = None,
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

    # variables stored per resolution are exported at their finest resolution
    resolution_by_id = {rid: res for res, rid in get_resolution_id_map(session).items()}

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
        covered_ids = [
            (row.grid_id, row.resolution_id)
            for row in session.query(VarCoverage.grid_id, VarCoverage.resolution_id)
            .filter(
                VarCoverage.var_id == vt.id,
                VarCoverage.grid_id.in_(grid_ids.keys()),
            )
            .all()
        ]
        if vt.aggregation != "subsample":
            finest_id = min(
                {vv.resolution_id for vv in values if vv.resolution_id is not None}
                | {rid for _, rid in covered_ids if rid is not None},
                key=lambda rid: resolution_by_id[rid],
                default=None,
            )
            values = [vv for vv in values if vv.resolution_id == finest_id]
            covered_ids = [(gid, rid) for gid, rid in covered_ids if rid == finest_id]

        # dummy values array
        values_array = np.full(
            (len(time_values), len(latitudes), len(longitudes)),
            np.nan,
            dtype=np.float64,
        )
        # covered grid points without a stored value are 0 (see VarCoverage)
        for grid_id, _ in covered_ids:
            lat_index, lon_index = grid_ids[grid_id]
            values_array[:, lat_index, lon_index] = 0.0

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
) -> list[int]:
    """Get grid point IDs that are within the NUTS regions.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        nuts_regions (gpd.GeoDataFrame): GeoDataFrame with NUTS region geometries.

    Returns:
        List[int]: List of grid point IDs that intersect with the NUTS regions.
    """
    if nuts_regions.empty:
        return []

    # the points are built from latitude/longitude, and only for the grid
    # points in the bounding box of the regions, not for the whole grid
    west, south, east, north = nuts_regions.to_crs(epsg=CRS).total_bounds
    sql = text(f"""
    SELECT id,
        ST_SetSRID(
            ST_MakePoint(longitude::float8, latitude::float8), {CRS}
        ) AS geometry
    FROM grid_point
    WHERE latitude BETWEEN :south AND :north
        AND longitude BETWEEN :west AND :east
    """)

    # turn the grid points into a GeoDataFrame
    grid_points_gdf = gpd.read_postgis(
        sql,
        engine,
        geom_col="geometry",
        crs=f"EPSG:{CRS!s}",
        params={
            "south": float(south),
            "north": float(north),
            "west": float(west),
            "east": float(east),
        },
    )
    if grid_points_gdf.empty:
        return []
    nuts_regions = nuts_regions.to_crs(epsg=CRS)

    # filter grid points that intersect with NUTS regions
    filtered_grid_points_gdf = gpd.sjoin(
        grid_points_gdf, nuts_regions, how="inner", predicate="intersects"
    )

    return sorted(set(filtered_grid_points_gdf["id"].tolist()))


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
