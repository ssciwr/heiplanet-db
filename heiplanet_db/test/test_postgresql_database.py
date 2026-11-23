import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import cleanup
from fastapi import HTTPException
from shapely.geometry import Polygon
from sqlalchemy import inspect, text
from sqlalchemy.orm.session import Session
from testcontainers.postgres import PostgresContainer

from heiplanet_db import postgresql_database as postdb

# for local docker desktop,
# environ["DOCKER_HOST"] is "unix:///home/[user]/.docker/desktop/docker.sock"


def test_install_postgis(get_engine_with_tables):
    postdb.install_postgis(get_engine_with_tables)
    # check if postgis extension is installed
    with get_engine_with_tables.connect() as conn:
        result = conn.execute(text("SELECT postgis_full_version();"))
        version_text = result.fetchone()
        assert version_text is not None
        assert "POSTGIS=" in version_text[0]


def test_create_session(get_engine_with_tables):
    session = postdb.create_session(get_engine_with_tables)
    assert session is not None
    assert isinstance(session, Session)
    assert session.bind is not None
    session.close()


def get_missing_tables(engine):
    inspector = inspect(engine)
    expected_tables = {
        "nuts_def",
        "grid_point",
        "time_point",
        "var_type",
        "var_value",
        "var_value_nuts",
        "resolution_group",
        "grid_point_resolution",
    }
    existing_tables = set(inspector.get_table_names(schema="public"))
    missing_tables = expected_tables - existing_tables
    return missing_tables


def test_create_tables(get_engine_without_tables):
    postdb.create_tables(get_engine_without_tables)
    missing_tables = get_missing_tables(get_engine_without_tables)
    assert not missing_tables, f"Missing tables: {missing_tables}"

    # clean up
    cleanup(get_engine_without_tables)


def test_create_or_replace_tables(get_engine_without_tables):
    postdb.create_tables(get_engine_without_tables)
    postdb.create_or_replace_tables(get_engine_without_tables)
    missing_tables = get_missing_tables(get_engine_without_tables)
    assert not missing_tables, f"Missing tables: {missing_tables}"

    # clean up
    cleanup(get_engine_without_tables)


def test_initialize_database(get_docker_image):
    with PostgresContainer(get_docker_image) as postgres:
        db_url = postgres.get_connection_url()

        # first initialization
        engine1 = postdb.initialize_database(db_url, replace=True)
        missing_tables = get_missing_tables(engine1)
        assert not missing_tables, f"Missing tables: {missing_tables}"
        with engine1.connect() as conn:
            result = conn.execute(text("SELECT postgis_full_version();"))
            version_text = result.fetchone()
            assert version_text is not None
            assert "POSTGIS=" in version_text[0]

        # add sample data to var_type table
        session = postdb.create_session(engine1)
        new_var_type = postdb.VarType(
            name="test_var",
            unit="1",
            description="Test variable",
        )
        session.add(new_var_type)
        session.commit()
        session.close()

        # initialize again without replacing
        engine2 = postdb.initialize_database(db_url, replace=False)
        # check if the data is still there
        session = postdb.create_session(engine2)
        var_types = session.query(postdb.VarType).all()
        assert len(var_types) == 1
        assert var_types[0].name == "test_var"
        session.close()

        # initialize again with replacing
        engine3 = postdb.initialize_database(db_url, replace=True)
        missing_tables = get_missing_tables(engine3)
        assert not missing_tables, f"Missing tables: {missing_tables}"
        # all tables should be empty
        session = postdb.create_session(engine3)
        assert session.query(postdb.VarType).count() == 0
        session.close()

        # clean up
        cleanup(engine1)
        cleanup(engine2)
        cleanup(engine3)


def test_insert_nuts_def(
    get_engine_with_tables, get_session, tmp_path, get_nuts_def_data
):
    nuts_path = tmp_path / "nuts_def.shp"
    gdf_nuts_data = get_nuts_def_data
    gdf_nuts_data.to_file(nuts_path, driver="ESRI Shapefile")

    postdb.insert_nuts_def(get_engine_with_tables, nuts_path)

    result = get_session.query(postdb.NutsDef).all()
    assert len(result) == 3
    assert result[0].nuts_id == "DE11"
    assert result[0].name_latn == "Test NUTS"
    assert result[1].nuts_id == "DE22"
    assert result[1].name_latn == "Test NUTS2"
    assert result[2].nuts_id == "DE501"
    assert result[2].name_latn == "Test NUTS3"

    # clean up
    get_session.execute(text("TRUNCATE TABLE nuts_def RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_add_data_list(get_session):
    data_list = [
        postdb.VarType(
            name="test_var",
            unit="1",
            description="Test variable",
        ),
        postdb.VarType(
            name="test_var2",
            unit="1",
            description="Test variable 2",
        ),
    ]
    postdb.add_data_list(get_session, data_list)

    result = get_session.query(postdb.VarType).all()
    assert len(result) == 2
    assert result[0].name == "test_var"
    assert result[1].name == "test_var2"

    # clean up
    get_session.execute(text("TRUNCATE TABLE var_type RESTART IDENTITY CASCADE"))
    get_session.commit()


@pytest.mark.filterwarnings("ignore::sqlalchemy.exc.SAWarning")
def test_add_data_list_invalid(get_session, capsys):
    # non unique name
    data_list = [
        postdb.VarType(name="test_var", unit="1"),
        postdb.VarType(name="test_var", unit="1"),
    ]
    postdb.add_data_list(get_session, data_list)
    result = get_session.query(postdb.VarType).all()
    assert len(result) == 0
    captured = capsys.readouterr()
    assert "Error inserting data:" in captured.out

    # missing required fields
    data_list = [postdb.VarType(name="test_var")]
    postdb.add_data_list(get_session, data_list)
    result = get_session.query(postdb.VarType).all()
    assert len(result) == 0
    captured = capsys.readouterr()
    assert "Error inserting data:" in captured.out

    # invalid data type
    # there is no exception raised for this case
    # TODO check it again


def test_add_data_list_bulk(get_session, get_var_type_list):
    postdb.add_data_list_bulk(get_session, get_var_type_list, postdb.VarType)

    result = get_session.query(postdb.VarType).all()
    assert len(result) == 2
    assert result[0].name == "test_var"
    assert result[1].name == "test_var2"

    # clean up
    get_session.execute(text("TRUNCATE TABLE var_type RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_add_data_list_bulk_empty(get_session):
    data_list = []
    postdb.add_data_list_bulk(get_session, data_list, postdb.VarType)

    result = get_session.query(postdb.VarType).all()
    assert len(result) == 0


@pytest.mark.filterwarnings("ignore::sqlalchemy.exc.SAWarning")
def test_add_data_list_bulk_invalid(get_session, capsys):
    # non unique name
    data_list = [{"name": "test_var", "unit": "1"}, {"name": "test_var", "unit": "1"}]
    with pytest.raises(RuntimeError):
        postdb.add_data_list_bulk(get_session, data_list, postdb.VarType)
    assert get_session.query(postdb.VarType).count() == 0

    # missing required fields
    data_list = [{"name": "test_var"}]
    with pytest.raises(RuntimeError):
        postdb.add_data_list_bulk(get_session, data_list, postdb.VarType)
    assert get_session.query(postdb.VarType).count() == 0
    # invalid data type
    # there is no exception raised for this case
    # TODO check it again


def test_insert_grid_points(get_session):
    latitudes = np.array([0.0, 1.0])
    longitudes = np.array([0.0, 1.0])
    postdb.insert_grid_points(get_session, latitudes, longitudes)

    result = get_session.query(postdb.GridPoint).all()
    assert len(result) == 4
    assert math.isclose(result[0].latitude, 0.0, abs_tol=1e-5)
    assert math.isclose(result[0].longitude, 0.0, abs_tol=1e-5)
    assert math.isclose(result[1].latitude, 0.0, abs_tol=1e-5)
    assert math.isclose(result[1].longitude, 1.0, abs_tol=1e-5)

    # clean up
    get_session.execute(text("TRUNCATE TABLE grid_point RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_insert_resolution_groups(get_session):
    # first default case
    postdb.insert_resolution_groups(get_session)
    result = get_session.query(postdb.ResolutionGroup).all()
    assert len(result) == 9
    assert math.isclose(result[0].resolution, 0.1, abs_tol=1e-5)
    assert result[0].description == "0.1 degree resolution"


def test_assign_grid_resolution_group_to_grid_point(get_session):
    # insert resolution groups first
    postdb.insert_resolution_groups(get_session)
    # insert grid points - mix of 0.1 and 0.2 degree resolution
    latitudes = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    longitudes = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    postdb.insert_grid_points(get_session, latitudes, longitudes)
    # assign resolution groups
    postdb.assign_grid_resolution_group_to_grid_point(get_session)
    # check results
    # 0.1 degree resolution: all points
    # 0.2 degree resolution: points where round(lat*10) % 2 == 0 AND round(lon*10) % 2 == 0
    # 0.5 degree resolution: points where round(lat*10) % 5 == 0 AND round(lon*10) % 5 == 0
    result = get_session.query(postdb.GridPointResolution).all()
    # All 25 grid points (5x5) should be in 0.1 degree resolution
    # 9 grid points should be in 0.2 degree resolution
    # 1 grid point should be each in 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0 degree resolution
    assert len(result) == 41, f"Expected 40 entries, got {len(result)}"
    # Check that all grid points have 0.1 degree resolution
    grid_points = get_session.query(postdb.GridPoint).all()
    assert len(grid_points) == 25
    resolution_0_1 = (
        get_session.query(postdb.ResolutionGroup)
        .filter(postdb.ResolutionGroup.resolution == 0.1)
        .first()
    )
    assert math.isclose(resolution_0_1.resolution, 0.1, abs_tol=1e-5)
    # get all the grid points with resolution_0_1.id
    gp_0_1 = (
        get_session.query(postdb.GridPoint)
        .join(
            postdb.GridPointResolution,
            postdb.GridPoint.id == postdb.GridPointResolution.grid_id,
        )
        .filter(postdb.GridPointResolution.resolution_id == resolution_0_1.id)
        .all()
    )
    assert gp_0_1 is not None
    assert len(gp_0_1) == 25, (
        f"All 25 grid points should have 0.1 degree resolution, got {len(gp_0_1)}"
    )
    assert math.isclose(gp_0_1[0].latitude, 0.0, abs_tol=1e-5)
    assert math.isclose(gp_0_1[0].longitude, 0.0, abs_tol=1e-5)
    assert math.isclose(gp_0_1[1].latitude, 0.0, abs_tol=1e-5)
    assert math.isclose(gp_0_1[1].longitude, 0.1, abs_tol=1e-5)
    # Check that 0.2 degree resolution points are correct
    resolution_0_2 = (
        get_session.query(postdb.ResolutionGroup)
        .filter(postdb.ResolutionGroup.resolution == 0.2)
        .first()
    )
    gp_0_2 = (
        get_session.query(postdb.GridPoint)
        .join(
            postdb.GridPointResolution,
            postdb.GridPoint.id == postdb.GridPointResolution.grid_id,
        )
        .filter(postdb.GridPointResolution.resolution_id == resolution_0_2.id)
        .all()
    )
    assert len(gp_0_2) == 9, (
        f"Should have 9 grid points at 0.2 degree resolution, got {len(gp_0_2)}"
    )
    assert math.isclose(gp_0_2[0].latitude, 0.0, abs_tol=1e-5)
    assert math.isclose(gp_0_2[0].longitude, 0.0, abs_tol=1e-5)
    assert math.isclose(gp_0_2[1].latitude, 0.0, abs_tol=1e-5)
    assert math.isclose(gp_0_2[1].longitude, 0.2, abs_tol=1e-5)
    # check that 0.5 degree resolution points are correct
    resolution_0_5 = (
        get_session.query(postdb.ResolutionGroup)
        .filter(postdb.ResolutionGroup.resolution == 0.5)
        .first()
    )
    # here we should have 1 grid point: (0.0,0.0)
    gp_0_5 = (
        get_session.query(postdb.GridPoint)
        .join(
            postdb.GridPointResolution,
            postdb.GridPoint.id == postdb.GridPointResolution.grid_id,
        )
        .filter(postdb.GridPointResolution.resolution_id == resolution_0_5.id)
        .all()
    )
    assert math.isclose(gp_0_5[0].latitude, 0.0, abs_tol=1e-5)
    assert math.isclose(gp_0_5[0].longitude, 0.0, abs_tol=1e-5)
    # clean up
    get_session.execute(
        text("TRUNCATE TABLE grid_point_resolution RESTART IDENTITY CASCADE")
    )
    get_session.execute(
        text("TRUNCATE TABLE resolution_group RESTART IDENTITY CASCADE")
    )
    get_session.execute(text("TRUNCATE TABLE grid_point RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_assign_grid_resolution_group_to_grid_point_no_resolution_group(get_session):
    # insert grid points - mix of 0.1 and 0.2 degree resolution
    latitudes = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    longitudes = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    postdb.insert_grid_points(get_session, latitudes, longitudes)
    # assign resolution groups without inserting resolution groups
    with pytest.raises(ValueError):
        postdb.assign_grid_resolution_group_to_grid_point(get_session)
    # clean up
    get_session.execute(text("TRUNCATE TABLE grid_point RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_assign_grid_resolution_group_to_grid_point_idempotent(get_session):
    # Arrange: insert resolution groups and grid points
    postdb.insert_resolution_groups(get_session)
    latitudes = np.array([0.0, 0.1, 0.2])
    longitudes = np.array([0.0, 0.1, 0.2])
    postdb.insert_grid_points(get_session, latitudes, longitudes)

    # Act: first assignment
    postdb.assign_grid_resolution_group_to_grid_point(get_session)
    first_count = get_session.query(postdb.GridPointResolution).count()
    assert first_count > 0

    # Act again: second assignment must not raise and must not create duplicates
    postdb.assign_grid_resolution_group_to_grid_point(get_session)
    second_count = get_session.query(postdb.GridPointResolution).count()

    # Assert: idempotent behaviour (same number of relationships)
    assert second_count == first_count


def test_extract_time_point():
    time_points = {
        np.datetime64("2024-01-01T00:00:00.000000000"): (2024, 1, 1),
        np.datetime64("2023-02-01 00:00:00"): (2023, 2, 1),
    }
    for time_point, expected_data in time_points.items():
        year, month, day, _, _, _ = postdb.extract_time_point(time_point)
        assert (year, month, day) == expected_data


def test_extract_time_point_invalid():
    with pytest.raises(TypeError):
        postdb.extract_time_point("2024-01-01")

    with pytest.raises(TypeError):
        postdb.extract_time_point(1)


def test_get_unique_time_points(get_time_point_lists):
    unique_time_points = postdb.get_unique_time_points(get_time_point_lists)
    assert len(unique_time_points) == 24
    assert unique_time_points[0] == np.datetime64("2023-01-01", "ns")
    assert unique_time_points[-1] == np.datetime64("2024-12-01", "ns")


def test_insert_time_points(get_session, get_time_point_lists):
    postdb.insert_time_points(get_session, get_time_point_lists)

    result = get_session.query(postdb.TimePoint).all()
    assert len(result) == 24
    assert result[0].year == 2023
    assert result[0].month == 1
    assert result[0].day == 1
    assert result[-1].year == 2024
    assert result[-1].month == 12
    assert result[-1].day == 1

    # clean up
    get_session.execute(text("TRUNCATE TABLE time_point RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_insert_var_types(get_session, get_var_type_list):
    postdb.insert_var_types(get_session, get_var_type_list)

    result = get_session.query(postdb.VarType).all()
    assert len(result) == 2
    assert result[0].name == "test_var"

    # clean up
    get_session.execute(text("TRUNCATE TABLE var_type RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_get_id_maps(get_session, insert_data):
    grid_id_map, time_id_map, var_id_map = postdb.get_id_maps(get_session)

    assert len(grid_id_map) == 6
    assert grid_id_map[(10.1, 10.1)] == 1
    assert len(time_id_map) == 2
    assert time_id_map[np.datetime64("2023-01-01", "ns")] == 1
    assert len(var_id_map) == 1
    assert var_id_map["t2m"] == 1

    # clean up
    get_session.execute(text("TRUNCATE TABLE grid_point RESTART IDENTITY CASCADE"))
    get_session.execute(text("TRUNCATE TABLE time_point RESTART IDENTITY CASCADE"))
    get_session.execute(text("TRUNCATE TABLE var_type RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_convert_monthly_to_yearly(get_dataset):
    assert get_dataset.sizes == {"time": 2, "latitude": 2, "longitude": 3}
    monthly_dataset = postdb.convert_yearly_to_monthly(get_dataset)
    assert monthly_dataset.sizes == {"time": 24, "latitude": 2, "longitude": 3}
    assert monthly_dataset.t2m.shape == (24, 2, 3)
    assert monthly_dataset.t2m[0, 0, 0] == get_dataset.t2m[0, 0, 0]
    assert monthly_dataset.t2m[1, 0, 0] == get_dataset.t2m[0, 0, 0]
    assert monthly_dataset.t2m[2, 0, 0] == get_dataset.t2m[0, 0, 0]
    assert monthly_dataset.t2m[11, 0, 0] == get_dataset.t2m[0, 0, 0]
    assert monthly_dataset.t2m[12, 0, 0] == get_dataset.t2m[1, 0, 0]


def test_insert_var_values_no_to_monthly(
    get_session, get_engine_with_tables, get_dataset, seed_base_data
):
    # get the id maps
    grid_id_map, time_id_map, var_id_map = postdb.get_id_maps(get_session)

    # insert var values
    postdb.insert_var_values(
        get_engine_with_tables,
        get_dataset,
        "t2m",
        grid_id_map,
        time_id_map,
        var_id_map,
        to_monthly=False,
    )

    # check if the data is inserted correctly
    session2 = postdb.create_session(get_engine_with_tables)
    result = session2.query(postdb.VarValue).all()
    assert len(result) == 12
    assert result[0].grid_id == 1
    assert result[0].time_id == 1
    assert result[0].var_id == 1
    session2.close()


def test_insert_var_values_no_to_monthly_no_var(
    get_engine_with_tables, get_dataset, get_session, seed_base_data
):
    # get the id maps
    grid_id_map, time_id_map, var_id_map = postdb.get_id_maps(get_session)

    # error due to no test_var var type
    with pytest.raises(ValueError):
        postdb.insert_var_values(
            get_engine_with_tables,
            get_dataset,
            "test_var",
            grid_id_map,
            time_id_map,
            var_id_map,
            to_monthly=False,
        )


def test_insert_var_values_to_monthly(get_engine_with_tables, get_dataset, get_session):
    postdb.insert_var_types(
        get_session, [{"name": "t2m", "unit": "K", "description": "2m temperature"}]
    )
    postdb.insert_grid_points(
        get_session, get_dataset.latitude.values, get_dataset.longitude.values
    )
    postdb.insert_time_points(get_session, [(get_dataset.time.values, True)])
    # get the id maps
    grid_id_map, time_id_map, var_id_map = postdb.get_id_maps(get_session)
    # insert var values
    postdb.insert_var_values(
        get_engine_with_tables,
        get_dataset,
        "t2m",
        grid_id_map,
        time_id_map,
        var_id_map,
        to_monthly=True,
    )

    # check if the data is inserted correctly
    result = (
        get_session.query(postdb.VarValue)
        .order_by(postdb.VarValue.grid_id, postdb.VarValue.time_id)
        .all()
    )
    assert len(result) == 144
    assert result[0].grid_id == 1
    assert result[0].time_id == 1
    assert result[0].var_id == 1
    assert result[0].value == result[6].value  # same year, different month


def _build_cartesian_ds(var_name: str = "t2m", fill_nan: bool = False) -> xr.Dataset:
    times = np.array(["2023-01-01", "2023-02-01"], dtype="datetime64[ns]")
    lats = np.array([10.1, 10.2], dtype=float)
    lons = np.array([10.1, 10.2, 10.3], dtype=float)

    if fill_nan:
        values = np.full((2, 2, 3), np.nan, dtype=float)
    else:
        values = np.arange(12, dtype=float).reshape(2, 2, 3)

    return xr.Dataset(
        {var_name: (("time", "latitude", "longitude"), values)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )


def _build_time_key(date_str):
    """Build a time key using the same method as get_var_values_mapping.

    This ensures test time_id_map keys match exactly how the function constructs
    lookup keys: np.datetime64(pd.to_datetime(f"{ts.year}-{ts.month}-{ts.day}"), "ns")
    """
    ts = pd.Timestamp(date_str)
    return np.datetime64(pd.to_datetime(f"{ts.year}-{ts.month}-{ts.day}"), "ns")


def _build_maps():
    # (2 lats * 3 lons) grid
    grid_id_map = {}
    gid = 1
    for lat in [10.1, 10.2]:
        for lon in [10.1, 10.2, 10.3]:
            grid_id_map[(round(lat, 4), round(lon, 4))] = gid
            gid += 1

    time_id_map = {
        _build_time_key("2023-01-01"): 1,
        _build_time_key("2023-02-01"): 2,
    }
    return grid_id_map, time_id_map


def test__q_respects_round_digits(monkeypatch):
    """Test that _q rounds floats according to ROUND_DIGITS setting."""
    # Test with ROUND_DIGITS=2
    monkeypatch.setattr(postdb, "ROUND_DIGITS", 2)
    assert postdb._q(10.12345) == 10.12
    assert postdb._q(10.126) == 10.13
    assert postdb._q(-5.6789) == -5.68
    assert postdb._q(0.001) == 0.0

    # Test with ROUND_DIGITS=4 (default)
    monkeypatch.setattr(postdb, "ROUND_DIGITS", 4)
    assert postdb._q(10.12345) == 10.1235
    assert postdb._q(10.123456789) == 10.1235
    assert postdb._q(-5.67891234) == -5.6789

    # Test type conversion for NumPy scalars
    assert postdb._q(np.float64(10.12345)) == 10.1235
    assert isinstance(postdb._q(np.float64(10.12345)), float)


def test__normalize_time_key():
    """Test that _normalize_time_key normalizes datetime64 to date-only midnight."""
    # Test with full datetime precision
    dt1 = np.datetime64("2023-01-15T14:30:45.123456789", "ns")
    normalized = postdb._normalize_time_key(dt1)
    expected = np.datetime64("2023-01-15T00:00:00", "ns")
    assert normalized == expected

    # Test with date-only input
    dt2 = np.datetime64("2023-02-28", "ns")
    normalized = postdb._normalize_time_key(dt2)
    expected = np.datetime64("2023-02-28T00:00:00", "ns")
    assert normalized == expected

    # Test with different time components
    dt3 = np.datetime64("2023-12-31T23:59:59", "ns")
    normalized = postdb._normalize_time_key(dt3)
    expected = np.datetime64("2023-12-31T00:00:00", "ns")
    assert normalized == expected

    # Test leap year date
    dt4 = np.datetime64("2024-02-29T12:00:00", "ns")
    normalized = postdb._normalize_time_key(dt4)
    expected = np.datetime64("2024-02-29T00:00:00", "ns")
    assert normalized == expected


def test_get_var_value(get_session):
    # sample data
    grid_point = postdb.GridPoint(latitude=10.0, longitude=20.0)
    time_point = postdb.TimePoint(year=2023, month=1, day=1)
    var_type = postdb.VarType(name="t2m", unit="K", description="2m temperature")
    var_value = postdb.VarValue(
        grid_id=1,
        time_id=1,
        var_id=1,
        value=300.0,
    )
    get_session.add(grid_point)
    get_session.add(time_point)
    get_session.add(var_type)
    get_session.add(var_value)
    get_session.commit()

    # test the function
    result = postdb.get_var_value(
        get_session,
        str(var_type.name),
        grid_point.latitude,
        grid_point.longitude,
        time_point.year,
        time_point.month,
        time_point.day,
    )
    assert result == var_value.value

    # None case
    result = postdb.get_var_value(
        get_session,
        "non_existing_var",
        grid_point.latitude,
        grid_point.longitude,
        time_point.year,
        time_point.month,
        time_point.day,
    )
    assert result is None


def test_get_var_value_nuts(
    get_engine_with_tables,
    get_dataset,
    get_session,
    tmp_path,
    get_nuts_def_data,
    get_varnuts_dataset,
):
    # create a sample NUTS shapefile
    nuts_path = tmp_path / "nuts_def.shp"
    gdf_nuts_data = get_nuts_def_data
    gdf_nuts_data.to_file(nuts_path, driver="ESRI Shapefile")

    # insert NUTS definitions
    postdb.insert_nuts_def(get_engine_with_tables, nuts_path)

    # sample data
    time_point = postdb.TimePoint(year=2023, month=1, day=1)
    var_type = postdb.VarType(name="t2m", unit="K", description="2m temperature")
    var_value = postdb.VarValueNuts(
        nuts_id="DE11",
        time_id=1,
        var_id=1,
        value=300.0,
    )
    get_session.add(time_point)
    get_session.add(var_type)
    get_session.add(var_value)
    get_session.commit()

    # test the function
    result = postdb.get_var_value_nuts(
        get_session,
        str(var_type.name),
        "DE11",
        time_point.year,
        time_point.month,
        time_point.day,
    )
    assert result == var_value.value

    # None case
    result = postdb.get_var_value_nuts(
        get_session,
        "non_existing_var",
        "DE11",
        time_point.year,
        time_point.month,
        time_point.day,
    )
    assert result is None


def test_get_time_points(get_session, get_dataset):
    # insert time points
    postdb.insert_time_points(get_session, [(get_dataset.time.values, False)])

    # test the function
    result = postdb.get_time_points(
        get_session, start_time_point=(2023, 1), end_time_point=None
    )
    assert len(result) == 1
    assert result[0].year == 2023
    assert result[0].month == 1
    assert result[0].day == 1

    result = postdb.get_time_points(
        get_session, start_time_point=(2023, 1), end_time_point=(2024, 1)
    )
    assert len(result) == 2
    assert result[0].year == 2023
    assert result[0].month == 1
    assert result[1].year == 2024
    assert result[1].month == 1

    # test with no time points
    result = postdb.get_time_points(get_session, start_time_point=(2025, 1))
    assert len(result) == 0
    get_session.commit()


def test_get_grid_points(get_session, get_dataset):
    # insert grid points
    postdb.insert_grid_points(
        get_session, get_dataset.latitude.values, get_dataset.longitude.values
    )

    # test the function
    result = postdb.get_grid_points(get_session, area=None)
    assert len(result) == 6  # 2 latitudes * 3 longitudes
    assert math.isclose(result[0].latitude, 10.1, abs_tol=1e-5)
    assert math.isclose(result[0].longitude, 10.1, abs_tol=1e-5)

    result = postdb.get_grid_points(
        get_session, area=(11.0, 10.0, 10.0, 12.0)
    )  # [N, W, S, E]
    assert len(result) == 6
    assert math.isclose(result[0].latitude, 10.1, abs_tol=1e-5)
    assert math.isclose(result[0].longitude, 10.1, abs_tol=1e-5)

    # no grid points case
    result = postdb.get_grid_points(get_session, area=(20.0, 20.0, 20.0, 20.0))
    assert len(result) == 0


def test_get_var_types(get_session):
    # insert var types
    var_type_data = [
        {
            "name": "t2m",
            "unit": "K",
            "description": "2m temperature",
        }
    ]
    postdb.insert_var_types(get_session, var_type_data)

    # test the function
    result = postdb.get_var_types(get_session, var_names=None)
    assert len(result) == 1
    assert result[0].name == "t2m"
    assert result[0].unit == "K"
    assert result[0].description == "2m temperature"

    result = postdb.get_var_types(get_session, var_names=["t2m"])
    assert len(result) == 1
    assert result[0].name == "t2m"

    # test with no var types
    result = postdb.get_var_types(get_session, var_names=["non_existing_var"])
    assert len(result) == 0


def test_sort_grid_points_get_ids(get_session, get_dataset, insert_data):
    grid_points = get_session.query(postdb.GridPoint).all()
    grid_ids, latitudes, longitudes = postdb.sort_grid_points_get_ids(grid_points)
    assert len(grid_ids) == 6  # 2 latitudes * 3 longitudes
    assert all(
        math.isclose(lat, ref, abs_tol=1e-5)
        for lat, ref in zip(latitudes, [10.1000, 10.2000])
    )
    assert all(
        math.isclose(lon, ref, abs_tol=1e-5)
        for lon, ref in zip(longitudes, [10.1000, 10.2000, 10.3000])
    )
    # check if the ids are correct
    assert grid_ids[1] == (0, 0)
    assert grid_ids[4] == (1, 0)


def test_get_var_values_cartesian(insert_data):
    # test the function
    # normal case with full map
    ds_result = postdb.get_var_values_cartesian(
        insert_data,
        time_point=(2023, 1),
        var_name="t2m",
    )
    assert len(ds_result["latitude, longitude, var_value"]) == 6
    values = ds_result["latitude, longitude, var_value"][0]
    assert math.isclose(values[0], 10.1000, abs_tol=1e-5)
    assert math.isclose(values[1], 10.1000, abs_tol=1e-5)

    # with default var and full map
    ds_result = postdb.get_var_values_cartesian(
        insert_data,
        time_point=(2023, 1),
        var_name=None,
    )
    assert len(ds_result["latitude, longitude, var_value"]) == 6
    values = ds_result["latitude, longitude, var_value"][0]
    assert math.isclose(values[0], 10.1, abs_tol=1e-5)
    assert math.isclose(values[1], 10.1, abs_tol=1e-5)

    # with area
    ds_result = postdb.get_var_values_cartesian(
        insert_data,
        time_point=(2023, 1),
        area=(10.2, 10.0, 10.0, 10.2),  # [N, W, S, E]
        var_name="t2m",
    )
    assert len(ds_result["latitude, longitude, var_value"]) == 4
    values = ds_result["latitude, longitude, var_value"][0]
    assert math.isclose(values[0], 10.1, abs_tol=1e-5)
    assert math.isclose(values[1], 10.1, abs_tol=1e-5)

    # test HTTP exceptions
    # test for missing time point
    with pytest.raises(HTTPException):
        postdb.get_var_values_cartesian(
            insert_data,
            time_point=(2020, 1),
            var_name=None,
        )
    # test for missing grid points in area
    with pytest.raises(HTTPException):
        postdb.get_var_values_cartesian(
            insert_data,
            time_point=(2023, 1),
            area=(20.0, 18.0, 18.0, 20.0),  # [N, W, S, E]
            var_name=None,
        )
    # test for missing variable name
    with pytest.raises(HTTPException):
        postdb.get_var_values_cartesian(
            insert_data,
            time_point=(2023, 1),
            var_name="non_existing_var",
        )


def test_get_var_values_cartesian_download(get_dataset, insert_data, tmp_path):
    # test the function
    netcdf_filename = tmp_path / "test_var_values.nc"
    postdb.get_var_values_cartesian_for_download(
        insert_data,
        start_time_point=(2023, 1),
        end_time_point=None,
        area=None,
        var_names=None,
        netcdf_file=netcdf_filename,
    )
    assert netcdf_filename.exists()
    ds_result = xr.open_dataset(netcdf_filename)
    assert len(ds_result.latitude) == 2
    assert len(ds_result.longitude) == 3
    assert len(ds_result.time) == 1
    assert ds_result.t2m.shape == (1, 2, 3)
    assert math.isclose(
        ds_result.t2m[0, 0, 0], get_dataset.t2m[0, 0, 0], rel_tol=1e-6
    )  # stored as float4
    assert math.isclose(
        ds_result.t2m[0, 1, 1], get_dataset.t2m[0, 1, 1], rel_tol=1e-6
    )  # stored as float4
    # remove the file after test
    netcdf_filename.unlink()
    # with end point
    postdb.get_var_values_cartesian_for_download(
        insert_data,
        start_time_point=(2023, 1),
        end_time_point=(2024, 1),
        area=None,
        var_names=None,
        netcdf_file=netcdf_filename,
    )
    assert netcdf_filename.exists()
    ds_result = xr.open_dataset(netcdf_filename)
    assert len(ds_result.latitude) == 2
    assert len(ds_result.longitude) == 3
    assert len(ds_result.time) == 2
    assert ds_result.t2m.shape == (2, 2, 3)
    # remove the file after test
    netcdf_filename.unlink()

    # with area
    postdb.get_var_values_cartesian_for_download(
        insert_data,
        start_time_point=(2023, 1),
        end_time_point=None,
        area=(10.2, 10.0, 10.0, 10.2),  # [N, W, S, E]
        var_names=None,
        netcdf_file=netcdf_filename,
    )
    assert netcdf_filename.exists()
    ds_result = xr.open_dataset(netcdf_filename)
    assert len(ds_result.latitude) == 2
    assert len(ds_result.longitude) == 2
    assert len(ds_result.time) == 1
    assert ds_result.t2m.shape == (1, 2, 2)
    # remove the file after test
    netcdf_filename.unlink()

    # with var names
    postdb.get_var_values_cartesian_for_download(
        insert_data,
        start_time_point=(2023, 1),
        end_time_point=None,
        area=None,
        var_names=["t2m"],
        netcdf_file=netcdf_filename,
    )
    assert netcdf_filename.exists()
    ds_result = xr.open_dataset(netcdf_filename)
    assert len(ds_result.latitude) == 2
    assert len(ds_result.longitude) == 3
    assert len(ds_result.time) == 1
    assert ds_result.t2m.shape == (1, 2, 3)
    # remove the file after test
    netcdf_filename.unlink()

    # none cases
    # no time points
    with pytest.raises(HTTPException):
        postdb.get_var_values_cartesian_for_download(
            insert_data,
            start_time_point=(2025, 1),
            end_time_point=None,
            area=None,
            var_names=None,
        )

    # no grid points
    with pytest.raises(HTTPException):
        postdb.get_var_values_cartesian_for_download(
            insert_data,
            start_time_point=(2023, 1),
            end_time_point=None,
            area=(20.0, 20.0, 20.0, 20.0),  # [N, W, S, E]
            var_names=None,
        )

    # no var types
    with pytest.raises(HTTPException):
        postdb.get_var_values_cartesian_for_download(
            insert_data,
            start_time_point=(2023, 1),
            end_time_point=None,
            area=None,
            var_names=["non_existing_var"],
        )


def test_get_nuts_regions(
    get_engine_with_tables, get_session, tmp_path, get_nuts_def_data
):
    # create a sample NUTS shapefile
    nuts_path = tmp_path / "nuts_def.shp"
    gdf_nuts_data = get_nuts_def_data
    gdf_nuts_data.to_file(nuts_path, driver="ESRI Shapefile")

    # insert NUTS definitions
    postdb.insert_nuts_def(get_engine_with_tables, nuts_path)

    # test the function
    # normal case
    result = postdb.get_nuts_regions(get_engine_with_tables)
    assert len(result) == 3
    assert result.loc[0, "nuts_id"] == "DE11"  # result is a geodataframe
    assert result.loc[0, "name_latn"] == "Test NUTS"
    assert result.loc[1, "nuts_id"] == "DE22"
    assert result.loc[1, "name_latn"] == "Test NUTS2"
    assert result.loc[2, "nuts_id"] == "DE501"
    assert result.loc[2, "name_latn"] == "Test NUTS3"


def test_get_nuts_regions_geojson(
    get_engine_with_tables, get_session, tmp_path, get_nuts_def_data
):
    nuts_path = tmp_path / "nuts_def.shp"
    gdf_nuts_data = get_nuts_def_data
    gdf_nuts_data.to_file(nuts_path, driver="ESRI Shapefile")
    postdb.insert_nuts_def(get_engine_with_tables, nuts_path)

    geojson = postdb.get_nuts_regions_geojson(get_engine_with_tables)
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 3
    nuts_ids = {feature["properties"]["nuts_id"] for feature in geojson["features"]}
    assert nuts_ids == {"DE11", "DE22", "DE501"}  # IDs copied from above

    filtered_geojson = postdb.get_nuts_regions_geojson(
        get_engine_with_tables, grid_resolution="NUTS2"
    )
    assert len(filtered_geojson["features"]) == 2
    assert filtered_geojson["features"][0]["properties"]["levl_code"] == 2
    filtered_geojson = postdb.get_nuts_regions_geojson(
        get_engine_with_tables, grid_resolution="NUTS3"
    )
    # DE501 case.
    assert len(filtered_geojson["features"]) == 1
    assert filtered_geojson["features"][0]["properties"]["levl_code"] == 3
    # test for invalid NUTS resolution
    with pytest.raises(HTTPException):
        postdb.get_nuts_regions_geojson(get_engine_with_tables, grid_resolution="NUTS4")
    # test for valid nuts resolution that does not exist in the db
    with pytest.raises(HTTPException):
        postdb.get_nuts_regions_geojson(get_engine_with_tables, grid_resolution="NUTS1")
    # test for empty nuts regions
    # remove nuts definitions from the db
    get_session.execute(text("TRUNCATE TABLE nuts_def RESTART IDENTITY CASCADE"))
    get_session.commit()
    with pytest.raises(HTTPException):
        postdb.get_nuts_regions_geojson(get_engine_with_tables)


def test_get_grid_ids_in_nuts(get_engine_with_tables, get_session):
    nuts_regions = gpd.GeoDataFrame(
        {
            "nuts_id": ["DE11", "DE22"],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        },
        crs="EPSG:4326",
    )
    latitudes = np.array([0.5, 1.0, 1.5])
    longitudes = np.array([0.5, 1.0, 1.5])
    postdb.insert_grid_points(get_session, latitudes, longitudes)

    # test the function
    # normal case
    grid_ids = postdb.get_grid_ids_in_nuts(get_engine_with_tables, nuts_regions)
    assert len(grid_ids) == 6
    assert grid_ids[0] == 1
    assert grid_ids[1] == 2

    # none cases
    grid_ids = postdb.get_grid_ids_in_nuts(
        get_engine_with_tables, nuts_regions=gpd.GeoDataFrame(geometry=[])
    )
    assert len(grid_ids) == 0


def test_filter_nuts_ids_for_resolution():
    nuts_ids = ["DE", "DE11", "DE1", "TR2", "TR22"]
    filtered_nuts_ids = postdb.filter_nuts_ids_for_resolution(nuts_ids, "NUTS0")
    assert len(filtered_nuts_ids) == 1
    assert "DE" in filtered_nuts_ids
    filtered_nuts_ids = postdb.filter_nuts_ids_for_resolution(nuts_ids, "NUTS2")
    assert len(filtered_nuts_ids) == 2
    assert "DE11" in filtered_nuts_ids
    assert "TR22" in filtered_nuts_ids
    # test with no matching nuts ids
    filtered_nuts_ids = postdb.filter_nuts_ids_for_resolution(nuts_ids, "NUTS3")
    assert len(filtered_nuts_ids) == 0

    nuts_3_ids = ["DE501"]
    filtered_nuts_3_ids = postdb.filter_nuts_ids_for_resolution(nuts_3_ids, "NUTS3")
    assert len(filtered_nuts_3_ids) == 1


def test_get_var_values_nuts(
    get_engine_with_tables,
    get_session,
    tmp_path,
    get_nuts_def_data,
    get_varnuts_dataset,
    seed_base_data,
):
    # create a sample NUTS shapefile
    nuts_path = tmp_path / "nuts_def.shp"
    gdf_nuts_data = get_nuts_def_data
    gdf_nuts_data.to_file(nuts_path, driver="ESRI Shapefile")

    # insert NUTS definitions
    postdb.insert_nuts_def(get_engine_with_tables, nuts_path)

    # get the id maps
    _, time_id_map, var_id_map = postdb.get_id_maps(get_session)

    # insert var value
    postdb.insert_var_value_nuts(
        get_engine_with_tables,
        get_varnuts_dataset,
        var_name="t2m",
        time_id_map=time_id_map,
        var_id_map=var_id_map,
    )

    # test the function
    # normal case
    result_dict = postdb.get_var_values_nuts(
        get_session,
        time_point=(2023, 1),
        var_name="t2m",
    )
    assert len(result_dict) == 2
    assert result_dict["DE11"] == get_varnuts_dataset["t2m"][0, 0]
    assert result_dict["DE22"] == get_varnuts_dataset["t2m"][1, 0]
    # a level whose regions are not first in the table keeps its own values
    result_dict = postdb.get_var_values_nuts(
        get_session, time_point=(2023, 1), var_name="t2m", grid_resolution="NUTS3"
    )
    assert result_dict == {"DE501": get_varnuts_dataset["t2m"][2, 0]}
    # none cases
    # no time points
    with pytest.raises(HTTPException):
        postdb.get_var_values_nuts(
            get_session,
            time_point=(2025, 1),
            var_name=None,
        )
    # no var types
    with pytest.raises(HTTPException):
        postdb.get_var_values_nuts(
            get_session,
            time_point=(2023, 1),
            var_name="non_existing_var",
        )
    # no var values
    get_session.execute(text("TRUNCATE TABLE var_value_nuts RESTART IDENTITY CASCADE"))
    get_session.commit()
    with pytest.raises(HTTPException):
        postdb.get_var_values_nuts(
            get_session,
            time_point=(2023, 1),
            var_name=None,
        )
    # no values for selected resolution
    get_session.execute(text("TRUNCATE TABLE var_value_nuts RESTART IDENTITY CASCADE"))
    get_session.commit()
    with pytest.raises(HTTPException):
        postdb.get_var_values_nuts(
            get_session,
            time_point=(2023, 1),
            var_name="t2m",
            grid_resolution="NUTS0",
        )


def test_insert_var_value_nuts_no_var(
    get_engine_with_tables,
    get_session,
    get_varnuts_dataset,
    tmp_path,
    get_nuts_def_data,
):
    # insert the time points
    postdb.insert_time_points(get_session, [(get_varnuts_dataset.time.values, False)])

    # get the id maps
    _, time_id_map, var_id_map = postdb.get_id_maps(get_session)

    # error due to no t2m var type
    with pytest.raises(ValueError):
        postdb.insert_var_value_nuts(
            get_engine_with_tables,
            get_varnuts_dataset,
            var_name="t2m",
            time_id_map=time_id_map,
            var_id_map=var_id_map,
        )


def test_insert_var_value_nuts(
    get_engine_with_tables,
    get_session,
    get_varnuts_dataset,
    get_nuts_def_data,
    tmp_path,
    seed_base_data,
):
    # create a sample NUTS shapefile
    nuts_path = tmp_path / "nuts_def.shp"
    gdf_nuts_data = get_nuts_def_data
    gdf_nuts_data.to_file(nuts_path, driver="ESRI Shapefile")

    # insert NUTS definitions
    postdb.insert_nuts_def(get_engine_with_tables, nuts_path)

    # get the id maps
    _, time_id_map, var_id_map = postdb.get_id_maps(get_session)

    # insert var value
    postdb.insert_var_value_nuts(
        get_engine_with_tables,
        get_varnuts_dataset,
        var_name="t2m",
        time_id_map=time_id_map,
        var_id_map=var_id_map,
    )

    # check if the data is inserted correctly
    result = get_session.query(postdb.VarValueNuts).all()
    assert len(result) == 6
    assert result[0].nuts_id == "DE11"
    assert result[0].time_id == 1
    assert result[0].var_id == 1
    assert math.isclose(result[0].value, get_varnuts_dataset.t2m[0, 0], abs_tol=1e-5)


def test_insert_var_value_nuts_rejects_extra_dims(
    get_engine_with_tables,
    get_session,
    get_varnuts_dataset,
    get_nuts_def_data,
    tmp_path,
    seed_base_data,
):
    nuts_path = tmp_path / "nuts_def.shp"
    gdf_nuts_data = get_nuts_def_data
    gdf_nuts_data.to_file(nuts_path, driver="ESRI Shapefile")
    postdb.insert_nuts_def(get_engine_with_tables, nuts_path)

    # get the id maps
    _, time_id_map, var_id_map = postdb.get_id_maps(get_session)

    ds_extra = get_varnuts_dataset.expand_dims(lat=[0, 1])
    with pytest.raises(ValueError, match="must be 2D over \\('time', 'NUTS_ID'\\)"):
        postdb.insert_var_value_nuts(
            get_engine_with_tables,
            ds_extra,
            var_name="t2m",
            time_id_map=time_id_map,
            var_id_map=var_id_map,
        )


# --- variables stored per resolution (aggregation "mean") ---


def _fine_grid_ds(var_name="adults", n_time=2, lon_lat_time=False):
    """5x5 grid on 0.1 degree spacing, non-periodic in longitude."""
    lats = np.round(np.arange(0.4, -0.05, -0.1), 1)
    lons = np.round(np.arange(0.0, 0.45, 0.1), 1)
    times = [np.datetime64(f"2023-0{i + 1}-01", "ns") for i in range(n_time)]
    values = np.arange(n_time * 25, dtype=float).reshape(n_time, 5, 5) + 1.0
    da = xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    if lon_lat_time:
        da = da.transpose("longitude", "latitude", "time")
    return xr.Dataset({var_name: da})


def test__window_matrix_odd_factor():
    m = postdb._window_matrix(9, np.array([1, 4, 7]), 3, periodic=False)
    assert m.shape == (3, 9)
    np.testing.assert_array_equal(m[1], [0, 0, 0, 1, 1, 1, 0, 0, 0])
    # every fine cell is counted exactly once
    np.testing.assert_array_equal(m.sum(axis=0), np.ones(9))


def test__window_matrix_even_factor_half_weights():
    m = postdb._window_matrix(8, np.array([0, 2, 4, 6]), 2, periodic=True)
    np.testing.assert_array_equal(m[1], [0, 0.5, 1, 0.5, 0, 0, 0, 0])
    # periodic: the first window wraps around to the last cell
    np.testing.assert_array_equal(m[0], [1, 0.5, 0, 0, 0, 0, 0, 0.5])
    np.testing.assert_array_equal(m.sum(axis=0), np.ones(8))
    # non-periodic: the window is cut off at the ends
    m = postdb._window_matrix(8, np.array([0]), 2, periodic=False)
    np.testing.assert_array_equal(m[0], [1, 0.5, 0, 0, 0, 0, 0, 0])


def test__block_mean_plan():
    ds = _fine_grid_ds()
    lats, lons = ds.latitude.values, ds.longitude.values
    native = postdb._block_mean_plan(lats, lons, 0.1)
    assert native["factor"] == 1
    np.testing.assert_array_equal(native["latitude"], lats)
    plan = postdb._block_mean_plan(lats, lons, 0.2)
    assert plan["factor"] == 2
    np.testing.assert_allclose(plan["latitude"], [0.4, 0.2, 0.0])
    np.testing.assert_allclose(plan["longitude"], [0.0, 0.2, 0.4])
    # not a multiple of the grid step, or finer than the data
    assert postdb._block_mean_plan(lats, lons, 0.25) is None
    assert postdb._block_mean_plan(lats, lons, 0.05) is None
    with pytest.raises(ValueError, match="not regularly spaced"):
        postdb._block_mean_plan(np.array([0.0, 0.1, 0.3]), lons, 0.2)


def test_block_mean_values_and_nan():
    ds = _fine_grid_ds(n_time=1)
    lats, lons = ds.latitude.values, ds.longitude.values
    values = ds["adults"].values.copy()
    values[0, 2, 2] = np.nan  # centre cell
    plan = postdb._block_mean_plan(lats, lons, 0.2)
    out = postdb.block_mean(values, plan)
    assert out.shape == (1, 3, 3)
    # centre coarse cell (0.2, 0.2): rows/cols 1..3, half weights on the edges
    w1 = np.array([0.5, 1.0, 0.5])
    area = np.cos(np.deg2rad(lats[1:4]))
    weights = (w1 * area)[:, None] * w1[None, :]
    block = values[0, 1:4, 1:4]
    valid = np.isfinite(block)
    expected = (np.where(valid, block, 0) * weights).sum() / (weights * valid).sum()
    np.testing.assert_allclose(out[0, 1, 1], expected)
    # a constant field stays constant
    const = np.full_like(values, 7.0)
    np.testing.assert_allclose(postdb.block_mean(const, plan), 7.0)
    # all-NaN input gives NaN
    assert np.isnan(postdb.block_mean(np.full_like(values, np.nan), plan)).all()
    with pytest.raises(ValueError, match="native resolution"):
        postdb.block_mean(values, postdb._block_mean_plan(lats, lons, 0.1))


def test__to_time_lat_lon():
    ds = _fine_grid_ds(lon_lat_time=True)
    assert ds["adults"].dims == ("longitude", "latitude", "time")
    out = postdb._to_time_lat_lon(ds, "adults")
    assert out["adults"].dims == ("time", "latitude", "longitude")
    with pytest.raises(ValueError, match="not found"):
        postdb._to_time_lat_lon(ds, "missing")
    with pytest.raises(ValueError, match="must have dims"):
        postdb._to_time_lat_lon(ds.isel(time=0, drop=True), "adults")


def test_insert_var_values_transposes_lon_lat_time(monkeypatch):
    ds = _fine_grid_ds(var_name="t2m", lon_lat_time=True)
    captured = {}

    def _fake_copy(engine, values, grid_ids, time_ids, var_id, resolution_id=None):
        captured["shape"] = values.shape
        captured["grid"] = grid_ids.shape
        return 0

    monkeypatch.setattr(postdb, "copy_var_values", _fake_copy)
    monkeypatch.setattr(postdb, "_check_no_coverage", lambda *a: None)
    monkeypatch.setattr(postdb, "insert_var_coverage", lambda *a, **k: 0)
    monkeypatch.setattr(postdb, "time_id_array", lambda times, m: np.arange(len(times)))
    postdb.insert_var_values(object(), ds, "t2m", {}, {}, {"t2m": 1})
    n_time = ds.sizes["time"]
    assert captured["shape"] == (
        min(n_time, postdb.VAR_TIME_CHUNK),
        ds.sizes["latitude"],
        ds.sizes["longitude"],
    )
    assert captured["grid"] == (ds.sizes["latitude"], ds.sizes["longitude"])


def test_pgcopy_binary_layout():
    data = postdb._pgcopy_binary(
        [(">i4", np.array([7, 8])), (">i2", None), (">f4", np.array([1.5, -2.0]))]
    )
    header = b"PGCOPY\n\xff\r\n\x00" + b"\x00" * 8
    assert data.startswith(header)
    assert data.endswith(b"\xff\xff")
    body = data[len(header) : -2]
    # per row: field count, (length, int4), (length -1 for NULL), (length, float4)
    row = np.dtype(
        [
            ("n", ">i2"),
            ("l0", ">i4"),
            ("v0", ">i4"),
            ("l1", ">i4"),
            ("l2", ">i4"),
            ("v2", ">f4"),
        ]
    )
    rows = np.frombuffer(body, dtype=row)
    assert rows["n"].tolist() == [3, 3]
    assert rows["v0"].tolist() == [7, 8]
    assert rows["l1"].tolist() == [-1, -1]
    assert rows["l2"].tolist() == [4, 4]
    assert rows["v2"].tolist() == [1.5, -2.0]
    with pytest.raises(ValueError, match="same length"):
        postdb._pgcopy_binary([(">i4", np.array([1])), (">i4", np.array([1, 2]))])


def test_grid_id_array(capsys):
    grid_id_map = {(10.1, 20.1): 1, (10.1, 20.2): 2, (10.2, 20.1): 3}
    ids = postdb.grid_id_array(
        np.array([10.1, 10.2]), np.array([20.1, 20.2000001]), grid_id_map
    )
    assert ids.tolist() == [[1, 2], [3, -1]]
    assert "1 grid points are not in the database" in capsys.readouterr().out


def test_time_id_array():
    time_id_map = {postdb._normalize_time_key(np.datetime64("2023-01-01")): 5}
    ids = postdb.time_id_array(
        np.array(["2023-01-01T12:00"], dtype="datetime64[ns]"), time_id_map
    )
    assert ids.tolist() == [5]
    with pytest.raises(ValueError, match="not in the database"):
        postdb.time_id_array(np.array(["2023-02-01"], dtype="datetime64[ns]"), {})


def test_copy_var_values(get_session, get_engine_with_tables):
    engine = get_engine_with_tables
    postdb.insert_var_types(get_session, [{"name": "t2m", "unit": "K"}])
    postdb.insert_grid_points(
        get_session, np.array([10.1, 10.2]), np.array([20.1, 20.2])
    )
    postdb.insert_time_points(
        get_session,
        [(np.array(["2023-01-01", "2023-02-01"], dtype="datetime64[ns]"), False)],
    )
    postdb.insert_resolution_groups(get_session, resolutions=np.array([0.1]))
    grid_id_map, time_id_map, var_id_map = postdb.get_id_maps(get_session)
    grid_ids = postdb.grid_id_array(
        np.array([10.1, 10.2]), np.array([20.1, 20.2]), grid_id_map
    )
    grid_ids[1, 1] = -1  # unknown grid point: skipped
    values = np.arange(8, dtype=float).reshape(2, 2, 2) + 0.25
    values[0, 0, 1] = np.nan  # no value: skipped
    time_ids = postdb.time_id_array(
        np.array(["2023-01-01", "2023-02-01"], dtype="datetime64[ns]"), time_id_map
    )
    n = postdb.copy_var_values(engine, values, grid_ids, time_ids, var_id_map["t2m"])
    assert n == 5
    # the same values with a resolution group are a different row
    n_res = postdb.copy_var_values(
        engine, values, grid_ids, time_ids, var_id_map["t2m"], resolution_id=1
    )
    assert n_res == 5
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT grid_id, time_id, resolution_id, value FROM var_value "
                "ORDER BY resolution_id NULLS FIRST, time_id, grid_id"
            )
        ).all()
    assert len(rows) == 10
    assert rows[0] == (grid_ids[0, 0], time_ids[0], None, 0.25)
    assert rows[5][2] == 1
    assert {r[3] for r in rows[:5]} == {0.25, 2.25, 4.25, 5.25, 6.25}
    with pytest.raises(ValueError, match="do not match the grid"):
        postdb.copy_var_values(engine, values[:, :1], grid_ids, time_ids, 1)
    with pytest.raises(ValueError, match="One time point ID"):
        postdb.copy_var_values(engine, values, grid_ids, time_ids[:1], 1)
    get_session.execute(text("TRUNCATE TABLE var_value"))
    get_session.commit()


def test_bulk_load_constraints_drop_and_add(get_session, get_engine_with_tables):
    engine = get_engine_with_tables

    def constraints():
        with engine.connect() as conn:
            return {
                row[0]
                for row in conn.execute(
                    text(
                        "SELECT conname FROM pg_constraint WHERE conrelid IN "
                        "('var_value'::regclass, 'var_coverage'::regclass)"
                    )
                )
            }

    expected = {
        "uq_time_grid_var_res",
        "fk_grid_id",
        "fk_time_id",
        "fk_var_id",
        "fk_resolution_id",
        "uq_coverage_var_res_grid",
        "fk_coverage_var_id",
        "fk_coverage_resolution_id",
        "fk_coverage_grid_id",
    }
    assert expected <= constraints()
    get_session.close()  # dropping needs exclusive locks on the referenced tables
    postdb.drop_bulk_load_constraints(engine)
    postdb.drop_bulk_load_constraints(engine)  # idempotent
    assert not (expected & constraints())
    # rows with unknown IDs are accepted without the constraints ...
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO var_value (grid_id, time_id, var_id, value) "
                "VALUES (999, 999, 99, 1.0)"
            )
        )
    # ... and rejected when they are restored
    with pytest.raises(Exception, match="fk_"):
        postdb.add_bulk_load_constraints(engine)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM var_value"))
    postdb.add_bulk_load_constraints(engine)
    postdb.add_bulk_load_constraints(engine)  # idempotent
    assert expected <= constraints()
    with engine.connect() as conn:
        definition = conn.execute(
            text(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname = 'uq_time_grid_var_res'"
            )
        ).scalar()
    assert "NULLS NOT DISTINCT" in definition


def _seed_mean_var(session, engine, ds, mask=None):
    """Insert grid/time points, resolution groups and `adults` per resolution."""
    postdb.insert_var_types(
        session,
        [
            {
                "name": "adults",
                "unit": "mosquitoes per 625 km²",
                "description": "adult mosquitoes",
                "aggregation": "mean",
            }
        ],
    )
    postdb.insert_grid_points(session, ds.latitude.values, ds.longitude.values)
    postdb.insert_time_points(session, [(ds.time.values, False)])
    postdb.insert_resolution_groups(session, resolutions=np.array([0.1, 0.2, 0.25]))
    grid_id_map, time_id_map, var_id_map = postdb.get_id_maps(session)
    resolution_id_map = postdb.get_resolution_id_map(session)
    total = postdb.insert_var_values_by_resolution(
        engine,
        ds,
        "adults",
        grid_id_map,
        time_id_map,
        var_id_map,
        resolution_id_map,
        mask=mask,
    )
    postdb.assign_grid_resolution_group_to_grid_point(session)
    return total, resolution_id_map


def test_insert_var_values_by_resolution_and_queries(
    get_engine_with_tables, get_session, tmp_path
):
    ds = _fine_grid_ds(lon_lat_time=True)
    mask = xr.ones_like(ds["adults"], dtype=bool)
    mask[{"longitude": 4, "latitude": 0}] = False  # (lat 0.4, lon 0.4)
    total, resolution_id_map = _seed_mean_var(
        get_session, get_engine_with_tables, ds, mask=mask
    )
    # 0.1: 25 - 1 masked per time step, 0.2: 3x3, 0.25 skipped
    assert total == 2 * (24 + 9)
    counts = dict(
        get_session.execute(
            text("SELECT resolution_id, COUNT(*) FROM var_value GROUP BY resolution_id")
        ).all()
    )
    assert counts == {resolution_id_map[0.1]: 48, resolution_id_map[0.2]: 18}

    values = ds["adults"].transpose("time", "latitude", "longitude").values
    values = np.where(
        mask.transpose("time", "latitude", "longitude").values, values, np.nan
    )
    plan = postdb._block_mean_plan(ds.latitude.values, ds.longitude.values, 0.2)
    expected = postdb.block_mean(values, plan)

    # cartesian query at 0.2 returns the coarse means
    result = postdb.get_var_values_cartesian(
        get_session, time_point=(2023, 1), grid_resolution=0.2, var_name="adults"
    )["latitude, longitude, var_value"]
    assert len(result) == 9
    lookup = {(round(float(la), 1), round(float(lo), 1)): v for la, lo, v in result}
    np.testing.assert_allclose(lookup[(0.2, 0.2)], expected[0, 1, 1])
    np.testing.assert_allclose(lookup[(0.4, 0.4)], expected[0, 0, 2])
    # at 0.1 it returns the native values, without the masked cell
    result = postdb.get_var_values_cartesian(
        get_session, time_point=(2023, 1), grid_resolution=0.1, var_name="adults"
    )["latitude, longitude, var_value"]
    assert len(result) == 24
    lookup = {(round(float(la), 1), round(float(lo), 1)): v for la, lo, v in result}
    np.testing.assert_allclose(lookup[(0.2, 0.2)], values[0, 2, 2])

    # single values: native by default, coarse on request
    native = postdb.get_var_value(get_session, "adults", 0.2, 0.2, 2023, 2, 1)
    np.testing.assert_allclose(native, values[1, 2, 2])
    coarse = postdb.get_var_value(
        get_session, "adults", 0.2, 0.2, 2023, 2, 1, resolution=0.2
    )
    np.testing.assert_allclose(coarse, expected[1, 1, 1])

    # download exports the native resolution only
    out_file = tmp_path / "download.nc"
    postdb.get_var_values_cartesian_for_download(
        get_session,
        start_time_point=(2023, 1),
        end_time_point=(2023, 2),
        var_names=["adults"],
        netcdf_file=str(out_file),
    )
    with xr.open_dataset(out_file) as downloaded:
        got = downloaded["adults"].sortby("latitude", ascending=False).values
    np.testing.assert_allclose(got, values)


def test_insert_var_values_by_resolution_errors(get_engine_with_tables, get_session):
    ds = _fine_grid_ds()
    with pytest.raises(ValueError, match="not found in var_type"):
        postdb.insert_var_values_by_resolution(
            get_engine_with_tables, ds, "adults", {}, {}, {}, {0.1: 1}
        )
    # no resolution group at the data's grid step
    with pytest.raises(ValueError, match="No resolution group matches"):
        postdb.insert_var_values_by_resolution(
            get_engine_with_tables, ds, "adults", {}, {}, {"adults": 1}, {0.2: 1}
        )
    # mask on another grid
    mask = xr.ones_like(ds["adults"], dtype=bool).isel(latitude=slice(0, 4))
    with pytest.raises(ValueError, match="different grid"):
        postdb.insert_var_values_by_resolution(
            get_engine_with_tables,
            ds,
            "adults",
            {},
            {},
            {"adults": 1},
            {0.1: 1},
            mask=mask,
        )
    # mask without the data's time points
    mask = xr.ones_like(ds["adults"], dtype=bool).assign_coords(
        time=[np.datetime64("2020-01-01", "ns"), np.datetime64("2020-02-01", "ns")]
    )
    with pytest.raises(ValueError, match="does not cover"):
        postdb.insert_var_values_by_resolution(
            get_engine_with_tables,
            ds,
            "adults",
            {},
            {},
            {"adults": 1},
            {0.1: 1},
            mask=mask,
        )


def test_var_value_unique_with_null_resolution(get_session):
    postdb.insert_var_types(get_session, [{"name": "t2m", "unit": "K"}])
    postdb.insert_grid_points(get_session, np.array([0.0]), np.array([0.0]))
    postdb.insert_time_points(
        get_session, [(np.array([np.datetime64("2023-01-01", "ns")]), False)]
    )
    row = {"grid_id": 1, "time_id": 1, "var_id": 1, "value": 1.0}
    postdb.add_data_list_bulk(get_session, [row], postdb.VarValue)
    # NULLS NOT DISTINCT: a second row without resolution is a duplicate
    with pytest.raises(RuntimeError, match="uq_time_grid_var_res"):
        postdb.add_data_list_bulk(get_session, [row], postdb.VarValue)
    assert (
        get_session.query(postdb.VarType).first().aggregation == "subsample"
    )  # server default


def test_upgrade_schema_from_old_tables(get_engine_with_tables):
    engine = get_engine_with_tables
    # turn the tables back into the old schema
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE var_value DROP CONSTRAINT uq_time_grid_var_res"))
        conn.execute(text("ALTER TABLE var_value DROP CONSTRAINT fk_resolution_id"))
        conn.execute(text("ALTER TABLE var_value DROP COLUMN resolution_id"))
        conn.execute(text("ALTER TABLE var_type DROP COLUMN aggregation"))
        # surrogate key and duplicate foreign keys of older versions
        conn.execute(text("ALTER TABLE var_value ADD COLUMN id BIGSERIAL PRIMARY KEY"))
        conn.execute(
            text(
                "ALTER TABLE var_value ADD CONSTRAINT var_value_grid_id_fkey "
                "FOREIGN KEY (grid_id) REFERENCES grid_point (id)"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE var_value ADD CONSTRAINT uq_time_grid_var "
                "UNIQUE (time_id, grid_id, var_id)"
            )
        )
        # wide types and the PostGIS point of older versions
        conn.execute(
            text(
                "ALTER TABLE var_value ALTER COLUMN var_id TYPE integer, "
                "ALTER COLUMN value TYPE double precision"
            )
        )
        conn.execute(text("ALTER TABLE var_coverage ALTER COLUMN var_id TYPE integer"))
        conn.execute(
            text("ALTER TABLE grid_point ADD COLUMN point geometry(POINT, 4326)")
        )
        conn.execute(
            text("CREATE INDEX idx_grid_point_point ON grid_point USING gist (point)")
        )
    postdb.upgrade_schema(engine)
    postdb.upgrade_schema(engine)  # idempotent
    with engine.connect() as conn:
        columns = {
            (t, c)
            for t, c in conn.execute(
                text(
                    "SELECT table_name, column_name FROM information_schema.columns "
                    "WHERE table_name IN ('var_value', 'var_type', 'grid_point')"
                )
            )
        }
        types = dict(
            conn.execute(
                text(
                    "SELECT table_name || '.' || column_name, data_type "
                    "FROM information_schema.columns "
                    "WHERE table_name IN ('var_value', 'var_coverage')"
                )
            ).all()
        )
        constraints = {
            row[0]
            for row in conn.execute(
                text(
                    "SELECT conname FROM pg_constraint "
                    "WHERE conrelid = 'var_value'::regclass"
                )
            )
        }
    assert ("var_value", "resolution_id") in columns
    assert ("var_type", "aggregation") in columns
    assert "uq_time_grid_var_res" in constraints
    assert "fk_resolution_id" in constraints
    assert "uq_time_grid_var" not in constraints
    assert ("var_value", "id") not in columns
    assert "var_value_pkey" not in constraints
    assert "var_value_grid_id_fkey" not in constraints
    assert "fk_grid_id" in constraints
    assert ("grid_point", "point") not in columns
    assert types["var_value.value"] == "real"
    assert types["var_value.var_id"] == "smallint"
    assert types["var_value.resolution_id"] == "smallint"
    assert types["var_coverage.var_id"] == "smallint"
    assert types["var_coverage.resolution_id"] == "smallint"


def test_normalize_db_url():
    assert (
        postdb.normalize_db_url("postgresql://u:p@db:5432/x")
        == "postgresql+psycopg2://u:p@db:5432/x"
    )
    assert (
        postdb.normalize_db_url("postgres://u:p@db/x")
        == "postgresql+psycopg2://u:p@db/x"
    )
    # an explicit driver is kept
    for url in ["postgresql+psycopg://u@db/x", "postgresql+psycopg2://u@db/x"]:
        assert postdb.normalize_db_url(url) == url


def test_apply_transform():
    values = np.array([np.nan, 0.0, 0.5, 1.0, 10.0, 1e9])
    out = postdb.apply_transform(values, "log10", floor=1.0)
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1:], [0.0, 0.0, 0.0, 1.0, 9.0])
    np.testing.assert_allclose(
        postdb.apply_transform(values, "log10", floor=0.1)[1:3], [-1.0, np.log10(0.5)]
    )
    assert postdb.apply_transform(values, None) is values
    # works lazily on xarray objects
    da = xr.DataArray(values).chunk()
    np.testing.assert_allclose(postdb.apply_transform(da, "log10").values[1:], out[1:])
    with pytest.raises(ValueError, match="Unknown transform"):
        postdb.apply_transform(values, "sqrt")
    with pytest.raises(ValueError, match="must be > 0"):
        postdb.apply_transform(values, "log10", floor=0)


def test_insert_var_values_by_resolution_transform_after_mean(
    get_engine_with_tables, get_session
):
    ds = _fine_grid_ds(n_time=1)
    # spread over orders of magnitude, with values below the floor
    values = 10.0 ** (np.arange(25).reshape(1, 5, 5) / 3.0 - 2.0)
    ds["adults"].values[:] = values
    postdb.insert_var_types(
        get_session,
        [{"name": "adults", "unit": "log10", "aggregation": "mean"}],
    )
    postdb.insert_grid_points(get_session, ds.latitude.values, ds.longitude.values)
    postdb.insert_time_points(get_session, [(ds.time.values, False)])
    postdb.insert_resolution_groups(get_session, resolutions=np.array([0.1, 0.2]))
    grid_id_map, time_id_map, var_id_map = postdb.get_id_maps(get_session)
    resolution_id_map = postdb.get_resolution_id_map(get_session)
    with pytest.raises(ValueError, match="Unknown transform"):
        postdb.insert_var_values_by_resolution(
            get_engine_with_tables,
            ds,
            "adults",
            grid_id_map,
            time_id_map,
            var_id_map,
            resolution_id_map,
            transform="sqrt",
        )
    postdb.insert_var_values_by_resolution(
        get_engine_with_tables,
        ds,
        "adults",
        grid_id_map,
        time_id_map,
        var_id_map,
        resolution_id_map,
        transform="log10",
        transform_floor=1.0,
    )
    postdb.assign_grid_resolution_group_to_grid_point(get_session)
    plan = postdb._block_mean_plan(ds.latitude.values, ds.longitude.values, 0.2)
    expected_coarse = np.log10(np.maximum(postdb.block_mean(values, plan), 1.0))
    # the log is taken of the mean, not the mean of the logs
    coarse = postdb.get_var_value(
        get_session, "adults", 0.2, 0.2, 2023, 1, 1, resolution=0.2
    )
    np.testing.assert_allclose(coarse, expected_coarse[0, 1, 1])
    assert not np.isclose(coarse, np.log10(np.maximum(values[0, 1:4, 1:4], 1)).mean())
    native = postdb.get_var_value(get_session, "adults", 0.4, 0.0, 2023, 1, 1)
    assert native == 0.0  # 10**-2 is below the floor
    native = postdb.get_var_value(get_session, "adults", 0.0, 0.4, 2023, 1, 1)
    np.testing.assert_allclose(native, np.log10(values[0, 4, 4]))


def test_block_valid_and_drop_covered_zeros():
    lats = np.round(np.arange(0.4, -0.05, -0.1), 1)
    lons = np.round(np.arange(0.0, 0.45, 0.1), 1)
    plan = postdb._block_mean_plan(lats, lons, 0.2)
    valid = np.zeros((5, 5), dtype=bool)
    valid[0, 0] = True  # only in the window of the coarse cell (0.4, 0.0)
    coarse = postdb.block_valid(valid, plan)
    values = np.full((1, 5, 5), np.nan)
    values[0, 0, 0] = 1.0
    np.testing.assert_array_equal(
        coarse, np.isfinite(postdb.block_mean(values, plan)[0])
    )
    native = postdb._block_mean_plan(lats, lons, 0.1)
    np.testing.assert_array_equal(postdb.block_valid(valid, native), valid)

    covered = np.array([[True, False]])
    out = postdb.drop_covered_zeros(np.array([[[0.0, 0.0]], [[1.0, 0.0]]]), covered)
    np.testing.assert_array_equal(np.isnan(out), [[[True, False]], [[False, False]]])
    da = xr.DataArray(np.array([[[0.0, 0.0]]]), dims=("time", "lat", "lon"))
    out = postdb.drop_covered_zeros(da, xr.DataArray(covered, dims=("lat", "lon")))
    assert np.isnan(out.values[0, 0, 0]) and out.values[0, 0, 1] == 0.0


def _zeros_ds(var_name):
    """5x5 grid, 2 time points: zeros and positive values on land, an ocean
    column, and one cell that is NaN only at the second time point."""
    ds = _fine_grid_ds(var_name=var_name, n_time=2)
    values = ds[var_name].values
    values[:, :, :2] = 0.0  # zeros on land
    values[:, :, 4] = np.nan  # ocean
    values[1, 2, 3] = np.nan  # missing at one time point only
    values[0, 2, 3] = 0.0  # ... and 0 at the other one
    return ds


def _cartesian_values(session, var_name, month, resolution):
    result = postdb.get_var_values_cartesian(
        session, (2023, month), grid_resolution=resolution, var_name=var_name
    )["latitude, longitude, var_value"]
    return {(round(float(a), 1), round(float(o), 1)): v for a, o, v in result}


def test_insert_var_values_skips_covered_zeros(get_engine_with_tables, get_session):
    ds = _zeros_ds("t2m")
    postdb.insert_var_types(get_session, [{"name": "t2m", "unit": "K"}])
    postdb.insert_grid_points(get_session, ds.latitude.values, ds.longitude.values)
    postdb.insert_time_points(get_session, [(ds.time.values, False)])
    postdb.insert_resolution_groups(get_session, resolutions=np.array([0.1, 0.2]))
    postdb.assign_grid_resolution_group_to_grid_point(get_session)
    grid_id_map, time_id_map, var_id_map = postdb.get_id_maps(get_session)
    postdb.insert_var_values(
        get_engine_with_tables, ds, "t2m", grid_id_map, time_id_map, var_id_map
    )
    with get_engine_with_tables.connect() as conn:
        n_rows = conn.execute(text("SELECT COUNT(*) FROM var_value")).scalar()
        n_coverage = conn.execute(text("SELECT COUNT(*) FROM var_coverage")).scalar()
    # 4 land columns, minus the cell that is NaN once
    assert n_coverage == 19
    # stored: positive values (2 columns x 5 rows x 2 times, minus the cell
    # that is 0 once and NaN once) and the 0 of that uncovered cell
    assert n_rows == 20 - 2 + 1
    for month in (1, 2):
        got = _cartesian_values(get_session, "t2m", month, 0.1)
        expected = {
            (round(float(a), 1), round(float(o), 1)): float(v)
            for (a, o), v in ds["t2m"].isel(time=month - 1).to_series().items()
            if np.isfinite(v)
        }
        assert got == expected
    assert postdb.get_var_value(get_session, "t2m", 0.4, 0.0, 2023, 1, 1) == 0.0
    assert postdb.get_var_value(get_session, "t2m", 0.2, 0.3, 2023, 1, 1) == 0.0
    assert postdb.get_var_value(get_session, "t2m", 0.2, 0.3, 2023, 2, 1) is None
    assert postdb.get_var_value(get_session, "t2m", 0.4, 0.4, 2023, 1, 1) is None
    assert postdb.get_var_value(get_session, "t2m", 0.4, 0.0, 2024, 1, 1) is None
    # a variable is inserted in one go
    with pytest.raises(ValueError, match="already in the database"):
        postdb.insert_var_values(
            get_engine_with_tables, ds, "t2m", grid_id_map, time_id_map, var_id_map
        )


def test_insert_var_values_by_resolution_skips_covered_zeros(
    get_engine_with_tables, get_session
):
    ds = _zeros_ds("adults")
    _seed_mean_var(get_session, get_engine_with_tables, ds)
    for resolution in (0.1, 0.2):
        plan = postdb._block_mean_plan(ds.latitude.values, ds.longitude.values, 0.2)
        for month in (1, 2):
            values = ds["adults"].values[month - 1 : month]
            got = _cartesian_values(get_session, "adults", month, resolution)
            if resolution == 0.1:
                expected_grid, lats, lons = values[0], ds.latitude, ds.longitude
            else:
                expected_grid = postdb.block_mean(values, plan)[0]
                lats, lons = plan["latitude"], plan["longitude"]
            expected = {
                (round(float(a), 1), round(float(o), 1)): float(expected_grid[i, j])
                for i, a in enumerate(np.asarray(lats))
                for j, o in enumerate(np.asarray(lons))
                if np.isfinite(expected_grid[i, j])
            }
            assert got.keys() == expected.keys()
            np.testing.assert_allclose(
                [got[k] for k in expected], list(expected.values())
            )
    with get_engine_with_tables.connect() as conn:
        zeros = conn.execute(
            text("SELECT COUNT(*) FROM var_value WHERE value = 0")
        ).scalar()
    # only the uncovered cell keeps its 0 (at 0.1 degree; at 0.2 degree its
    # coarse cell always has other valid cells)
    assert zeros == 1


def test_download_fills_covered_zeros(get_engine_with_tables, get_session, tmp_path):
    ds = _zeros_ds("adults")
    _seed_mean_var(get_session, get_engine_with_tables, ds)
    out = tmp_path / "out.nc"
    postdb.get_var_values_cartesian_for_download(
        get_session, (2023, 1), (2023, 2), var_names=["adults"], netcdf_file=str(out)
    )
    with xr.open_dataset(out) as result:
        got = result["adults"].sel(latitude=ds.latitude, longitude=ds.longitude)
        np.testing.assert_array_equal(got.values, ds["adults"].values)
