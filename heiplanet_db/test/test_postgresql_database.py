import pytest
from heiplanet_db import postgresql_database as postdb
import numpy as np
import xarray as xr
from testcontainers.postgres import PostgresContainer
from sqlalchemy import text, inspect
from sqlalchemy.orm.session import Session
import geopandas as gpd
from shapely.geometry import Polygon
import math
from fastapi import HTTPException
from conftest import cleanup
import pandas as pd


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


def test_extract_time_point():
    time_points = {
        np.datetime64("2024-01-01T00:00:00.000000000"): (2024, 1, 1),
        np.datetime64("2023-02-01 00:00:00"): (2023, 2, 1),
    }
    for time_point, expected_data in time_points.items():
        year, month, day, _, _, _ = postdb.extract_time_point(time_point)
        assert (year, month, day) == expected_data


def test_extract_time_point_invalid():
    with pytest.raises(ValueError):
        postdb.extract_time_point("2024-01-01")

    with pytest.raises(ValueError):
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


# Test successful threaded inserts with proper batching
def test_generate_threaded_inserts_success(monkeypatch):
    ds = _build_cartesian_ds()
    grid_id_map, time_id_map = _build_maps()

    calls = []

    def _fake_insert_batch(batch, engine, var_cls):
        calls.append(len(batch))

    monkeypatch.setattr(postdb, "insert_batch", _fake_insert_batch)
    monkeypatch.setattr(postdb, "tqdm", lambda it, total=None: it)
    monkeypatch.setattr(postdb, "BATCH_SIZE", 4)  # internal uses BATCH_SIZE//2 => 2

    total = postdb.generate_threaded_inserts(
        t_chunk=10,
        lat_chunk=10,
        lon_chunk=10,
        ds=ds,
        var_name="t2m",
        grid_id_map=grid_id_map,
        time_id_map=time_id_map,
        var_id=1,
        engine=object(),
    )

    assert total == 12
    assert sum(calls) == 12
    assert len(calls) == 6  # 12 rows in batches of 2


# Test that empty chunks (all NaNs) are skipped and produce zero batches
def test_generate_threaded_inserts_edge_empty_chunk(monkeypatch):
    ds = _build_cartesian_ds(fill_nan=True)
    grid_id_map, time_id_map = _build_maps()

    called = {"n": 0}

    def _fake_insert_batch(batch, engine, var_cls):
        called["n"] += 1

    monkeypatch.setattr(postdb, "insert_batch", _fake_insert_batch)
    monkeypatch.setattr(postdb, "tqdm", lambda it, total=None: it)

    total = postdb.generate_threaded_inserts(
        t_chunk=10,
        lat_chunk=10,
        lon_chunk=10,
        ds=ds,
        var_name="t2m",
        grid_id_map=grid_id_map,
        time_id_map=time_id_map,
        var_id=1,
        engine=object(),
    )

    assert total == 0
    assert called["n"] == 0


# Test missing ID maps result in zero inserts, and worker exceptions propagate
def test_generate_threaded_inserts_edge_missing_ids_and_worker_error(monkeypatch):
    ds = _build_cartesian_ds()
    grid_id_map, time_id_map = _build_maps()

    # missing IDs -> no inserts
    monkeypatch.setattr(postdb, "insert_batch", lambda batch, engine, var_cls: None)
    monkeypatch.setattr(postdb, "tqdm", lambda it, total=None: it)

    total = postdb.generate_threaded_inserts(
        t_chunk=10,
        lat_chunk=10,
        lon_chunk=10,
        ds=ds,
        var_name="t2m",
        grid_id_map={},  # missing all grid ids
        time_id_map={},  # missing all time ids
        var_id=1,
        engine=object(),
    )
    assert total == 0

    # worker failure should propagate
    def _raise_insert_batch(batch, engine, var_cls):
        raise RuntimeError("boom")

    monkeypatch.setattr(postdb, "insert_batch", _raise_insert_batch)

    with pytest.raises(RuntimeError, match="boom"):
        postdb.generate_threaded_inserts(
            t_chunk=10,
            lat_chunk=10,
            lon_chunk=10,
            ds=ds,
            var_name="t2m",
            grid_id_map=grid_id_map,
            time_id_map=time_id_map,
            var_id=1,
            engine=object(),
        )


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


def test__yield_chunks_full_cover_and_counts():
    """Test that _yield_chunks produces correct number of chunks and covers all data."""
    # Create dataset with sizes that exercise chunk boundaries
    times = np.array(
        ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"], dtype="datetime64[ns]"
    )
    lats = np.array([10.1, 10.2, 10.3], dtype=float)
    lons = np.array([10.1, 10.2, 10.3, 10.4, 10.5], dtype=float)
    values = np.arange(60, dtype=float).reshape(4, 3, 5)

    ds = xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), values)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )

    # Chunk sizes: t_chunk=2, lat_chunk=2, lon_chunk=3
    # Expected chunks: ceil(4/2) * ceil(3/2) * ceil(5/3) = 2 * 2 * 2 = 8 chunks
    chunks = list(postdb._yield_chunks(ds, t_chunk=2, lat_chunk=2, lon_chunk=3))
    assert len(chunks) == 8

    # Verify each chunk has correct sizes
    for chunk in chunks:
        assert chunk.sizes["time"] <= 2
        assert chunk.sizes["latitude"] <= 2
        assert chunk.sizes["longitude"] <= 3

    # Verify all coordinates are covered (de-duplicate and check)
    all_times = []
    all_lats = []
    all_lons = []
    for chunk in chunks:
        all_times.extend(chunk.time.values)
        all_lats.extend(chunk.latitude.values)
        all_lons.extend(chunk.longitude.values)

    assert len(set(all_times)) == 4  # All 4 time points covered
    assert len(set(all_lats)) == 3  # All 3 latitudes covered
    assert len(set(all_lons)) == 5  # All 5 longitudes covered


def test__yield_chunks_single_large_chunk():
    """Test that chunk sizes larger than dataset yield a single chunk."""
    ds = _build_cartesian_ds()

    # Chunk sizes larger than dataset dimensions
    chunks = list(postdb._yield_chunks(ds, t_chunk=100, lat_chunk=100, lon_chunk=100))
    assert len(chunks) == 1

    # Single chunk should match original dataset sizes
    chunk = chunks[0]
    assert chunk.sizes["time"] == ds.sizes["time"]
    assert chunk.sizes["latitude"] == ds.sizes["latitude"]
    assert chunk.sizes["longitude"] == ds.sizes["longitude"]


def test__process_chunk_drops_all_nan_latitudes():
    """Test that _process_chunk drops latitude rows that are all NaN."""
    # Create dataset where one latitude row is all NaN
    times = np.array(["2023-01-01"], dtype="datetime64[ns]")
    lats = np.array([10.1, 10.2, 10.3], dtype=float)
    lons = np.array([10.1, 10.2], dtype=float)
    values = np.array(
        [
            [[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]]  # middle lat row is all NaN
        ]
    )

    ds_chunk = xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), values)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )

    grid_id_map = {
        (10.1, 10.1): 1,
        (10.1, 10.2): 2,
        (10.2, 10.1): 3,
        (10.2, 10.2): 4,
        (10.3, 10.1): 5,
        (10.3, 10.2): 6,
    }
    time_id_map = {_build_time_key("2023-01-01"): 1}
    var_id = 1

    result = postdb._process_chunk(ds_chunk, "t2m", grid_id_map, time_id_map, var_id)

    # Should have 4 entries (2 for lat=10.1, 2 for lat=10.3), skipping lat=10.2
    assert len(result) == 4
    # Verify no entries for lat=10.2
    grid_ids_in_result = {r["grid_id"] for r in result}
    assert 3 not in grid_ids_in_result  # grid_id 3 corresponds to (10.2, 10.1)
    assert 4 not in grid_ids_in_result  # grid_id 4 corresponds to (10.2, 10.2)


def test__process_chunk_returns_empty_for_all_nan():
    """Test that _process_chunk returns empty list for all-NaN datasets."""
    ds_chunk = _build_cartesian_ds(fill_nan=True)
    grid_id_map, time_id_map = _build_maps()

    result = postdb._process_chunk(ds_chunk, "t2m", grid_id_map, time_id_map, var_id=1)
    assert result == []


def test__process_chunk_returns_empty_for_invalid_ndim(monkeypatch):
    """Test that _process_chunk returns empty list for corrupted data (wrong ndim)."""
    # Create a normal 3D dataset
    ds_chunk = _build_cartesian_ds()
    grid_id_map, time_id_map = _build_maps()

    # Monkeypatch the DataArray.values property to return a 2D array
    # This simulates the corruption scenario that the guard is meant to catch
    class MockDataArray:
        def __init__(self, original_da):
            self._original = original_da
            self.time = original_da.time
            self.latitude = original_da.latitude
            self.longitude = original_da.longitude
            self.sizes = original_da.sizes

        def load(self):
            return self

        def dropna(self, dim, how):
            return self

        @property
        def size(self):
            return self._original.size

        @property
        def values(self):
            # Return 2D array instead of 3D to trigger the guard
            orig_values = self._original.values
            return orig_values.reshape(orig_values.shape[0], -1)

    original_getitem = xr.Dataset.__getitem__

    def _mock_getitem(self, key):
        result = original_getitem(self, key)
        if key == "t2m":
            return MockDataArray(result)
        return result

    monkeypatch.setattr(xr.Dataset, "__getitem__", _mock_getitem)

    result = postdb._process_chunk(ds_chunk, "t2m", grid_id_map, time_id_map, var_id=1)
    # Should return empty list due to ndim != 3 guard
    assert result == []


def test__process_chunk_forwards_to_get_var_values_mapping(monkeypatch):
    """Test that _process_chunk correctly forwards data to get_var_values_mapping."""
    ds_chunk = _build_cartesian_ds()
    grid_id_map, time_id_map = _build_maps()

    captured_args = {}

    def _capture_get_var_values_mapping(
        times, time_id_map, grid_id_map, var_id, lats, lons, values
    ):
        captured_args["times"] = times
        captured_args["time_id_map"] = time_id_map
        captured_args["grid_id_map"] = grid_id_map
        captured_args["var_id"] = var_id
        captured_args["lats"] = lats
        captured_args["lons"] = lons
        captured_args["values"] = values
        return [{"grid_id": 1, "time_id": 1, "var_id": 99, "value": 1.0}]  # sentinel

    monkeypatch.setattr(
        postdb, "get_var_values_mapping", _capture_get_var_values_mapping
    )

    result = postdb._process_chunk(ds_chunk, "t2m", grid_id_map, time_id_map, var_id=99)

    # Verify forwarding
    assert len(result) == 1
    assert result[0]["var_id"] == 99
    assert captured_args["var_id"] == 99
    assert captured_args["times"].shape == (2,)  # 2 time points
    assert captured_args["lats"].shape == (2,)  # 2 latitudes
    assert captured_args["lons"].shape == (3,)  # 3 longitudes
    assert captured_args["values"].shape == (2, 2, 3)  # (time, lat, lon)


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


def test__build_var_value_entry():
    """Test that _build_var_value_entry creates correct dictionary structure."""
    result = postdb._build_var_value_entry(grid_id=1, time_id=2, var_id=3, value=42.5)

    assert isinstance(result, dict)
    assert result == {
        "grid_id": 1,
        "time_id": 2,
        "var_id": 3,
        "value": 42.5,
    }

    # Test type conversions
    result2 = postdb._build_var_value_entry(
        grid_id=np.int64(100), time_id=np.int32(200), var_id=300, value=np.float64(99.9)
    )
    assert isinstance(result2["grid_id"], int)
    assert isinstance(result2["time_id"], int)
    assert isinstance(result2["var_id"], int)
    assert isinstance(result2["value"], float)
    assert result2["grid_id"] == 100
    assert result2["time_id"] == 200
    assert result2["var_id"] == 300
    assert math.isclose(result2["value"], 99.9, rel_tol=1e-4)

    # Test with zero and negative values
    result3 = postdb._build_var_value_entry(
        grid_id=0, time_id=0, var_id=0, value=np.float64(-1.5)
    )
    assert result3 == {
        "grid_id": 0,
        "time_id": 0,
        "var_id": 0,
        "value": np.float64(-1.5),
    }


def test__process_time_point_happy_path(monkeypatch):
    """Test _process_time_point with complete mappings and valid data."""
    monkeypatch.setattr(postdb, "ROUND_DIGITS", 4)

    times = np.array(["2023-01-01", "2023-02-01"], dtype="datetime64[ns]")
    lats = np.array([10.1, 10.2], dtype=float)
    lons = np.array([10.1, 10.2], dtype=float)
    values = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

    grid_id_map = {
        (10.1, 10.1): 1,
        (10.1, 10.2): 2,
        (10.2, 10.1): 3,
        (10.2, 10.2): 4,
    }
    time_id_map = {_build_time_key("2023-01-01"): 10, _build_time_key("2023-02-01"): 20}
    var_id = 5

    # Process first time point
    result = postdb._process_time_point(
        0, times[0], time_id_map, grid_id_map, var_id, lats, lons, values
    )

    assert len(result) == 4  # 2 lats * 2 lons
    assert all(entry["time_id"] == 10 for entry in result)
    assert all(entry["var_id"] == 5 for entry in result)
    assert {entry["value"] for entry in result} == {1.0, 2.0, 3.0, 4.0}
    assert {entry["grid_id"] for entry in result} == {1, 2, 3, 4}


def test__process_time_point_skips_nans(monkeypatch):
    """Test that _process_time_point skips NaN values."""
    monkeypatch.setattr(postdb, "ROUND_DIGITS", 4)

    times = np.array(["2023-01-01"], dtype="datetime64[ns]")
    lats = np.array([10.1, 10.2], dtype=float)
    lons = np.array([10.1, 10.2], dtype=float)
    values = np.array([[[1.0, np.nan], [np.nan, 4.0]]])

    grid_id_map = {
        (10.1, 10.1): 1,
        (10.1, 10.2): 2,
        (10.2, 10.1): 3,
        (10.2, 10.2): 4,
    }
    time_id_map = {_build_time_key("2023-01-01"): 1}
    var_id = 1

    result = postdb._process_time_point(
        0, times[0], time_id_map, grid_id_map, var_id, lats, lons, values
    )

    assert len(result) == 2  # Only non-NaN values
    assert {entry["value"] for entry in result} == {1.0, 4.0}
    assert all(not np.isnan(entry["value"]) for entry in result)


def test__process_time_point_missing_time_id(monkeypatch, capsys):
    """Test that _process_time_point returns empty list when time_id is missing."""
    monkeypatch.setattr(postdb, "ROUND_DIGITS", 4)

    times = np.array(["2023-01-01"], dtype="datetime64[ns]")
    lats = np.array([10.1], dtype=float)
    lons = np.array([10.1], dtype=float)
    values = np.array([[[1.0]]])

    grid_id_map = {(10.1, 10.1): 1}
    time_id_map = {}  # Missing time_id
    var_id = 1

    result = postdb._process_time_point(
        0, times[0], time_id_map, grid_id_map, var_id, lats, lons, values
    )

    assert result == []
    captured = capsys.readouterr()
    assert "Missing time_id" in captured.out


def test__process_time_point_missing_grid_id(monkeypatch, capsys):
    """Test that _process_time_point skips entries with missing grid_id."""
    monkeypatch.setattr(postdb, "ROUND_DIGITS", 4)

    times = np.array(["2023-01-01"], dtype="datetime64[ns]")
    lats = np.array([10.1, 10.2], dtype=float)
    lons = np.array([10.1, 10.2], dtype=float)
    values = np.array([[[1.0, 2.0], [3.0, 4.0]]])

    grid_id_map = {
        (10.1, 10.1): 1,
        # Missing (10.1, 10.2), (10.2, 10.1), (10.2, 10.2)
    }
    time_id_map = {_build_time_key("2023-01-01"): 1}
    var_id = 1

    result = postdb._process_time_point(
        0, times[0], time_id_map, grid_id_map, var_id, lats, lons, values
    )

    assert len(result) == 1
    assert result[0]["grid_id"] == 1
    assert result[0]["value"] == 1.0
    captured = capsys.readouterr()
    assert "Missing grid_id" in captured.out


def test_get_var_values_mapping_happy_path(monkeypatch):
    """Test get_var_values_mapping with complete ID maps and no NaNs."""
    monkeypatch.setattr(postdb, "ROUND_DIGITS", 4)

    times = np.array(
        ["2023-01-01T00:00:00", "2023-02-01T12:30:00"], dtype="datetime64[ns]"
    )
    lats = np.array([10.12345, 10.23456], dtype=float)
    lons = np.array([10.11111, 10.22222, 10.33333], dtype=float)
    values = np.arange(12, dtype=np.float64).reshape(2, 2, 3)

    # Build maps with rounded keys
    # ensure consistent rounding for the different float types
    # and float comparisons
    grid_id_map = {
        (postdb._q(float(lats[0])), postdb._q(lons[0])): 1,
        (postdb._q(lats[0]), postdb._q(lons[1])): 2,
        (postdb._q(lats[0]), postdb._q(lons[2])): 3,
        (postdb._q(lats[1]), postdb._q(lons[0])): 4,
        (postdb._q(lats[1]), postdb._q(lons[1])): 5,
        (postdb._q(lats[1]), postdb._q(lons[2])): 6,
    }
    time_id_map = {
        _build_time_key("2023-01-01"): 1,
        _build_time_key("2023-02-01"): 2,
    }
    var_id = 42

    result = postdb.get_var_values_mapping(
        times, time_id_map, grid_id_map, var_id, lats, lons, values
    )

    # Should have 12 entries (2 * 2 * 3)
    assert len(result) == 12

    # Verify structure of entries
    for entry in result:
        assert isinstance(entry, dict)
        assert set(entry.keys()) == {"grid_id", "time_id", "var_id", "value"}
        assert isinstance(entry["grid_id"], int)
        assert isinstance(entry["time_id"], int)
        assert isinstance(entry["var_id"], int)
        assert isinstance(entry["value"], float)
        assert entry["var_id"] == 42

    # Verify values match expected order (time, lat, lon)
    assert result[0]["value"] == 0.0  # values[0, 0, 0]
    assert result[0]["time_id"] == 1
    assert result[0]["grid_id"] == 1

    assert result[11]["value"] == 11.0  # values[1, 1, 2]
    assert result[11]["time_id"] == 2
    assert result[11]["grid_id"] == 6

    # Verify all time points are represented
    assert {entry["time_id"] for entry in result} == {1, 2}


def test_get_var_values_mapping_skips_missing_ids_and_nans(monkeypatch, capsys):
    """Test that get_var_values_mapping skips NaNs and missing ID mappings."""
    monkeypatch.setattr(postdb, "ROUND_DIGITS", 4)

    times = np.array(["2023-01-01", "2023-02-01"], dtype="datetime64[ns]")
    lats = np.array([10.1, 10.2], dtype=float)
    lons = np.array([10.1, 10.2, 10.3], dtype=float)
    values = np.array(
        [
            [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]],  # one NaN in first time slice
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )

    # Missing some grid IDs and time IDs
    grid_id_map = {
        (10.1, 10.1): 1,
        (10.1, 10.3): 3,  # missing (10.1, 10.2)
        (10.2, 10.1): 4,
        (10.2, 10.2): 5,
        # missing (10.2, 10.3)
    }
    time_id_map = {
        _build_time_key("2023-01-01"): 1,
        # missing 2023-02-01
    }
    var_id = 1

    result = postdb.get_var_values_mapping(
        times, time_id_map, grid_id_map, var_id, lats, lons, values
    )

    # Expected: only entries for time=2023-01-01, non-NaN values, and existing grid IDs
    # time=0: (10.1,10.1)=1.0, (10.1,10.3)=3.0, (10.2,10.1)=4.0, (10.2,10.2)=5.0 = 4 entries
    # time=1: skipped (missing time_id)
    assert len(result) == 4

    # Verify no NaN values
    for entry in result:
        assert not np.isnan(entry["value"])
        assert isinstance(entry["value"], (int, float))

    # Verify all entries have valid IDs
    for entry in result:
        assert entry["time_id"] == 1  # only first time point
        assert entry["grid_id"] in [1, 3, 4, 5]  # only existing grid IDs
        assert entry["var_id"] == 1

    # Verify warning messages are printed
    captured = capsys.readouterr()
    assert "Missing time_id" in captured.out
    assert "Missing grid_id" in captured.out


def test_get_var_values_mapping_time_key_normalization(monkeypatch):
    """Test that get_var_values_mapping normalizes time keys correctly."""
    monkeypatch.setattr(postdb, "ROUND_DIGITS", 4)

    # Times with full precision (hours/minutes/seconds)
    times = np.array(
        [
            "2023-01-01T14:30:00",
            "2023-02-01T23:59:59",
        ],
        dtype="datetime64[ns]",
    )
    lats = np.array([10.1], dtype=float)
    lons = np.array([10.1], dtype=float)
    values = np.array([[[1.0]], [[2.0]]])

    # Map uses normalized midnight dates
    time_id_map = {
        _build_time_key("2023-01-01"): 1,
        _build_time_key("2023-02-01"): 2,
    }
    grid_id_map = {(10.1, 10.1): 1}
    var_id = 1

    result = postdb.get_var_values_mapping(
        times, time_id_map, grid_id_map, var_id, lats, lons, values
    )

    # Should find both time IDs despite different hour/minute precision
    assert len(result) == 2
    assert {r["time_id"] for r in result} == {1, 2}
    assert {r["value"] for r in result} == {1.0, 2.0}
    assert all(r["grid_id"] == 1 for r in result)
    assert all(r["var_id"] == 1 for r in result)


def test_get_var_values_mapping_empty_arrays(monkeypatch):
    """Test get_var_values_mapping with empty input arrays."""
    monkeypatch.setattr(postdb, "ROUND_DIGITS", 4)

    times = np.array([], dtype="datetime64[ns]")
    lats = np.array([], dtype=float)
    lons = np.array([], dtype=float)
    values = np.array([]).reshape(0, 0, 0)

    grid_id_map = {}
    time_id_map = {}
    var_id = 1

    result = postdb.get_var_values_mapping(
        times, time_id_map, grid_id_map, var_id, lats, lons, values
    )

    assert result == []


def test_get_var_values_mapping_single_point(monkeypatch):
    """Test get_var_values_mapping with single time/lat/lon point."""
    monkeypatch.setattr(postdb, "ROUND_DIGITS", 4)

    times = np.array(["2023-01-01"], dtype="datetime64[ns]")
    lats = np.array([10.1], dtype=float)
    lons = np.array([20.2], dtype=float)
    values = np.array([[[42.5]]])

    grid_id_map = {(10.1, 20.2): 100}
    time_id_map = {_build_time_key("2023-01-01"): 50}
    var_id = 7

    result = postdb.get_var_values_mapping(
        times, time_id_map, grid_id_map, var_id, lats, lons, values
    )

    assert len(result) == 1
    assert result[0] == {"grid_id": 100, "time_id": 50, "var_id": 7, "value": 42.5}


def test_get_var_values_mapping_all_nans(monkeypatch):
    """Test get_var_values_mapping when all values are NaN."""
    monkeypatch.setattr(postdb, "ROUND_DIGITS", 4)

    times = np.array(["2023-01-01"], dtype="datetime64[ns]")
    lats = np.array([10.1, 10.2], dtype=float)
    lons = np.array([20.1, 20.2], dtype=float)
    values = np.full((1, 2, 2), np.nan)

    grid_id_map = {
        (10.1, 20.1): 1,
        (10.1, 20.2): 2,
        (10.2, 20.1): 3,
        (10.2, 20.2): 4,
    }
    time_id_map = {_build_time_key("2023-01-01"): 1}
    var_id = 1

    result = postdb.get_var_values_mapping(
        times, time_id_map, grid_id_map, var_id, lats, lons, values
    )

    assert result == []


def test_get_var_values_mapping_coordinate_rounding(monkeypatch):
    """Test that get_var_values_mapping correctly rounds coordinates using _q."""
    monkeypatch.setattr(postdb, "ROUND_DIGITS", 2)

    times = np.array(["2023-01-01"], dtype="datetime64[ns]")
    lats = np.array([10.123456, 10.234567], dtype=float)
    lons = np.array([20.111111, 20.222222], dtype=float)
    values = np.array([[[1.0, 2.0], [3.0, 4.0]]])

    # Grid map uses rounded coordinates (ROUND_DIGITS=2)
    grid_id_map = {
        (round(10.123456, 2), round(20.111111, 2)): 1,
        (round(10.123456, 2), round(20.222222, 2)): 2,
        (round(10.234567, 2), round(20.111111, 2)): 3,
        (round(10.234567, 2), round(20.222222, 2)): 4,
    }
    time_id_map = {_build_time_key("2023-01-01"): 1}
    var_id = 1

    result = postdb.get_var_values_mapping(
        times, time_id_map, grid_id_map, var_id, lats, lons, values
    )

    assert len(result) == 4
    assert {entry["grid_id"] for entry in result} == {1, 2, 3, 4}
    assert {entry["value"] for entry in result} == {1.0, 2.0, 3.0, 4.0}


def test_get_var_values_mapping_value_types(monkeypatch):
    """Test get_var_values_mapping handles different numeric value types."""
    monkeypatch.setattr(postdb, "ROUND_DIGITS", 4)

    times = np.array(["2023-01-01"], dtype="datetime64[ns]")
    lats = np.array([10.1], dtype=float)
    lons = np.array([20.1], dtype=float)

    # Test with different numeric types
    values_int = np.array([[[42]]], dtype=int)
    values_float32 = np.array([[[99.5]]], dtype=np.float32)
    values_float64 = np.array([[[-15.25]]], dtype=np.float64)

    grid_id_map = {(10.1, 20.1): 1}
    time_id_map = {_build_time_key("2023-01-01"): 1}
    var_id = 1

    # Test integer values
    result1 = postdb.get_var_values_mapping(
        times, time_id_map, grid_id_map, var_id, lats, lons, values_int
    )
    assert result1[0]["value"] == 42.0
    assert isinstance(result1[0]["value"], float)

    # Test float32 values
    result2 = postdb.get_var_values_mapping(
        times, time_id_map, grid_id_map, var_id, lats, lons, values_float32
    )
    assert result2[0]["value"] == 99.5
    assert isinstance(result2[0]["value"], float)

    # Test float64 values
    result3 = postdb.get_var_values_mapping(
        times, time_id_map, grid_id_map, var_id, lats, lons, values_float64
    )
    assert result3[0]["value"] == -15.25
    assert isinstance(result3[0]["value"], float)


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
    assert math.isclose(ds_result.t2m[0, 0, 0], get_dataset.t2m[0, 0, 0], abs_tol=1e-5)
    assert math.isclose(ds_result.t2m[0, 1, 1], get_dataset.t2m[0, 1, 1], abs_tol=1e-5)
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
    #


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
