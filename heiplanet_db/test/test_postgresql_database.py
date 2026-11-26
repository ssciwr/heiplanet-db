import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import cleanup
from fastapi import HTTPException
from sqlalchemy import inspect, text
from sqlalchemy.orm.session import Session
from testcontainers.postgres import PostgresContainer

from heiplanet_db import postgresql_database as postdb
from heiplanet_db import tiles

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
        "time_point",
        "var_type",
        "var_value_nuts",
        "resolution_group",
        "grid_def",
        "var_tile",
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


def test_insert_resolution_groups(get_session):
    # first default case
    postdb.insert_resolution_groups(get_session)
    result = get_session.query(postdb.ResolutionGroup).all()
    assert len(result) == 9
    assert math.isclose(result[0].resolution, 0.1, abs_tol=1e-5)
    assert result[0].description == "0.1 degree resolution"


def test_extract_time_point():
    time_points = {
        np.datetime64("2024-01-01T00:00:00.000000000"): (2024, 1, 1),
        np.datetime64("2023-02-01 00:00:00"): (2023, 2, 1),
    }
    for time_point, expected_data in time_points.items():
        assert postdb.extract_time_point(time_point)[:3] == expected_data


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
    time_id_map, var_id_map = postdb.get_id_maps(get_session)

    assert len(time_id_map) == 2
    assert time_id_map[np.datetime64("2023-01-01", "ns")] == 1
    assert len(var_id_map) == 1
    assert var_id_map["t2m"] == 1

    # clean up
    get_session.execute(text("TRUNCATE TABLE time_point RESTART IDENTITY CASCADE"))
    get_session.execute(text("TRUNCATE TABLE var_type RESTART IDENTITY CASCADE"))
    get_session.commit()


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


def test_get_var_value(insert_data, get_dataset):
    session = insert_data
    values = get_dataset["t2m"].values
    result = postdb.get_var_value(session, "t2m", 10.2, 10.3, 2024, 1, 1)
    np.testing.assert_allclose(result, values[1, 1, 2], rtol=1e-6)
    assert isinstance(result, float)
    # only monthly data: any day gives the first of the month
    assert postdb.get_var_value(session, "t2m", 10.1, 10.1, 2023, 1, 15) == (
        postdb.get_var_value(session, "t2m", 10.1, 10.1, 2023, 1, 1)
    )
    # "subsample" values are the same at every resolution
    assert postdb.get_var_value(
        session, "t2m", 10.2, 10.3, 2024, 1, 1, resolution=0.2
    ) == pytest.approx(result)

    # None cases
    assert postdb.get_var_value(session, "non_existing", 10.2, 10.3, 2024, 1, 1) is None
    assert postdb.get_var_value(session, "t2m", 10.2, 10.3, 2020, 1, 1) is None
    # off the grid and between grid points
    assert postdb.get_var_value(session, "t2m", 50.0, 10.3, 2024, 1, 1) is None
    assert postdb.get_var_value(session, "t2m", 10.15, 10.3, 2024, 1, 1) is None


def test_get_var_value_no_tiles(get_session):
    postdb.insert_var_types(get_session, [{"name": "t2m", "unit": "K"}])
    assert postdb.get_var_value(get_session, "t2m", 10.2, 10.3, 2024, 1, 1) is None


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
    # test for non-finite area bounds
    for bad in (math.inf, math.nan):
        with pytest.raises(HTTPException) as e:
            postdb.get_var_values_cartesian(
                insert_data,
                time_point=(2023, 1),
                area=(bad, 10.0, 10.0, 10.2),
                var_name="t2m",
            )
        assert e.value.status_code == 400

    # values come back in their shortest form, without float32 noise
    ds_result = postdb.get_var_values_cartesian(
        insert_data, time_point=(2023, 1), var_name="t2m"
    )
    for _, _, value in ds_result["latitude, longitude, var_value"]:
        assert value == float(str(np.float32(value)))


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
    time_id_map, var_id_map = postdb.get_id_maps(get_session)

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
    time_id_map, var_id_map = postdb.get_id_maps(get_session)

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
    time_id_map, var_id_map = postdb.get_id_maps(get_session)

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
    time_id_map, var_id_map = postdb.get_id_maps(get_session)

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
    ds_no_time = ds.isel(time=0, drop=True)
    with pytest.raises(ValueError, match="must have dims"):
        postdb._to_time_lat_lon(ds_no_time, "adults")


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


def test_time_id_array():
    time_id_map = {postdb._normalize_time_key(np.datetime64("2023-01-01")): 5}
    ids = postdb.time_id_array(
        np.array(["2023-01-01T12:00"], dtype="datetime64[ns]"), time_id_map
    )
    assert ids.tolist() == [5]
    with pytest.raises(ValueError, match="not in the database"):
        postdb.time_id_array(np.array(["2023-02-01"], dtype="datetime64[ns]"), {})


def _create_legacy_tables(engine):
    """The tables of the old row-per-value storage, with their foreign keys."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE grid_point (
                    id serial PRIMARY KEY, latitude numeric, longitude numeric);
                CREATE TABLE grid_point_resolution (
                    grid_id integer REFERENCES grid_point (id),
                    resolution_id integer REFERENCES resolution_group (id));
                CREATE TABLE var_value (
                    grid_id integer REFERENCES grid_point (id),
                    time_id integer REFERENCES time_point (id),
                    var_id smallint REFERENCES var_type (id),
                    resolution_id smallint, value real);
                CREATE TABLE var_coverage (
                    var_id smallint REFERENCES var_type (id),
                    resolution_id smallint,
                    grid_id integer REFERENCES grid_point (id));
                """
            )
        )


def _table_names(engine):
    return set(inspect(engine).get_table_names(schema="public"))


def test_upgrade_schema_from_old_tables(get_engine_with_tables):
    engine = get_engine_with_tables
    _create_legacy_tables(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE var_type DROP COLUMN aggregation"))
    assert set(postdb.LEGACY_TABLES) <= _table_names(engine)

    postdb.upgrade_schema(engine)
    postdb.upgrade_schema(engine)  # idempotent

    assert not set(postdb.LEGACY_TABLES) & _table_names(engine)
    columns = {c["name"] for c in inspect(engine).get_columns("var_type")}
    assert "aggregation" in columns


def test_create_or_replace_tables_with_legacy_tables(get_engine_without_tables):
    engine = get_engine_without_tables
    postdb.create_tables(engine)
    _create_legacy_tables(engine)
    # the legacy tables reference time_point and var_type
    postdb.create_or_replace_tables(engine)
    tables = _table_names(engine)
    assert not set(postdb.LEGACY_TABLES) & tables
    assert {"var_tile", "grid_def", "time_point"} <= tables


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


# --- tiled storage ---------------------------------------------------------


def _seed_tiles(session, engine, ds, aggregation, tile_size=2, **kwargs):
    """Insert var type, time points, resolution groups and `adults` as tiles."""
    postdb.insert_var_types(
        session,
        [{"name": "adults", "unit": "-", "aggregation": aggregation}],
    )
    postdb.insert_time_points(session, [(ds.time.values, False)])
    postdb.insert_resolution_groups(session, resolutions=np.array([0.1, 0.2, 0.25]))
    time_id_map, var_id_map = postdb.get_id_maps(session)
    resolution_id_map = postdb.get_resolution_id_map(session)
    total = postdb.insert_var_tiles(
        engine,
        ds,
        "adults",
        time_id_map,
        var_id_map,
        resolution_id_map,
        aggregation=aggregation,
        tile_size=tile_size,
        **kwargs,
    )
    return total, resolution_id_map


def _read_tiles(session, resolution_id, time_id=1):
    grid = session.get(postdb.GridDefinition, resolution_id).to_grid()
    rows = session.execute(
        text(
            "SELECT tile_row, tile_col, data FROM var_tile "
            "WHERE resolution_id = :r AND time_id = :t"
        ),
        {"r": resolution_id, "t": time_id},
    ).all()
    return grid, tiles.tiles_to_array(
        rows, grid.tile_size, 0, grid.n_lat, 0, grid.n_lon
    )


def test_var_tile_schema(get_engine_with_tables, get_session):
    compression = get_session.execute(
        text(
            "SELECT attcompression FROM pg_attribute "
            "WHERE attrelid = 'var_tile'::regclass AND attname = 'data'"
        )
    ).scalar()
    assert compression == "l"  # lz4
    # values decoded in SQL match numpy's float32
    expected = np.array([1.0, -2.0, 0.15625, 1e-40, np.inf, 3.4e38], dtype="<f4")
    got = get_session.execute(
        text("SELECT tile_value(:d, k) FROM generate_series(0, 5) AS k ORDER BY k"),
        {"d": expected.tobytes()},
    ).scalars()
    np.testing.assert_array_equal(np.array(list(got), dtype="<f4"), expected)
    nan = get_session.execute(
        text("SELECT tile_value(:d, 0) = 'NaN'::real"),
        {"d": np.array([np.nan], dtype="<f4").tobytes()},
    ).scalar()
    assert nan


def test_insert_var_tiles_mean(get_engine_with_tables, get_session):
    ds = _fine_grid_ds(lon_lat_time=True)
    mask = xr.ones_like(ds["adults"], dtype=bool)
    mask[{"longitude": 4, "latitude": 0}] = False  # (lat 0.4, lon 0.4)
    total, resolution_id_map = _seed_tiles(
        get_session,
        get_engine_with_tables,
        ds,
        "mean",
        mask=mask,
        transform="log10",
    )
    # 0.1: 5x5 cells in 3x3 tiles of 2, but the corner tile holds only the
    # masked cell and is skipped; 0.2: 3x3 cells in 2x2 tiles; per time step
    assert total == 2 * (8 + 4)

    values = ds["adults"].transpose("time", "latitude", "longitude").values
    values = np.where(
        mask.transpose("time", "latitude", "longitude").values, values, np.nan
    )
    plan = postdb._block_mean_plan(ds.latitude.values, ds.longitude.values, 0.2)
    expected = np.log10(postdb.block_mean(values, plan))

    grid, native = _read_tiles(get_session, resolution_id_map[0.1])
    assert (grid.n_lat, grid.n_lon, grid.tile_size) == (5, 5, 2)
    np.testing.assert_allclose([grid.lat0, grid.lon0, grid.step], [0.0, 0.0, 0.1])
    # stored with ascending latitude
    np.testing.assert_allclose(native, np.log10(values[0, ::-1]), rtol=1e-6)
    assert np.isnan(native[4, 4])

    grid, coarse = _read_tiles(get_session, resolution_id_map[0.2], time_id=2)
    assert (grid.n_lat, grid.n_lon, grid.step) == (3, 3, 0.2)
    np.testing.assert_allclose(coarse, expected[1, ::-1], rtol=1e-6)
    # 0.25 is not a multiple of 0.1: no grid, no tiles
    assert get_session.get(postdb.GridDefinition, resolution_id_map[0.25]) is None

    # the debug view returns one row per value
    rows = get_session.execute(
        text(
            "SELECT latitude, longitude, value FROM var_value_v "
            "WHERE resolution_id = :r AND time_id = 1"
        ),
        {"r": resolution_id_map[0.1]},
    ).all()
    assert len(rows) == 24
    lookup = {(float(la), float(lo)): v for la, lo, v in rows}
    np.testing.assert_allclose(lookup[(0.2, 0.3)], np.log10(values[0, 2, 3]), rtol=1e-6)

    # a variable on another grid is rejected
    shifted = ds.assign_coords(longitude=ds.longitude + 1.0)
    get_session.execute(text("UPDATE var_type SET name = 'other'"))
    get_session.commit()
    with pytest.raises(ValueError, match="differs from the stored one"):
        postdb.insert_var_tiles(
            get_engine_with_tables,
            shifted.rename({"adults": "other"}),
            "other",
            {},
            {"other": 1},
            resolution_id_map,
        )


def test_insert_var_tiles_subsample(get_engine_with_tables, get_session):
    ds = _fine_grid_ds()
    total, resolution_id_map = _seed_tiles(
        get_session, get_engine_with_tables, ds, "subsample", tile_size=100
    )
    # one tile per resolution and time step
    assert total == 2 * 2
    values = ds["adults"].values
    _, native = _read_tiles(get_session, resolution_id_map[0.1], time_id=2)
    np.testing.assert_array_equal(native, values[1, ::-1])
    # every second point, ascending: latitudes 0.0, 0.2, 0.4
    _, coarse = _read_tiles(get_session, resolution_id_map[0.2])
    np.testing.assert_array_equal(coarse, values[0, ::-2][:, ::2])


def test_ensure_grid_defs_tolerance(get_engine_with_tables, get_session):
    postdb.insert_resolution_groups(get_session, np.array([0.1]))
    resolution_id = postdb.get_resolution_id_map(get_session)[0.1]
    grid = tiles.GridDef(-89.9, -179.9, 0.1, 1799, 3599, 100)
    postdb.ensure_grid_defs(get_engine_with_tables, {resolution_id: grid})
    # within COORD_ATOL: same grid
    close = tiles.GridDef(-89.9, -179.9 + 0.4e-4, 0.1, 1799, 3599, 100)
    postdb.ensure_grid_defs(get_engine_with_tables, {resolution_id: close})
    # outside COORD_ATOL, though within the default relative tolerance
    shifted = tiles.GridDef(-89.9, -179.9015, 0.1, 1799, 3599, 100)
    with pytest.raises(ValueError, match="differs from the stored one"):
        postdb.ensure_grid_defs(get_engine_with_tables, {resolution_id: shifted})


def test_insert_var_tiles_errors(get_engine_with_tables):
    ds = _fine_grid_ds()
    with pytest.raises(ValueError, match="Unknown aggregation"):
        postdb.insert_var_tiles(
            get_engine_with_tables, ds, "adults", {}, {}, {}, aggregation="sum"
        )
    with pytest.raises(ValueError, match="not found in var_type"):
        postdb.insert_var_tiles(get_engine_with_tables, ds, "adults", {}, {}, {})
    with pytest.raises(ValueError, match="No resolution group matches"):
        postdb.insert_var_tiles(
            get_engine_with_tables, ds, "adults", {}, {"adults": 1}, {0.2: 1}
        )


def test_insert_var_tiles_mask_errors(get_engine_with_tables):
    ds = _fine_grid_ds()
    # mask on another grid
    mask = xr.ones_like(ds["adults"], dtype=bool).isel(latitude=slice(0, 4))
    with pytest.raises(ValueError, match="different grid"):
        postdb.insert_var_tiles(
            get_engine_with_tables, ds, "adults", {}, {"adults": 1}, {0.1: 1}, mask=mask
        )
    # mask without the data's time points
    mask = xr.ones_like(ds["adults"], dtype=bool).assign_coords(
        time=[np.datetime64("2020-01-01", "ns"), np.datetime64("2020-02-01", "ns")]
    )
    with pytest.raises(ValueError, match="does not cover"):
        postdb.insert_var_tiles(
            get_engine_with_tables, ds, "adults", {}, {"adults": 1}, {0.1: 1}, mask=mask
        )


def test_queries_on_mean_var(get_engine_with_tables, get_session, tmp_path):
    ds = _fine_grid_ds()
    ds["adults"][{"latitude": slice(3, 5), "longitude": slice(0, 2)}] = 0.0
    mask = xr.ones_like(ds["adults"], dtype=bool)
    mask[{"longitude": 4, "latitude": 0}] = False  # (lat 0.4, lon 0.4)
    _, _ = _seed_tiles(get_session, get_engine_with_tables, ds, "mean", mask=mask)
    values = np.where(mask.values, ds["adults"].values, np.nan)  # lat descending
    plan = postdb._block_mean_plan(ds.latitude.values, ds.longitude.values, 0.2)
    coarse = postdb.block_mean(values, plan)  # lat descending: 0.4, 0.2, 0.0

    def cartesian(resolution, month=1, area=None):
        result = postdb.get_var_values_cartesian(
            get_session,
            time_point=(2023, month),
            grid_resolution=resolution,
            area=area,
            var_name="adults",
        )["latitude, longitude, var_value"]
        return {(la, lo): v for la, lo, v in result}

    # native: the masked cell is left out, zeros are returned as 0
    native = cartesian(0.1)
    assert len(native) == 24
    assert (0.4, 0.4) not in native
    assert native[(0.0, 0.0)] == 0.0
    np.testing.assert_allclose(native[(0.2, 0.3)], values[0, 2, 3])
    # ordered by latitude, then longitude
    assert list(native) == sorted(native)
    # coarse: area-weighted means
    means = cartesian(0.2, month=2)
    assert len(means) == 9
    np.testing.assert_allclose(means[(0.2, 0.2)], coarse[1, 1, 1], rtol=1e-6)
    np.testing.assert_allclose(means[(0.4, 0.4)], coarse[1, 0, 2], rtol=1e-6)
    assert means[(0.0, 0.0)] == 0.0
    # an area, boundaries included
    assert set(cartesian(0.1, area=(0.3, 0.1, 0.1, 0.2))) == {
        (la, lo) for la in (0.1, 0.2, 0.3) for lo in (0.1, 0.2)
    }
    # 0.25 is not a multiple of the grid step: no grid
    with pytest.raises(HTTPException, match="No grid points"):
        cartesian(0.25)

    # single values: native by default, coarse on request, no fallback to
    # another resolution
    get = postdb.get_var_value
    np.testing.assert_allclose(
        get(get_session, "adults", 0.2, 0.2, 2023, 2, 1), values[1, 2, 2]
    )
    np.testing.assert_allclose(
        get(get_session, "adults", 0.2, 0.2, 2023, 2, 1, resolution=0.2),
        coarse[1, 1, 1],
        rtol=1e-6,
    )
    assert get(get_session, "adults", 0.1, 0.1, 2023, 2, 1, resolution=0.2) is None
    assert get(get_session, "adults", 0.2, 0.2, 2023, 2, 1, resolution=0.25) is None
    assert get(get_session, "adults", 0.4, 0.4, 2023, 1, 1) is None  # masked
    assert get(get_session, "adults", 0.0, 0.0, 2023, 1, 1) == 0.0

    # the download has the native resolution, ascending coordinates
    out_file = tmp_path / "download.nc"
    postdb.get_var_values_cartesian_for_download(
        get_session,
        start_time_point=(2023, 1),
        end_time_point=(2023, 2),
        var_names=["adults"],
        netcdf_file=str(out_file),
    )
    with xr.open_dataset(out_file) as downloaded:
        np.testing.assert_allclose(downloaded.latitude, [0.0, 0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(
            downloaded["adults"].values, values[:, ::-1], rtol=1e-6
        )
        assert downloaded["adults"].attrs["unit"] == "-"
