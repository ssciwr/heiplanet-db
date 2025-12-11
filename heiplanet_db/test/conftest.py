import pytest
from heiplanet_db import postgresql_database as postdb
from testcontainers.postgres import PostgresContainer
from sqlalchemy import create_engine, text
from sqlalchemy.orm.session import sessionmaker
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd
import xarray as xr


# for local docker desktop,
# environ["DOCKER_HOST"] is "unix:///home/[user]/.docker/desktop/docker.sock"


@pytest.fixture(scope="module")
def get_docker_image():
    return "postgis/postgis:17-3.5"


@pytest.fixture(scope="module")
def get_engine_with_tables(get_docker_image):
    with PostgresContainer(get_docker_image) as postgres:
        engine = create_engine(postgres.get_connection_url())
        postdb.Base.metadata.create_all(engine)
        yield engine
        postdb.Base.metadata.drop_all(engine)


@pytest.fixture(scope="module")
def get_engine_without_tables(get_docker_image):
    with PostgresContainer(get_docker_image) as postgres:
        engine = create_engine(postgres.get_connection_url())
        yield engine
        postdb.Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def get_session(get_engine_with_tables):
    connection = get_engine_with_tables.connect()
    session_class = sessionmaker(bind=connection)
    session = session_class()

    yield session

    # tear down
    session.close()


def retrieve_id_maps(session, dataset, var_type_data, to_monthly=False):
    # insert data into the database
    postdb.insert_grid_points(
        session, dataset.latitude.values, dataset.longitude.values
    )
    postdb.insert_time_points(session, [(dataset.time.values, to_monthly)])
    postdb.insert_var_types(session, var_type_data)

    # get the id maps
    grid_id_map, time_id_map, var_id_map = postdb.get_id_maps(session)

    return grid_id_map, time_id_map, var_id_map


@pytest.fixture(scope="function")
def insert_data(get_session, get_dataset, get_engine_with_tables):
    var_type_data = [
        {
            "name": "t2m",
            "unit": "K",
            "description": "2m temperature",
        }
    ]
    postdb.insert_var_types(get_session, var_type_data)
    # insert grid points
    postdb.insert_grid_points(
        get_session, get_dataset.latitude.values, get_dataset.longitude.values
    )
    # insert time points
    postdb.insert_time_points(get_session, [(get_dataset.time.values, False)])

    # get the id maps
    grid_id_map, time_id_map, var_id_map = retrieve_id_maps(
        get_session, get_dataset, var_type_data
    )

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

    # insert resolution values
    postdb.insert_resolution_groups(get_session, resolutions=np.array([0.1, 0.2]))
    postdb.assign_grid_resolution_group_to_grid_point(get_session)

    yield get_session
    # clean up
    get_session.execute(text("TRUNCATE TABLE var_value RESTART IDENTITY CASCADE"))
    get_session.execute(text("TRUNCATE TABLE var_type RESTART IDENTITY CASCADE"))
    get_session.execute(text("TRUNCATE TABLE time_point RESTART IDENTITY CASCADE"))
    get_session.execute(text("TRUNCATE TABLE grid_point RESTART IDENTITY CASCADE"))
    get_session.execute(
        text("TRUNCATE TABLE resolution_group RESTART IDENTITY CASCADE")
    )
    get_session.commit()


def cleanup(engine):
    # drop all tables
    postdb.Base.metadata.drop_all(engine)


@pytest.fixture
def get_nuts_def_data():
    # create a sample NUTS shapefile data
    polygon1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    polygon2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    polygon3 = Polygon([(50, 49.5), (50.5, 49.5), (50.5, 50), (50, 50)])
    gdf = gpd.GeoDataFrame(
        {
            "NUTS_ID": ["DE11", "DE22", "DE501"],
            "LEVL_CODE": [1, 2, 3],
            "CNTR_CODE": [1, 1, 1],
            "NAME_LATN": ["Test NUTS", "Test NUTS2", "Test NUTS3"],
            "NUTS_NAME": ["Test NUTS", "Test NUTS2", "Test NUTS3"],
            "MOUNT_TYPE": [0.0, 0.0, 0.0],
            "URBN_TYPE": [1.0, 1.0, 1.0],
            "COAST_TYPE": [1.0, 1.0, 1.0],
        },
        geometry=[polygon1, polygon2, polygon3],
        crs="EPSG:4326",
    )
    return gdf


@pytest.fixture
def get_var_type_list():
    return [
        {
            "name": "test_var",
            "unit": "1",
            "description": "Test variable",
        },
        {
            "name": "test_var2",
            "unit": "1",
            "description": "Test variable 2",
        },
    ]


@pytest.fixture
def get_time_point_lists():
    return [
        (
            np.array(
                [
                    np.datetime64("2024-01-01T00:00:00.000000000"),
                    np.datetime64("2023-01-01 00:00:00"),
                ]
            ),
            True,  # yearly
        ),
        (
            np.array(
                [
                    np.datetime64("2024-01-01T00:00:00.000000000"),
                    np.datetime64("2024-02-01 00:00:00"),
                    np.datetime64("2024-03-01 00:00:00"),
                ]
            ),
            False,  # monthly
        ),
    ]


@pytest.fixture()
def get_dataset():
    rng = np.random.default_rng(42)
    data = rng.random((2, 3, 2)) * 1000 + 273.15
    data_array = xr.DataArray(
        data,
        dims=["latitude", "longitude", "time"],
        coords={
            "latitude": [10.1, 10.2],
            "longitude": [10.1, 10.2, 10.3],
            "time": [
                np.datetime64("2023-01-01", "ns"),
                np.datetime64("2024-01-01", "ns"),
            ],
        },
    )
    dataset = xr.Dataset({"t2m": data_array})
    return dataset


@pytest.fixture
def get_varnuts_dataset():
    rng = np.random.default_rng(42)
    data = rng.random((2, 2)) * 200
    data_array = xr.DataArray(
        data,
        dims=["NUTS_ID", "time"],
        coords={
            "NUTS_ID": ["DE11", "DE22", "DE501"],
            "time": [
                np.datetime64("2023-01-01", "ns"),
                np.datetime64("2024-01-01", "ns"),
            ],
        },
    )
    dataset = xr.Dataset({"t2m_mean": data_array})
    return dataset
