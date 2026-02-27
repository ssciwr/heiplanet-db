from heiplanet_db import production as prod
import pytest
from pathlib import Path
from importlib import resources
from importlib.resources.abc import Traversable
from unittest.mock import patch
from sqlalchemy import text


@pytest.fixture(scope="module")
def production_config() -> Traversable:
    """Fixture to provide the path to the test configuration file."""
    dict_path = (
        resources.files("heiplanet_db") / "test" / "data" / "test_production_config.yml"
    )
    return dict_path


def test_read_production_config(production_config: Traversable):
    config_dict = prod.read_production_config()
    assert config_dict
    assert len(config_dict) == 2
    dict1 = config_dict["data_to_fetch"][0]
    assert dict1["var_name"][0]["name"] == "NUTS-definition"
    assert dict1["filename"] == "NUTS_RG_20M_2024_4326.shp.zip"
    assert dict1["host"] == "heibox"
    assert dict1["var_name"][0]["description"]
    dict2 = config_dict["data_to_fetch"][1]
    assert dict2["var_name"][0]["name"] == "R0"
    assert (
        dict2["filename"]
        == "era5_data_2025_07_2t_monthly_unicoords_adjlon_celsius_output_JModel_global.nc"
    )
    assert dict2["host"] == "heibox"
    # read another config file
    config_dict = prod.read_production_config(production_config)
    assert config_dict["data_to_fetch"][0]["var_name"][0]["name"] == "t2m"
    assert (
        config_dict["data_to_fetch"][0]["filename"]
        == "era5_data_2016_01_2t_tp_monthly_celsius_mm_resampled_0.5degree_trim.nc"
    )
    assert "local" in config_dict["data_to_fetch"][0]["host"]
    config_dict = prod.read_production_config(str(production_config))
    assert config_dict


def test_get_production_data(tmp_path: Path):
    filename = "test_download.md"
    filehash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    url = "https://heibox.uni-heidelberg.de/f/9641b5bf547b4d97995f/?dl=1"
    outputdir = tmp_path / "test_download"
    outputdir.mkdir(parents=True, exist_ok=True)
    completion_code = prod.get_production_data(url, filename, filehash, outputdir)
    assert completion_code == 0
    assert (outputdir / filename).is_file()


def test_create_directories(tmp_path: Path):
    prod.create_directories(str(tmp_path) + "/test")
    testdir = tmp_path / "test"
    assert testdir.exists()


def test_get_engine():
    # Avoid relying on a locally running production Postgres.
    # This test asserts the wiring to initialize_database, not connectivity.
    with (
        patch.dict("os.environ", {"DB_URL": "postgresql://example/db"}, clear=False),
        patch(
            "heiplanet_db.production.db.initialize_database",
            return_value=object(),
        ) as init_db,
    ):
        engine = prod.get_engine()
        assert engine is not None
        init_db.assert_called_once_with("postgresql://example/db", replace=False)


def test_insert_data(get_engine_with_tables, get_nuts_def_data, tmp_path):
    # here we test that the NUTS data can be inserted into the database
    # we could test this on the production db
    # but for the purpose of this test, we will use the test db
    shapefile_folder_path = tmp_path / "nuts_def.shp"
    gdf_nuts_data = get_nuts_def_data
    gdf_nuts_data.to_file(shapefile_folder_path, driver="ESRI Shapefile")
    completion_code = prod.insert_data(
        get_engine_with_tables, shapefile_folder_path.parents[0]
    )
    assert completion_code == 0


def test_get_var_types_from_config():
    # test that the variable types can be extracted from the config
    config_dict = {
        "data_to_fetch": [
            {
                "var_name": [
                    {"name": "t2m", "unit": "Celsius", "description": "2m temperature"}
                ],
            },
            {
                "var_name": [
                    {
                        "name": "total-population",
                        "unit": "1",
                        "description": "Total population",
                    }
                ],
            },
        ]
    }
    var_types = prod.get_var_types_from_config(config_dict["data_to_fetch"])
    assert len(var_types) == 2
    assert var_types[0]["name"] == "t2m"
    assert var_types[0]["unit"] == "Celsius"
    assert var_types[0]["description"] == "2m temperature"
    assert var_types[1]["name"] == "total-population"
    assert var_types[1]["unit"] == "1"
    assert var_types[1]["description"] == "Total population"
    # test that duplicates are removed
    config_dict = {
        "data_to_fetch": [
            {
                "var_name": [
                    {"name": "t2m", "unit": "Celsius", "description": "2m temperature"},
                    {"name": "t2m", "unit": "Celsius", "description": "2m temperature"},
                ],
            },
            {
                "var_name": [
                    {
                        "name": "total-population",
                        "unit": "1",
                        "description": "Total population",
                    },
                    {
                        "name": "total-population",
                        "unit": "1",
                        "description": "Total population",
                    },
                ],
            },
        ]
    }
    var_types = prod.get_var_types_from_config(config_dict["data_to_fetch"])
    assert len(var_types) == 2


def test_check_paths(tmp_path: Path):
    # Test that the check_paths function raises an error for None paths
    with pytest.raises(ValueError):
        prod.check_paths([None, None])

    # Test that the check_paths function raises an error for non-existent files
    with pytest.raises(FileNotFoundError):
        prod.check_paths([Path("non_existent_file.nc")])

    # Test that the check_paths function does not raise an error for valid paths
    valid_path = tmp_path / "test_data.nc"
    valid_path.touch()  # Create a dummy file for testing
    prod.check_paths([valid_path])
    valid_path.unlink()  # Clean up the dummy file


def test_get_engine_drop_tables_true_calls_initialize_database_replace_true(
    monkeypatch,
):
    sentinel_engine = object()
    monkeypatch.setenv("DB_URL", "postgresql://user:pass@host:5432/dbname")

    with patch(
        "heiplanet_db.production.db.initialize_database",
        return_value=sentinel_engine,
    ) as init_db:
        engine = prod.get_engine(drop_tables=True)

    assert engine is sentinel_engine
    init_db.assert_called_once_with(
        "postgresql://user:pass@host:5432/dbname",
        replace=True,
    )


def test_main_uses_explicit_config_path(tmp_path):
    # minimal shape needed to reach create_directories loop
    fake_config = {"datalake": {"bronze": str(tmp_path)}, "data_to_fetch": []}

    with (
        patch(
            "heiplanet_db.production.read_production_config", return_value=fake_config
        ) as read_cfg,
        patch(
            "heiplanet_db.production.create_directories",
            side_effect=RuntimeError("stop"),
        ) as _,
    ):
        with pytest.raises(RuntimeError, match="stop"):
            prod.main(config_path="my-config.yml")

    read_cfg.assert_called_once_with("my-config.yml")


@pytest.mark.skip(
    reason="This test requires a lot of resources and is not suitable for CI."
)
def test_main():
    """Test the main function to ensure it runs without errors."""
    # This test will not check the actual functionality but will ensure
    # that the main function can be called without raising exceptions.
    try:
        prod.main()
    except Exception as e:
        pytest.fail(f"Main function raised an exception: {e}")


def test_autovacuum_restoration_on_failure(get_engine_with_tables):
    engine = get_engine_with_tables

    # Mock insert_var_values to raise an exception
    with patch(
        "heiplanet_db.production.insert_var_values",
        side_effect=RuntimeError("Insertion failed"),
    ):
        with pytest.raises(RuntimeError, match="Insertion failed"):
            prod.load_data_with_optimization(engine, r0_path=Path("dummy.nc"))

    # Verify autovacuum is enabled for all tables
    with engine.connect() as conn:
        for table in ["grid_point", "time_point", "var_value", "var_value_nuts"]:
            result = conn.execute(
                text(f"SELECT reloptions FROM pg_class WHERE relname = '{table}'")
            )
            row = result.fetchone()
            # If autovacuum was disabled and NOT re-enabled, it would likely show up in reloptions.
            # Postgres doesn't always clear the option when set to true, it might show 'autovacuum_enabled=true'.
            # Or if it was reset, it might be None.
            # IMPORTANT: The code explicitly sets it to true.
            reloptions = row[0] if row else []
            if reloptions:
                opts = {k: v for k, v in (x.split("=") for x in reloptions)}
                assert opts.get("autovacuum_enabled") == "true", (
                    f"Autovacuum not enabled for {table}: {reloptions}"
                )


def test_autovacuum_restoration_on_success(get_engine_with_tables):
    engine = get_engine_with_tables

    # Mock inserts to do nothing
    with (
        patch("heiplanet_db.production.insert_var_values"),
        patch("heiplanet_db.production.insert_var_values_nuts"),
    ):
        prod.load_data_with_optimization(engine, r0_path=Path("dummy.nc"))

    # Verify autovacuum is enabled
    with engine.connect() as conn:
        for table in ["grid_point", "time_point", "var_value", "var_value_nuts"]:
            result = conn.execute(
                text(f"SELECT reloptions FROM pg_class WHERE relname = '{table}'")
            )
            row = result.fetchone()
            reloptions = row[0] if row else []
            if reloptions:
                opts = {k: v for k, v in (x.split("=") for x in reloptions)}
                assert opts.get("autovacuum_enabled") == "true", (
                    f"Autovacuum not enabled for {table}: {reloptions}"
                )
