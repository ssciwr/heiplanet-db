from heiplanet_db import production as prod
import pytest
from pathlib import Path
from importlib import resources
from importlib.resources.abc import Traversable


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
        == "era5_data_2025_07_2t_monthly_unicoords_adjlon_celsius_silver_output_JModel_global_ts20260206-161638_hssclap09.nc"
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
    # test that the engine can be created with production db
    engine = prod.get_engine()
    assert engine is not None


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
