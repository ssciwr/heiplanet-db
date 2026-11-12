from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text

from heiplanet_db import production as prod


@pytest.fixture(scope="module")
def production_config() -> Traversable:
    """Fixture to provide the path to the test configuration file."""
    dict_path = (
        resources.files("heiplanet_db") / "test" / "data" / "test_production_config.yml"
    )
    return dict_path


def test_read_production_config_file_not_found():
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        prod.read_production_config("nonexistent_config.yml")


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


def test_get_production_data_fetch_error(tmp_path: Path):
    outputdir = tmp_path / "out"
    outputdir.mkdir(parents=True, exist_ok=True)
    with (
        patch(
            "heiplanet_db.production.pooch.retrieve",
            side_effect=Exception("Connection refused"),
        ),
        pytest.raises(RuntimeError, match="Failed to fetch data from"),
    ):
        prod.get_production_data(
            url="https://example.com/missing",
            filename="missing.nc",
            filehash="abc123",
            outputdir=outputdir,
        )


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


def test_insert_data_invalid_path(get_engine_with_tables, tmp_path):
    with pytest.raises(ValueError, match="Shapefile folder path .* does not exist"):
        prod.insert_data(
            get_engine_with_tables,
            shapefiles_folder_path=tmp_path / "nonexistent_folder",
        )


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


def test_get_data_files_local(tmp_path: Path):
    config = {"datalake": {"bronze": str(tmp_path / "bronze")}}
    data = {
        "host": "local",
        "url": str(tmp_path / "local_file.nc"),
        "filename": "local_file.nc",
    }
    (tmp_path / "local_file.nc").touch()
    prod.get_data_files(data, config, "bronze")
    assert data["url"] == str((tmp_path / "local_file.nc").resolve())


def test_get_data_files_heibox(tmp_path: Path):
    config = {"datalake": {"bronze": str(tmp_path / "bronze")}}
    (tmp_path / "bronze").mkdir(parents=True)
    data = {
        "host": "heibox",
        "url": "https://heibox.example.com/f/x",
        "filename": "remote.nc",
        "filehash": "sha256:abc123",
    }
    with patch(
        "heiplanet_db.production.get_production_data",
        return_value=0,
    ) as get_data:
        prod.get_data_files(data, config, "bronze")
        get_data.assert_called_once()
        call_kw = get_data.call_args.kwargs
        assert call_kw["url"] == data["url"]
        assert call_kw["filename"] == data["filename"]
        assert call_kw["outputdir"] == tmp_path / "bronze"


def test_get_data_files_unknown_host():
    config = {"datalake": {"bronze": "/tmp/bronze"}}
    data = {"host": "unknown", "filename": "file.nc"}
    with pytest.raises(ValueError, match="Unknown host"):
        prod.get_data_files(data, config, "bronze")


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


def test_get_engine_raises_on_init_failure(monkeypatch):
    monkeypatch.setenv("DB_URL", "postgresql://bad/url")
    with (
        patch(
            "heiplanet_db.production.db.initialize_database",
            side_effect=Exception("connection refused"),
        ),
        pytest.raises(ValueError, match="Could not initialize engine"),
    ):
        prod.get_engine()


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


def test_reset_production_data_executes_truncate_statement():
    engine = MagicMock()
    conn = engine.begin.return_value.__enter__.return_value

    prod.reset_production_data(engine)

    engine.begin.assert_called_once()
    conn.execute.assert_called_once()
    sql_arg = conn.execute.call_args.args[0]
    # The SQLAlchemy text object should contain our TRUNCATE statement and key tables.
    sql_str = str(sql_arg)
    assert "TRUNCATE TABLE" in sql_str
    assert "grid_point_resolution" in sql_str
    assert "var_value_nuts" in sql_str
    assert "var_value" in sql_str
    assert "grid_point" in sql_str
    assert "time_point" in sql_str
    assert "var_type" in sql_str
    assert "resolution_group" in sql_str
    assert "nuts_def" in sql_str


def test_main_uses_config_from_env(tmp_path, monkeypatch):
    fake_config = {"datalake": {"bronze": str(tmp_path)}, "data_to_fetch": []}
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "env_config.yml"))
    (tmp_path / "env_config.yml").touch()
    with (
        patch(
            "heiplanet_db.production.read_production_config",
            return_value=fake_config,
        ) as read_cfg,
        patch(
            "heiplanet_db.production.create_directories",
            side_effect=RuntimeError("stop"),
        ),
        pytest.raises(RuntimeError, match="stop"),
    ):
        prod.main()
    read_cfg.assert_called_once_with(str(tmp_path / "env_config.yml"))


def test_main_value_error_when_no_shapefile(tmp_path, monkeypatch):
    fake_config = {
        "datalake": {"bronze": str(tmp_path), "silver": str(tmp_path / "silver")},
        "data_to_fetch": [
            {
                "var_name": [{"type": "R0", "name": "t2m", "level": "bronze"}],
                "filename": "r0.nc",
            },
        ],
    }
    (tmp_path / "r0.nc").touch()
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    with (
        patch(
            "heiplanet_db.production.read_production_config", return_value=fake_config
        ),
        patch("heiplanet_db.production.create_directories"),
        patch("heiplanet_db.production.get_data_files"),
        patch("heiplanet_db.production.get_engine"),
        pytest.raises(ValueError, match="Shapefile path could not be generated"),
    ):
        prod.main()


def test_main_uses_explicit_config_path(tmp_path):
    fake_config = {"datalake": {"bronze": str(tmp_path)}, "data_to_fetch": []}
    with (
        patch(
            "heiplanet_db.production.read_production_config", return_value=fake_config
        ) as read_cfg,
        patch(
            "heiplanet_db.production.create_directories",
            side_effect=RuntimeError("stop"),
        ),
        pytest.raises(RuntimeError, match="stop"),
    ):
        prod.main(config_path="my-config.yml")
    read_cfg.assert_called_once_with("my-config.yml")


def _make_minimal_definition_config(tmp_path: Path) -> dict:
    bronze_dir = tmp_path / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    # Dummy zip file that will be "opened" by the patched ZipFile
    shapefile_zip = bronze_dir / "nuts_def.zip"
    shapefile_zip.touch()
    return {
        "datalake": {"bronze": str(bronze_dir)},
        "data_to_fetch": [
            {
                "var_name": [
                    {
                        "type": "definition",
                        "name": "NUTS-definition",
                        "level": "bronze",
                    }
                ],
                "filename": shapefile_zip.name,
                "host": "local",
            }
        ],
    }


def test_main_calls_reset_production_data_when_drop_tables_false(tmp_path, monkeypatch):
    fake_config = _make_minimal_definition_config(tmp_path)
    sentinel_engine = object()

    with (
        patch(
            "heiplanet_db.production.read_production_config",
            return_value=fake_config,
        ),
        patch("heiplanet_db.production.create_directories"),
        patch("heiplanet_db.production.get_data_files"),
        patch(
            "heiplanet_db.production.zipfile.ZipFile"
        ) as zip_cls,  # avoid working with a real zip
        patch(
            "heiplanet_db.production.get_engine",
            return_value=sentinel_engine,
        ),
        patch("heiplanet_db.production.insert_data"),
        patch("heiplanet_db.production.get_var_types_from_config", return_value=[]),
        patch("heiplanet_db.production.db.insert_var_types"),
        patch("heiplanet_db.production.db.insert_resolution_groups"),
        patch("heiplanet_db.production.load_data_with_optimization"),
        patch("heiplanet_db.production.reset_production_data") as reset_data,
    ):
        # Make the ZipFile context manager a no-op
        zip_inst = zip_cls.return_value.__enter__.return_value
        zip_inst.extractall.return_value = None

        prod.main(drop_tables=False, config_path="ignored.yml")

    reset_data.assert_called_once_with(sentinel_engine)


def test_main_does_not_call_reset_production_data_when_drop_tables_true(
    tmp_path, monkeypatch
):
    fake_config = _make_minimal_definition_config(tmp_path)
    sentinel_engine = object()

    with (
        patch(
            "heiplanet_db.production.read_production_config",
            return_value=fake_config,
        ),
        patch("heiplanet_db.production.create_directories"),
        patch("heiplanet_db.production.get_data_files"),
        patch("heiplanet_db.production.zipfile.ZipFile") as zip_cls,
        patch(
            "heiplanet_db.production.get_engine",
            return_value=sentinel_engine,
        ),
        patch("heiplanet_db.production.insert_data"),
        patch("heiplanet_db.production.get_var_types_from_config", return_value=[]),
        patch("heiplanet_db.production.db.insert_var_types"),
        patch("heiplanet_db.production.db.insert_resolution_groups"),
        patch("heiplanet_db.production.load_data_with_optimization"),
        patch("heiplanet_db.production.reset_production_data") as reset_data,
    ):
        zip_inst = zip_cls.return_value.__enter__.return_value
        zip_inst.extractall.return_value = None

        prod.main(drop_tables=True, config_path="ignored.yml")

    reset_data.assert_not_called()


@pytest.mark.skip(
    reason="This test requires a lot of resources and is not suitable for CI."
)
def test_main():
    """Test the main function to ensure it runs without errors."""
    # This test will not check the actual functionality but will ensure
    # that the main function can be called without raising exceptions.
    try:
        prod.main()
    except Exception as e:  # noqa: BLE001 - smoke test must catch any failure
        pytest.fail(f"Main function raised an exception: {e}")


def test_autovacuum_restoration_on_failure(get_engine_with_tables):
    engine = get_engine_with_tables

    # Mock insert_grid_var_values to raise an exception
    with (
        patch(
            "heiplanet_db.production.insert_grid_var_values",
            side_effect=RuntimeError("Insertion failed"),
        ),
        pytest.raises(RuntimeError, match="Insertion failed"),
    ):
        prod.load_data_with_optimization(
            engine, grid_vars=[prod.GridVar("R0", [Path("dummy.nc")])]
        )

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
    grid_vars = [prod.GridVar("R0", [Path("dummy.nc")])]
    with (
        patch("heiplanet_db.production.insert_grid_var_values") as insert_grid,
        patch("heiplanet_db.production.insert_nuts_var_values") as insert_nuts,
    ):
        prod.load_data_with_optimization(engine, grid_vars=grid_vars)
    insert_grid.assert_called_once_with(engine, grid_vars)
    insert_nuts.assert_not_called()


def test_load_data_with_optimization_nuts_only(get_engine_with_tables):
    engine = get_engine_with_tables
    nuts_vars = [prod.NutsVar("R0", Path("dummy_nuts.nc"))]
    with (
        patch("heiplanet_db.production.insert_grid_var_values") as insert_grid,
        patch("heiplanet_db.production.insert_nuts_var_values") as insert_nuts,
    ):
        prod.load_data_with_optimization(engine, grid_vars=None, nuts_vars=nuts_vars)
    insert_grid.assert_not_called()
    insert_nuts.assert_called_once_with(engine, nuts_vars)

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


# --- config handling for several variables and files ---


def test_get_var_types_from_config_aggregation():
    config = [
        {"var_name": [{"name": "R0", "unit": "1", "description": "r0"}]},
        {
            "var_name": [
                {
                    "name": "adults",
                    "unit": "u",
                    "description": "grid",
                    "aggregation": "mean",
                }
            ]
        },
        # a later NUTS entry without aggregation keeps the grid entry's setting
        {"var_name": [{"name": "adults", "unit": "u", "description": "nuts"}]},
    ]
    var_types = prod.get_var_types_from_config(config)
    assert [vt["name"] for vt in var_types] == ["R0", "adults"]
    assert var_types[0]["aggregation"] == "subsample"
    assert var_types[1]["aggregation"] == "mean"
    assert var_types[1]["description"] == "grid"
    config[1]["var_name"][0]["aggregation"] = "sum"
    with pytest.raises(ValueError, match="Unknown aggregation"):
        prod.get_var_types_from_config(config)


def test_fetch_files_local(tmp_path: Path):
    data_dir = tmp_path / "gold"
    data_dir.mkdir()
    for name in ["a.nc", "b.nc"]:
        (data_dir / name).touch()
    config = {"datalake": {"gold": str(tmp_path / "lake")}}
    # url is the folder holding a list of files
    data = {"host": "local", "url": str(data_dir), "filename": ["a.nc", "b.nc"]}
    assert prod.fetch_files(data, config, "gold") == [
        (data_dir / "a.nc").resolve(),
        (data_dir / "b.nc").resolve(),
    ]
    # url is the file itself
    data = {"host": "local", "url": str(data_dir / "a.nc"), "filename": "x.nc"}
    assert prod.fetch_files(data, config, "gold") == [(data_dir / "a.nc").resolve()]
    # no url: read from the data lake
    config = {"datalake": {"gold": str(data_dir)}}
    data = {"host": "local", "filename": "b.nc"}
    assert prod.fetch_files(data, config, "gold") == [(data_dir / "b.nc").resolve()]
    # missing files are reported
    data = {"host": "local", "url": str(data_dir), "filename": ["a.nc", "c.nc"]}
    with pytest.raises(FileNotFoundError, match="c.nc"):
        prod.fetch_files(data, config, "gold")


def test_fetch_files_heibox_list(tmp_path: Path):
    config = {"datalake": {"gold": str(tmp_path)}}
    (tmp_path / "a.nc").touch()  # already downloaded
    data = {
        "host": "heibox",
        "filename": ["a.nc", "b.nc"],
        "url": ["https://example.com/a", "https://example.com/b"],
        "filehash": ["ha", "hb"],
    }
    with patch("heiplanet_db.production.get_data_files") as get_files:
        paths = prod.fetch_files(data, config, "gold")
    assert paths == [(tmp_path / "a.nc").resolve(), (tmp_path / "b.nc").resolve()]
    get_files.assert_called_once()
    entry = get_files.call_args.args[0]
    assert entry["filename"] == "b.nc"
    assert entry["url"] == "https://example.com/b"
    assert entry["filehash"] == "hb"
    data["filehash"] = ["ha"]
    with pytest.raises(ValueError, match="filehash must list one entry per file"):
        prod.fetch_files(data, config, "gold")


def _main_with_config(fake_config, tmp_path):
    with (
        patch(
            "heiplanet_db.production.read_production_config",
            return_value=fake_config,
        ),
        patch("heiplanet_db.production.create_directories"),
        patch("heiplanet_db.production.zipfile.ZipFile"),
        patch("heiplanet_db.production.get_engine", return_value="engine"),
        patch("heiplanet_db.production.reset_production_data"),
        patch("heiplanet_db.production.insert_data"),
        patch("heiplanet_db.production.get_var_types_from_config", return_value=[]),
        patch("heiplanet_db.production.db.insert_var_types"),
        patch("heiplanet_db.production.db.insert_resolution_groups"),
        patch("heiplanet_db.production.load_data_with_optimization") as load,
    ):
        prod.main(config_path="ignored.yml")
    return load


def test_main_collects_grid_and_nuts_vars(tmp_path: Path):
    fake_config = _make_minimal_definition_config(tmp_path)
    for name in ["r0.nc", "p1.nc", "p2.nc", "nuts.nc"]:
        (tmp_path / name).touch()
    var = {"unit": "1", "description": "d"}
    fake_config["data_to_fetch"] += [
        {
            "var_name": [{"type": "R0", "name": "R0", **var}],
            "filename": "r0.nc",
            "host": "local",
            "url": str(tmp_path),
        },
        {
            "var_name": [
                {
                    "type": "grid",
                    "name": "adults",
                    "aggregation": "mean",
                    "mask_by": "R0",
                    **var,
                }
            ],
            "filename": ["p1.nc", "p2.nc"],
            "host": "local",
            "url": str(tmp_path),
        },
        {
            "var_name": [
                {"type": "R0_nuts", "name": "R0", **var},
                {"type": "nuts", "name": "adults", **var},
            ],
            "filename": "nuts.nc",
            "host": "local",
            "url": str(tmp_path),
        },
    ]
    load = _main_with_config(fake_config, tmp_path)
    kwargs = load.call_args.kwargs
    assert kwargs["grid_vars"] == [
        prod.GridVar("R0", [(tmp_path / "r0.nc").resolve()]),
        prod.GridVar(
            "adults",
            [(tmp_path / "p1.nc").resolve(), (tmp_path / "p2.nc").resolve()],
            aggregation="mean",
            mask_by="R0",
        ),
    ]
    nuts_path = (tmp_path / "nuts.nc").resolve()
    assert kwargs["nuts_vars"] == [
        prod.NutsVar("R0", nuts_path),
        prod.NutsVar("adults", nuts_path),
    ]


def test_main_rejects_unknown_type_and_multi_file_nuts(tmp_path: Path):
    fake_config = _make_minimal_definition_config(tmp_path)
    (tmp_path / "a.nc").touch()
    (tmp_path / "b.nc").touch()
    fake_config["data_to_fetch"].append(
        {
            "var_name": [{"type": "other", "name": "x"}],
            "filename": "a.nc",
            "host": "local",
            "url": str(tmp_path),
        }
    )
    with pytest.raises(ValueError, match="Unknown type other"):
        _main_with_config(fake_config, tmp_path)
    fake_config["data_to_fetch"][-1] = {
        "var_name": [{"type": "nuts", "name": "x"}],
        "filename": ["a.nc", "b.nc"],
        "host": "local",
        "url": str(tmp_path),
    }
    with pytest.raises(ValueError, match="must be a single file"):
        _main_with_config(fake_config, tmp_path)


def test_production_config_2014_2018():
    config = prod.read_production_config(
        resources.files("heiplanet_db") / "data" / "production_config_2014-2018.yml"
    )
    var_types = {
        vt["name"]: vt for vt in prod.get_var_types_from_config(config["data_to_fetch"])
    }
    assert var_types["R0"]["aggregation"] == "subsample"
    assert var_types["adults"]["aggregation"] == "mean"
    assert var_types["adults"]["unit"] == "mosquitoes per 625 km²"
    pmodel = config["data_to_fetch"][2]
    assert len(pmodel["filename"]) == 5
    assert pmodel["var_name"][0]["mask_by"] == "R0"


# --- loading gridded variables ---


def _write_grid_files(tmp_path: Path):
    """R0 on a 5x5 0.1 degree grid with NaN 'ocean', and two yearly files of
    adults stored as (longitude, latitude, time) with zeros on the ocean."""
    import numpy as np
    import xarray as xr

    lats = np.round(np.arange(0.4, -0.05, -0.1), 1)
    lons = np.round(np.arange(0.0, 0.45, 0.1), 1)
    times = np.array(
        ["2023-01-01", "2023-02-01", "2024-01-01", "2024-02-01"],
        dtype="datetime64[ns]",
    )
    r0 = np.ones((4, 5, 5))
    r0[:, :, 0] = np.nan  # ocean in the first column
    xr.Dataset(
        {"R0": (("time", "latitude", "longitude"), r0)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    ).to_netcdf(tmp_path / "r0.nc")
    paths = []
    for year, sl in ((2023, slice(0, 2)), (2024, slice(2, 4))):
        adults = np.full((4, 5, 5), 2.0)[sl]
        adults[:, :, 0] = 0.0
        ds = xr.Dataset(
            {"adults": (("time", "latitude", "longitude"), adults)},
            coords={"time": times[sl], "latitude": lats, "longitude": lons},
        ).transpose("longitude", "latitude", "time")
        path = tmp_path / f"adults_{year}.nc"
        ds.to_netcdf(path)
        paths.append(path)
    return tmp_path / "r0.nc", paths


def test_open_grid_dataset_multi_file(tmp_path: Path):
    _, paths = _write_grid_files(tmp_path)
    ds = prod.open_grid_dataset(prod.GridVar("adults", paths))
    assert ds["adults"].dims == ("time", "latitude", "longitude")
    assert ds.sizes["time"] == 4
    with pytest.raises(ValueError, match="not found"):
        prod.open_grid_dataset(prod.GridVar("eggs", paths))


def test_insert_grid_var_values_mask_by_unknown(tmp_path: Path):
    with pytest.raises(ValueError, match="masked by R0"):
        prod.insert_grid_var_values(
            engine=None,
            grid_vars=[prod.GridVar("adults", [], aggregation="mean", mask_by="R0")],
        )


def test_insert_grid_var_values_different_grids(tmp_path: Path):
    r0_path, paths = _write_grid_files(tmp_path)
    import xarray as xr

    with xr.open_dataset(paths[0]) as ds:
        ds.isel(latitude=slice(0, 4)).to_netcdf(tmp_path / "small.nc")
    with pytest.raises(ValueError, match="different grid"):
        prod.insert_grid_var_values(
            engine=None,
            grid_vars=[
                prod.GridVar("R0", [r0_path]),
                prod.GridVar("adults", [tmp_path / "small.nc"]),
            ],
        )


def test_insert_grid_var_values_end_to_end(get_engine_with_tables, tmp_path):
    import numpy as np

    from heiplanet_db import postgresql_database as db

    engine = get_engine_with_tables
    r0_path, paths = _write_grid_files(tmp_path)
    grid_vars = [
        prod.GridVar("R0", [r0_path]),
        prod.GridVar("adults", paths, aggregation="mean", mask_by="R0"),
    ]
    session = db.create_session(engine)
    db.insert_var_types(
        session,
        [
            {"name": "R0", "unit": "1", "description": "r0"},
            {
                "name": "adults",
                "unit": "mosquitoes per 625 km²",
                "description": "adults",
                "aggregation": "mean",
            },
        ],
    )
    db.insert_resolution_groups(session, resolutions=np.array([0.1, 0.2]))
    session.close()

    prod.insert_grid_var_values(engine, grid_vars)

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT vt.name, rg.resolution, COUNT(*), MIN(vv.value), "
                "MAX(vv.value) FROM var_value vv "
                "JOIN var_type vt ON vt.id = vv.var_id "
                "LEFT JOIN resolution_group rg ON rg.id = vv.resolution_id "
                "GROUP BY vt.name, rg.resolution ORDER BY vt.name, rg.resolution"
            )
        ).all()
    summary = {(name, None if res is None else float(res)): r for name, res, *r in rows}
    # R0: 4 months x 20 land cells, one row each (no resolution)
    assert summary[("R0", None)] == [80, 1.0, 1.0]
    # adults at 0.1: ocean column masked out, so the zeros are gone
    assert summary[("adults", 0.1)] == [80, 2.0, 2.0]
    # adults at 0.2: 3x3 coarse cells; the column at lon 0.0 still has land
    # (lon 0.1) in its window, so all 9 exist and the mean ignores the ocean
    assert summary[("adults", 0.2)] == [36, 2.0, 2.0]
