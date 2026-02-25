from heiplanet_db import postgresql_database as db
from heiplanet_db import model_metadata_service as model_metadata
from fastapi import FastAPI, Depends, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from typing import Annotated, Union
import datetime
import dotenv
import os
import logging
import ipaddress
import json
from pydantic import BaseModel

logging.basicConfig(level=logging.DEBUG)

# get the db url from dotenv
dotenv.load_dotenv()
db_url = os.environ.get("DB_URL")
# with the initialization of the engine like this,
# we cannot use the same db mocking strategy as in the other tests
# because the engine is created before the tests are run.
if not db_url:
    raise ValueError("DB_URL environment variable is not set.")
engine = create_engine(db_url)

# get the IP address from the environment variable
ip_address = os.environ.get("IP_ADDRESS")
# check that the IP address is a string
if not isinstance(ip_address, str):
    raise ValueError("IP_ADDRESS environment variable must be a string.")
try:
    ipaddress.IPv4Address(ip_address)
except Exception:
    raise ValueError(
        f"IP_ADDRESS environment variable is not a valid IPv4 address: {ip_address}"
    )
allowed_origins = [
    f"http://{ip_address}",
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.initialize_database(db_url, replace=False)
    yield


def get_session():
    if engine is None:
        raise RuntimeError(
            "Database engine is not initialized. \
                           Please check the DB_URL environment variable."
        )
    session = db.create_session(engine)
    try:
        yield session
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(get_session)]


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # update this to https later when using ssl
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root(message: str = "Hello World") -> dict:
    return {"message": message}


@app.get("/models")
def get_models() -> dict:
    try:
        models = model_metadata.get_model_cards()
        return {"models": models}
    except model_metadata.ModelMetadataError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/db-status")
def db_status():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


class CartesianRequest(BaseModel):
    requested_time_point: datetime.date
    requested_variable_type: str | None = "R0"
    requested_area: tuple[float, float, float, float] = (180.0, -180.0, -90.0, 90.0)
    requested_grid_resolution: float = 0.2


@app.post("/cartesian")
def get_cartesian(
    session: Annotated[Session, Depends(get_session)],
    request: CartesianRequest,  # Now everything comes from the body
) -> Union[dict, None]:
    # the frontend will request a variable over all available lat, long values for that variable
    # the date input is 2016-01-01 (a date object)
    # the variable input is a matching string, ie "t2m" for temperature
    # the variable type name will be supplied via the model yaml files for each selected model
    # the requested area is a list of 4 floats, representing the bounding box
    # [N, W, S, E], i.e. [50, 8, 48, 10] for a box over Heidelberg
    # if no area is provided, default to the whole world

    if not isinstance(request.requested_time_point, datetime.date):
        return {"error": "Invalid date format. Use YYYY-MM-DD."}
    date_requested = (
        request.requested_time_point.year,
        request.requested_time_point.month,
    )
    var_name = request.requested_variable_type
    try:
        var_value = db.get_var_values_cartesian(
            session,
            time_point=date_requested,
            area=request.requested_area,
            grid_resolution=request.requested_grid_resolution,
            var_name=var_name,
        )
        return {"result": var_value}
    except Exception as e:
        return {"error": str(e)}


@app.get("/nuts_data")
def get_nuts_data(
    session: SessionDep,
    requested_time_point: datetime.date,
    requested_variable_type: str | None,
    requested_grid_resolution: str | None,
) -> Union[dict, None]:
    # the frontend will request a variable over all available lat, long values for that variable
    # the date input is 2016-01-01 (a date object)
    # the variable input is a matching string, ie "t2m" for temperature
    # the variable type will be supplied via the model yaml files for each selected model
    # the grid resolution is a string, either "NUTS0" for country-level data, or
    # "NUTS1" up to "NUTS3" for finer resolution data
    # if no grid resolution is provided, default to "NUTS2"
    if not isinstance(requested_time_point, datetime.date):
        return {"error": "Invalid date format. Use YYYY-MM-DD."}
    date_requested = (requested_time_point.year, requested_time_point.month)
    var_name = requested_variable_type
    try:
        grid_resolution = (
            requested_grid_resolution
            if requested_grid_resolution is not None
            else "NUTS2"
        )
        var_value = db.get_var_values_nuts(
            session,
            time_point=date_requested,
            var_name=var_name,
            grid_resolution=grid_resolution,
        )
        return {"result": var_value}
    except Exception as e:
        return {"error": str(e)}


@app.get("/nuts_regions")
def get_nuts_regions(
    grid_resolution: str | None = None,
):
    try:
        resolution = grid_resolution.upper() if grid_resolution else None
        geojson = db.get_nuts_regions_geojson(engine, resolution)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    suffix = f"_{resolution.lower()}" if resolution else ""
    filename = f"nuts_regions{suffix}.geojson"
    return Response(
        content=json.dumps(geojson),
        media_type="application/geo+json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
