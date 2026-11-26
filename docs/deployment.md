---
title: heiplanet platform deployment
hide:
- navigation
---

# Deployment of the Database, API and Frontend

The Heiplanet platform is deployed using [Docker Compose](https://docs.docker.com/compose/), which orchestrates multiple containerized services to create a complete surveillance system. Access to GitHub Container Registry (GHCR) images requires appropriate GitHub permissions and authentication tokens.

## System Architecture

The deployment consists of three main containerized services:

### 1. PostgreSQL/PostGIS Database
- **Image:** `postgis/postgis:17-3.5` (public image)
- **Purpose:** Stores all spatial and temporal disease surveillance data
- **Features:**
  - PostgreSQL 17 with PostGIS 3.5 for advanced geospatial queries
  - NUTS region geometries and definitions
  - Time-series disease transmission data (R0 values)
  - Efficient spatial indexing for fast geographic queries

### 2. Web Frontend
- **Image:** `ghcr.io/ssciwr/heiplanet-frontend:<tag>`
- **Purpose:** Interactive web interface for data visualization
- **Features:**
  - Geographic map-based visualization
  - Temporal data exploration
  - Regional comparison tools
  - Accessible via web browser on port 80

The `<tag>` can be a specific version number, `latest` for the most recent release, or a branch name for development versions.

### 3. Python Backend
- **Image:** `ghcr.io/ssciwr/heiplanet-db:<tag>` or locally built
- **Purpose:** Multi-function backend service providing:
  - **Database ORM**: SQLAlchemy-based object-relational mapping
  - **REST API**: FastAPI service processing frontend requests
  - **Data Pipeline**: ETL (Extract, Transform, Load) processes for ingesting surveillance data
  - **Data Validation**: Quality checks and integrity verification

The `<tag>` can be a version number, `latest`, or a branch name. Using a locally built image allows customization of the data configuration, enabling you to modify which datasets are ingested or add new data sources.


## Development Environment

The development environment allows you to run the complete Heiplanet platform locally for testing, development, and experimentation with test datasets.

### Environment Configuration

Create a `.env` file in the `heiplanet-db` root directory with the following environment variables:

```bash
# Database credentials
POSTGRES_USER=<user>
POSTGRES_PASSWORD=<password>
POSTGRES_DB=<db-name>

# Backend database connection
DB_USER=<user>
DB_PASSWORD=<password>
DB_HOST=db
DB_PORT=5432
DB_URL=postgresql://<user>:<password>@db:5432/<db-name>

# Startup configuration
WAIT_FOR_DB=true

# CORS configuration (for production deployment)
IP_ADDRESS=0.0.0.0
```

**Configuration notes:**
- Replace `<user>`, `<password>`, and `<db-name>` with your chosen credentials
  - Example: `POSTGRES_USER=heiplanet`, `POSTGRES_PASSWORD=secure_password`, `POSTGRES_DB=heiplanet_dev`
- `DB_HOST=db` refers to the database container name in the Docker network
- `WAIT_FOR_DB=true` ensures the backend waits for the database to be ready before starting
- `IP_ADDRESS` configures Cross-Origin Resource Sharing (CORS) for web requests
  - For local development: leave as `0.0.0.0` or omit
  - For production: set to your server's public IP address

### Starting the Development Environment

#### Step 1: Initialize Database with Test Data

Build the database tables and populate with test/development data:

```bash
docker compose up --abort-on-container-exit production-runner
```

This command:
- Creates the database schema (tables, indexes, constraints)
- Downloads test datasets defined in the configuration
- Processes data through the bronze/silver/gold pipeline
- Inserts data into the PostgreSQL database
- Exits automatically when data loading completes

**Note:** The first run will take several minutes as it downloads and processes data files.

#### Step 2: Start API and Frontend Services

Once data loading is complete, start the application services:

```bash
docker compose up api
```

This starts:
- **FastAPI backend** (internal port 8000)
- **Frontend web application** (port 80)
- **Database server** (internal to Docker network)

#### Accessing the Application

Open your web browser and navigate to:
- **Frontend:** `http://localhost:80` or `http://127.0.0.1:80`
- The API is accessible to the frontend through the internal Docker network

### Advanced Development Options

#### Direct API Access

To test the API directly (useful for debugging or development), expose port 8000 by modifying `docker-compose.yaml`:

```yaml
services:
  api:
    ports:
      - "8000:8000"  # Add this line
```

After making this change, the API documentation will be available at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

#### Direct Database Access

Similarly, you can expose the database port for external database clients:

```yaml
services:
  db:
    ports:
      - "5432:5432"  # Add this line
```

This allows connecting with tools like pgAdmin, DBeaver, or psql:
```bash
psql -h localhost -p 5432 -U <user> -d <db-name>
```

### Alternative: Minimal Container Development

If you lack GitHub permissions for GHCR access or want a simpler setup with just the database and API (without the frontend), follow this standalone procedure.

This approach manually creates and links containers instead of using Docker Compose.

**Prerequisites:** Navigate to the `heiplanet-db/` directory

#### Step 1: Start PostgreSQL/PostGIS Database

Create and start a database container:

```bash
docker run -d \
  --name postgres_heiplanet \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=heiplanet_db \
  postgis/postgis:17-3.5
```

This creates a database accessible at `localhost:5432` with:
- Username: `postgres`
- Password: `postgres`
- Database name: `heiplanet_db`

#### Step 2: Build and Start API Container

Build the backend image and start the API:

```bash
# Build the image locally
docker build -t heiplanet-db .

# Start the API container linked to the database
docker run -d \
  --name heiplanet_api \
  -p 8000:8000 \
  --link postgres_heiplanet:db \
  -e IP_ADDRESS=0.0.0.0 \
  -e DB_URL=postgresql+psycopg2://postgres:postgres@db:5432/heiplanet_db \
  heiplanet-db
```


### Local run without docker compose
To run with a local Python process and only PostGIS in Docker, use the following workflow from the `heiplanet-db/` root directory.

##### 1. Configure `.env`
Create/update `.env` with:
```
POSTGRES_USER=postgres
POSTGRES_PASSWORD=heiplanet-db-secret-name
POSTGRES_DB=heiplanet_db

DB_USER=postgres
DB_PASSWORD=heiplanet-db-secret-name
DB_HOST=127.0.0.1
DB_PORT=5432
DB_URL=postgresql://postgres:heiplanet-db-secret-name@127.0.0.1:5432/heiplanet_db
WAIT_FOR_DB=true
IP_ADDRESS=127.0.0.1
BATCH_SIZE=10000
MAX_WORKERS=4
VAR_TIME_CHUNK=6
VAR_LAT_CHUNK=45
VAR_LON_CHUNK=90
```

##### 2. Start PostgreSQL/PostGIS
```
docker run --name my-postgres \
  -e POSTGRES_USER=$POSTGRES_USER \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  -e POSTGRES_DB=$POSTGRES_DB \
  -p 5432:5432 \
  --shm-size=512mb \
  -d postgis/postgis:17-3.5
```

##### 3. Populate database tables
```
python heiplanet_db/production.py
```

#### Alternative 3 when using docker compose: Populate Database with Test Data

Run the data ingestion script inside the API container:

```bash
docker exec -it heiplanet_api python3 /heiplanet_db/production.py
```
##### 4. Run the API locally
```
cd heiplanet_db
fastapi dev
```

With the currently reduced input files (Grid and NUTS only), database setup is typically fast.
The API will be accessible at `http://localhost:8000`


This processes and loads test data into the database. Depending on the configuration file used, this may take several minutes.

#### Accessing the Services

- **API Documentation:** `http://localhost:8000/docs`
- **Database:** `localhost:5432` (connect with any PostgreSQL client)

#### Stopping and Cleaning Up

```bash
# Stop containers
docker stop heiplanet_api postgres_heiplanet

# Remove containers
docker rm heiplanet_api postgres_heiplanet

# Remove volumes (optional - deletes all data)
docker volume prune
```

### Building the Image Locally

Building a local Docker image is necessary when:
- Testing a custom data configuration file
- Modifying the data ingestion pipeline
- Developing new features for the backend
- Working offline without GHCR access

#### Build the Image

From the `heiplanet-db/` root directory:

```bash
docker build -t heiplanet-db .
```

`.dockerignore` keeps `.env` files, local data folders (`.data_heiplanet_db`), tests and docs out of the image, so a locally built image contains no credentials and stays small (about 0.85 GB).

This creates a local image tagged as `heiplanet-db` containing:
- Python backend application
- Data processing pipeline
- Configuration files
- Database ORM and API code

#### Using the Local Image with Docker Compose

**Option 1: Modify docker-compose.yaml**

Change the image reference in `docker-compose.yaml`:

```yaml
# Before
image: ghcr.io/ssciwr/heiplanet-db:latest

# After
image: heiplanet-db
```

**Option 2: Use Build Configuration**

Alternatively, uncomment the `build:` lines in `docker-compose.yaml` to force local builds:

```yaml
services:
  api:
    build: .
    # image: ghcr.io/ssciwr/heiplanet-db:latest
```

Then run:
```bash
docker compose build
docker compose up
```

#### Publishing to GitHub Container Registry (GHCR)

For maintainers with push access, publish your local image to GHCR:

**Step 1: Authenticate with GHCR**

```bash
export CR_PAT=<your-github-personal-access-token>
echo $CR_PAT | docker login ghcr.io -u <github-username> --password-stdin
```

**Step 2: Tag the Image**

```bash
docker image tag heiplanet-db ghcr.io/ssciwr/heiplanet-db:latest

# For version-specific tags
docker image tag heiplanet-db ghcr.io/ssciwr/heiplanet-db:v1.2.3
```

**Step 3: Push to GHCR**

```bash
docker push ghcr.io/ssciwr/heiplanet-db:latest
docker push ghcr.io/ssciwr/heiplanet-db:v1.2.3
```

**Note:** Ensure your GitHub Personal Access Token has `write:packages` and `read:packages` permissions.

## Production environment

### Deployment Configuration Options

The system provides three pre-configured deployment options based on data volume and temporal coverage. Choose the configuration that best matches your deployment requirements and available resources.

#### Configuration Comparison

| Feature | Small | Medium | Large |
|---------|-------|--------|-------|
| **Config File** | `production_config_small.yml` | `production_config_medium.yml` | `production_config.yml` |
| **Temporal Coverage** | Single month (July 2025) | 3 months (May-July 2025) | Full year (2025) |
| **Use Case** | Testing, demos, development | Seasonal analysis, typical production | Comprehensive analysis, research |
| **Deployment Time** | Fastest (~5-10 min) | Moderate (~15-25 min) | Slower (~30-60 min) |
| **Storage Required** | Minimal (~500 MB) | Moderate (~1-2 GB) | Higher (~3-5 GB) |
| **Data Resolution** | 0.1° global grid | 0.1° global grid | 0.1° global grid |
| **NUTS Aggregation** | Yes | Yes | Yes |
| **Recommended For** | Development environments | Production deployments | Research and analysis |

#### Small Deployment
**Configuration file:** `heiplanet_db/data/production_config_small.yml`

This minimal deployment is suitable for testing, demonstrations, or resource-constrained environments.

**Included data:**
- NUTS region definitions (European geographic boundaries)
- West Nile virus transmission suitability (R0) for a single month (July 2025)
  - High-resolution global grid (0.1° resolution)
  - NUTS-aggregated regional data

**Characteristics:**
- Fastest deployment time
- Minimal storage requirements
- Ideal for development and testing

#### Medium Deployment
**Configuration file:** `heiplanet_db/data/production_config_medium.yml`

This balanced deployment provides seasonal data coverage suitable for typical use cases.

**Included data:**
- NUTS region definitions (European geographic boundaries)
- West Nile virus transmission suitability (R0) for summer months (May-July 2025)
  - High-resolution global grid (0.1° resolution)
  - NUTS-aggregated regional data

**Characteristics:**
- Moderate deployment time
- Covers critical transmission season
- Recommended for production deployments with seasonal focus

#### Large Deployment
**Configuration file:** `heiplanet_db/data/production_config.yml`

This comprehensive deployment provides full annual data coverage for detailed analysis.

**Included data:**
- NUTS region definitions (European geographic boundaries)
- West Nile virus transmission suitability (R0) for the entire year (2025)
  - High-resolution global grid (0.1° resolution)
  - NUTS-aggregated regional data

**Characteristics:**
- Longer deployment time
- Higher storage requirements
- Provides complete annual transmission dynamics
- Recommended for comprehensive analysis and research

#### Historical Data Configuration (Advanced)

**Configuration file:** `heiplanet_db/data/production_config_45yrs.yml`

An additional configuration is available for long-term historical analysis covering 45 years of data (1980-2025).

**Included data:**
- NUTS region definitions
- West Nile virus transmission suitability (R0) for 1980-2025
  - Lower resolution global grid (0.5° resolution due to data volume)
  - NUTS-aggregated regional data
  - Requires local data files (not available via remote download)

**Characteristics:**
- Extensive deployment time (several hours)
- Significant storage requirements (>20 GB)
- Enables long-term trend analysis and climate change impact assessment
- Intended for advanced research applications

**Note:** This configuration requires data files to be available locally (`host: "local"` in the configuration), as the dataset is too large for standard remote hosting.

#### J-Model and P-Model 2014-2018

**Configuration file:** `heiplanet_db/data/production_config_2014-2018.yml`

**Included data:**
- NUTS region definitions
- West Nile virus transmission suitability (`R0`, J-Model) for 2014-2018, global 0.1° grid
- Adult mosquito abundance (P-Model) for 2014-2018, global 0.1° grid, read from five yearly files, stored as `adults_log10`
- `R0` and `adults_log10` aggregated over NUTS regions

**Characteristics:**
- Requires the gold files from heiplanet-data locally. `docker-compose.yaml` mounts `../heiplanet-data/.data_heiplanet_db/gold` read-only at `/data/gold`, which is where the `host: "local"` entries look (`url: "/data/gold"`). For a run outside Docker, set `url` to that folder.
- `adults` is a density (mosquitoes per 625 km²), so the value at a coarse grid point is the area-weighted mean of the 0.1° cells in its cell.
- The values span about 10 orders of magnitude, which a linear colour scale cannot show. They are therefore stored as `adults_log10` = log10(max(adults, 1)), so values below 1 mosquito per 625 km² become 0. The means are taken before the logarithm. The frontend has to request `adults_log10`.
- Gridded values are stored as compressed tiles, see [Storage of gridded values](#storage-of-gridded-values). Most values are 0 (57 % of `R0`, 73 % of `adults_log10`), which compresses well.

**Measured resources of the gridded variables** (`R0` and `adults_log10`, 60 months, all 9 resolutions; loaded on a workstation without a CPU limit, database with the settings from `docker-compose.yaml`, November 2026). The previous storage, one row per value, is shown for comparison; it was measured the same way.

| | Tiles (current) | One row per value (previous) |
|---|---|---|
| Size in the database | 0.5 GB (`var_tile`, 73,320 tiles; 2.9 GB before compression) | 9.8 GB (`var_value` 8.3 GB with 103 million rows, `grid_point` and `grid_point_resolution` 1.3 GB, `var_coverage` 0.3 GB) |
| Loading time, including `VACUUM ANALYZE` | 1.5 min | 8 min |
| Peak memory, loader | 4.6 GiB | 6.3 GiB |
| `/cartesian`, global 0.1° (6.5 million values) | 0.2 s, +0.4 GiB in the API process | 10 s, +0.9 GiB |
| `/cartesian`, Europe 0.2° | 0.04 s | 0.5 s |
| NetCDF download, Europe 0.1°, 12 months, both variables | 0.5 s, +0.2 GiB | 14 s, +1.8 GiB |

The times are measured in Python without the JSON encoding and transfer of the response, which then dominate for large areas. Both storages returned the same values (within float32 precision) for 216 `/cartesian` requests over all resolutions, 1,600 single values and two downloads.

The full production load (including the NUTS data, loader limited to 4 CPUs) took 1 h 19 min with the previous storage and needed 17 GB of database data directory, of which 4.1 GB were WAL; it has not been measured again with tiles. Docker images take 0.85 GB (`heiplanet-db`), 0.6 GB (`postgis`) and 0.1 GB (frontend), the gold input files 3.8 GB.

On a server with 30 GB of disk:

- Copy the gold files to the server only for loading and delete them afterwards (3.8 GB). Alternatively, load on another machine and move the database with `pg_dump -Fc` / `pg_restore`; the restore needs the same space as the database.
- The WAL keeps up to `max_wal_size` (4 GB in `docker-compose.yaml`) after loading. Setting `max_wal_size=1GB` saves about 3 GB, at the cost of more checkpoints and a somewhat slower load.
- Remove unused Docker images and build cache (`docker system prune`).

#### Configuration keys for variables

Each entry in `data_to_fetch` lists one or more variables in `var_name`. Each variable has a `type`:

| `type` | Meaning |
|---|---|
| `definition` | NUTS shapefile archive |
| `grid` (or `R0`) | Gridded variable over `(time, latitude, longitude)`, in any dimension order |
| `nuts` (or `R0_nuts`) | Variable over `(NUTS_ID, time)` |

Optional keys for `grid` variables:

- `aggregation`: how coarser grid resolutions are obtained.
  - `subsample` (default): coarse grid points show the 0.1° value at that point. Suitable for smooth fields such as R0.
  - `mean`: coarse grid points show the area-weighted mean over their cell. Use this for quantities that vary strongly between neighbouring cells, such as mosquito densities.
- `mask_by`: name of another `grid` variable. Cells where that variable is missing are left out before aggregating, e.g. `mask_by: "R0"` drops the ocean cells that the P-Model writes as 0.

Optional keys for `grid` and `nuts` variables:

- `source_variable`: name of the variable in the file, if it is stored under a different `name`.
- `transform`: `log10` stores log10(max(value, `transform_floor`)) instead of the value. For `aggregation: mean`, the means over coarse cells are taken before the transform.
- `transform_floor`: lower limit before the logarithm (default 1, must be > 0).

#### Grid alignment

All `grid` variables must be on the same grid. The grid points of a coarser resolution are the points whose latitude and longitude are multiples of that resolution (e.g. every second 0.1° point for 0.2°). A resolution is only stored if it is a whole multiple of the grid step of the data.

The frontend only requests the resolutions of the resolution groups (0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 and 5.0°), on grid points that are multiples of the resolution. Data on any other grid could not be shown, so the loader stops with an error before it stores any values of that variable if a `grid` variable:

- is not on a regular grid with the same step on both axes and at least two points per axis,
- has a grid step that is not one of the resolution groups (e.g. 0.25° ERA5 data), or
- has coordinates that are not multiples of its grid step, such as a grid that is offset by half a step (cell centres at 0.05, 0.15, … for 0.1°, or at 0.25, 0.75, … for 0.5°).

The current J-Model and P-Model data are on a 0.1° grid with coordinates on whole tenths of a degree (90.0, 89.9, …, -179.9) and load as they are. Regrid or shift other data to such a grid first.

Rows that do not depend on the data, such as NUTS definitions, variable types, resolution groups and time points, may already be stored at that point.

#### Time points and reloading

The database is monthly. Every time point is the first day of a month, and a value dated on any other day (e.g. a NUTS file stamped on the 15th) is stored under the first day of its month. A `grid` variable with more than one time step in a month is rejected; aggregate it to monthly values first.

A `grid` variable that already has values for one of its months is rejected before anything is stored, e.g. when the load is run again without resetting the tables, or when two files or configuration entries cover the same months. If a load fails partway through, the values of that variable stored so far are removed again, and a stored grid that no values use yet is replaced by the grid of the next variable.

#### Precision

Gridded values are stored as float32, which keeps about 7 significant digits. Integers above 2^24 (about 16.7 million) are no longer exact: 123456789 is returned as 123456792. NUTS values are stored as float64 and are not limited this way, so the gridded and NUTS values of the same quantity can differ in the 8th digit.

#### Storage of gridded values

For every `grid` variable, resolution and month, the loader computes the values on the whole grid of that resolution and cuts them into tiles of 100 × 100 grid points. Each tile is one row of `var_tile`, holding the values as 32-bit floats, compressed by PostgreSQL (lz4). `grid_def` describes the grid of each resolution (first latitude and longitude, step, size), so the grid points themselves are not stored.

A missing value (NaN, e.g. ocean cells) means that there is no data; the API leaves these grid points out. Zeros are stored as zeros. Tiles without any data are not stored.

For debugging, the view `var_value_v` shows the tiles as one row per value (variable, resolution, time, latitude, longitude, value). It decodes every tile and is slow; the API does not use it.

`filename` can be a list of files, which are combined along `time`. For downloads, `url` and `filehash` must then be lists of the same length. For `host: "local"`, `url` can be the folder that holds the files.

Databases created by an earlier version are updated automatically when the production runner starts: the column `var_type.aggregation` is added, and the tables of the previous storage (`var_value`, `var_coverage`, `grid_point`, `grid_point_resolution`) are dropped. The data is then loaded again as tiles. To recreate all tables instead, use `--drop-tables`:

```bash
docker compose --profile init run --rm production-runner python -m heiplanet_db.production --drop-tables
```

### Deploying with a Configuration

To deploy the system in production with your chosen configuration:

1. **Select your configuration** by specifying the appropriate config file when building your Docker image. You can either:

   a. Copy your chosen configuration file to `production_config.yml`:
   ```bash
   cp heiplanet_db/data/production_config_small.yml heiplanet_db/data/production_config.yml
   # or production_config_medium.yml, or keep the default production_config.yml
   ```

   b. Or modify the `production.py` script to reference your chosen configuration file directly.

2. **Build the Docker image** with your selected configuration ([see above](./deployment.md#building-the-image-locally)):
   ```bash
   docker build -t heiplanet-db .
   ```

3. **Run the deployment** using docker compose:
   ```bash
   docker compose up --abort-on-container-exit production-runner
   docker compose up api
   ```

**Note:** You can also create custom configuration files based on these templates to include specific datasets tailored to your research needs. The configuration files follow a YAML structure defining data sources, file locations, and metadata for each dataset to be ingested into the database.


## Troubleshooting

### Common Issues and Solutions

#### Database Connection Errors

**Symptom:** Backend fails to connect to database with `connection refused` or `could not connect` errors.

**Solutions:**
- Verify database container is running: `docker ps | grep postgres`
- Check database logs: `docker logs <postgres-container-name>`
- Ensure `WAIT_FOR_DB=true` is set in your `.env` file
- Verify database credentials match between `.env` and database initialization

#### Stale Volume Data

**Symptom:** Old data persists after configuration changes, or database initialization fails.

**Solution:** Remove Docker volumes to start fresh:

```bash
# Stop and remove containers and volumes
docker compose down -v

# Or manually prune volumes
docker volume prune

# For aggressive cleanup (removes ALL unused Docker resources)
docker system prune --force --volumes
```

**Warning:** This deletes all data in the database. Export important data first if needed.

#### Network Connectivity Issues

**Symptom:** Containers can't communicate, frontend can't reach API, or API can't reach database.

**Solution:** Reset Docker networking:

```bash
# Stop services
docker compose down

# Restart services (Docker recreates networks)
docker compose up -d
```

For persistent issues:
```bash
# Remove all networks
docker network prune

# Restart Docker daemon (Linux)
sudo systemctl restart docker
```

#### Port Already in Use

**Symptom:** `Error: bind: address already in use` when starting containers.

**Solution:** Identify and stop conflicting services:

```bash
# Find process using port 80 (frontend)
sudo lsof -i :80

# Find process using port 8000 (API)
sudo lsof -i :8000

# Find process using port 5432 (database)
sudo lsof -i :5432

# Kill the process or change ports in docker-compose.yaml
```

#### Out of Disk Space

**Symptom:** Container crashes during data loading or image build fails.

**Solution:** Free up Docker disk space:

```bash
# Check Docker disk usage
docker system df

# Remove unused images
docker image prune -a

# Remove all stopped containers
docker container prune

# Complete cleanup
docker system prune -a --volumes
```

#### Permission Denied Errors

**Symptom:** Cannot access files or directories inside containers.

**Solution:** Check file permissions and ownership:

```bash
# Fix ownership of data directories
sudo chown -R $USER:$USER .data_heiplanet_db/

# Ensure Docker has proper permissions
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect
```

#### Image Pull Authentication Failures

**Symptom:** `Error response from daemon: unauthorized` when pulling GHCR images.

**Solution:** Authenticate with GitHub Container Registry:

```bash
# Create GitHub Personal Access Token with read:packages permission
export CR_PAT=<your-token>
echo $CR_PAT | docker login ghcr.io -u <github-username> --password-stdin
```

#### Data Loading Takes Too Long

**Symptom:** `production-runner` container runs for extended periods without completing.

**Potential causes:**
- Large configuration file (e.g., using full year or historical data)
- Slow network connection for data downloads
- Insufficient system resources

**Solutions:**
- Use a smaller configuration file for testing (e.g., `production_config_small.yml`)
- Check available disk space and memory
- Monitor container logs: `docker logs -f <container-name>`
- Ensure stable internet connection for data downloads

#### CORS Errors in Browser

**Symptom:** Frontend shows CORS policy errors in browser console.

**Solution:** Configure `IP_ADDRESS` in `.env` file:

```bash
# For local development
IP_ADDRESS=0.0.0.0

# For production deployment
IP_ADDRESS=<your-server-public-ip>
```

Restart the API container after changes:
```bash
docker compose restart api
```

### Getting Additional Help

If issues persist:
1. Check container logs: `docker logs <container-name>`
2. Review the [GitHub Issues](https://github.com/ssciwr/heiplanet-db/issues)
3. Consult the [Issues documentation](./issues.md) for known problems
4. Report new bugs with full error messages and system information
