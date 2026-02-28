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

The API will be accessible at `http://localhost:8000`

#### Step 3: Populate Database with Test Data

Run the data ingestion script inside the API container:

```bash
docker exec -it heiplanet_api python3 /heiplanet_db/production.py
```

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
docker build -t heiplanet-backend .
```

This creates a local image tagged as `heiplanet-backend` containing:
- Python backend application
- Data processing pipeline
- Configuration files
- Database ORM and API code

#### Using the Local Image with Docker Compose

**Option 1: Modify docker-compose.yaml**

Change the image reference in `docker-compose.yaml`:

```yaml
# Before
image: ghcr.io/ssciwr/heiplanet-backend:latest

# After
image: heiplanet-backend
```

**Option 2: Use Build Configuration**

Alternatively, uncomment the `build:` lines in `docker-compose.yaml` to force local builds:

```yaml
services:
  api:
    build: .
    # image: ghcr.io/ssciwr/heiplanet-backend:latest
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
docker image tag heiplanet-backend ghcr.io/ssciwr/heiplanet-backend:latest

# For version-specific tags
docker image tag heiplanet-backend ghcr.io/ssciwr/heiplanet-backend:v1.2.3
```

**Step 3: Push to GHCR**

```bash
docker push ghcr.io/ssciwr/heiplanet-backend:latest
docker push ghcr.io/ssciwr/heiplanet-backend:v1.2.3
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
   docker build -t heiplanet-backend .
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
