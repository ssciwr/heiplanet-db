---
title: heiplanet platform deployment
hide:
- navigation
---

# Deployment of the database, API and frontend
The system is set up using [docker compose](https://docs.docker.com/compose/). These require adequate Github permissions/tokens for the GHCR images. There are different containers spun up using `docker compose`:

- The postgresql database. This uses the public `postgis/postgis:17-3.5` image.
- The frontend. This uses the docker image as pushed to GHCR, `ghcr.io/ssciwr/onehealth-map-frontend:<tag>`, where `<tag>` is replaced by the version number, `latest` or the branch name.
- The Python backend. This contains both the ORM for the database and API, to process requests to the database from the frontend, but also the data feeding into the database. This uses the docker image as pushed to GHCR, `ghcr.io/ssciwr/heiplanet-db:<tag>`, where `<tag>` is replaced by the version number, `latest` or the branch name; or can use a locally built image. The reason to supply a locally built image would be, for example, if one where to provide a changed config for the data feeding into the database, to include more or different data.


## Development environment

To bring up your development environment, add a `.env` file in the `heiplanet-db` root directory, that contains the following environment variables:
```
POSTGRES_USER=<user>
POSTGRES_PASSWORD=<password>
POSTGRES_DB=<db-name>
DB_USER=<user>
DB_PASSWORD=<password>
DB_HOST=db
DB_PORT=5432
DB_URL=postgresql://<user>:<password>@db:5432/<db-name>
WAIT_FOR_DB=true
IP_ADDRESS=0.0.0.0
```
Replace the entries `<user>`, `<password>`, and `<db-name>` with a username, password, and database name of your choosing. You only need to set the IP address for a server running in production (this is relevant for the Cross-Origin Resource Sharing (CORS), a security feature for handling the requests across the web).

To bring the database up and feed the development data into the database, run the command
```
docker compose up --abort-on-container-exit production-runner
```
This will insert the testing data into the database for the development environment. After the data has been inserted, you need to run
```
docker compose up api
```
to start the frontend and API service (request handling to the database). If you are running this locally, you should be able to access the frontend through your browser at `127.0.0.1:80` or `localhost:80`.

If you know what you are doing, and want to test the API directly, you can open port 8000 through changing the `docker-compose.yaml` file, exposing this port from the network by including `ports:` under the service `api:`:
```
  ports:
    - "8000:8000"
```
Similarly you can expose the database, to test the connectivity from outside of the docker network.

### Alternative minimal container local development
In order to run just the database and the API, if for example you lack Github permissions to get tokens for the Dockercompose to run, you can follow this three step procedure:
Here are the commands to run the OneHealth DB backend, which the frontend connects to:

From `heiplanet-db/` folder:

##### 1. Start the database we will connect to:

`docker run -d --name postgres\_heiplanet -p 5432:5432 -e POSTGRES\_PASSWORD=postgres -e POSTGRES\_DB=heiplanet\_db postgis/postgis:17-3.5`

##### 2. Start the API using the Dockerfile:

`docker run --name heiplanet\_api -p 8000:8000 --link postgres\_heiplanet:db -e IP\_ADDRESS=1.1.1.1 -e DB\_URL=postgresql+psycopg2://postgres:postgres@db:5432/heiplanet\_db heiplanet-db`

##### 3. Fill the API with mock data - Run the "production.py" script to generate mock data

`docker exec -it heiplanet\_api python3 /heiplanet\_db/production.py`

### Building the image locally 
To build the docker image locally, i.e. for a changed database config file, execute
```
docker build -t heiplanet-backend .
```
This will build the image locally. In the `docker-compose.yaml` file, you need to change the line `image: ghcr.io/ssciwr/heiplanet-backend:latest` to use your local image. Alternatively, you can also force docker compose to rebuild the image locally by uncommenting the `build: ...` lines in the respective sections. To tag a local image with the correct name so it can be pushed to GHCR, use
```
docker image tag heiplanet-backend ghcr.io/ssciwr/heiplanet-backend:latest
```
This image can be pushed to GHCR (provided, you have set your `CR_PAT` key in your local environment):
```
docker push ghcr.io/ssciwr/heiplanet-backend:latest
```

## Production environment
To run the system in production, change the [database configuration file](../heiplanet_db/data/production_config.yml) to include all the data you want to ingest in the database. Then trigger a local build of the docker image ([see above](./deployment.md#building-the-image-locally)) and run the two docker compose commands, to build the tables locally and start the API service and frontend:
```
docker compose up --abort-on-container-exit production-runner
docker compose up api
```


## Troubleshooting
Sometimes issues arise with old, leftover volumes from the database. To remove old volumes, use `docker compose down -v` or `docker volume prune` (or `docker system prune --force --volumes` to remove all old images, containers and volumes).

The same applies for networking issues, this is usually resolved by a `docker compose down` and `docker compose up -d`.
