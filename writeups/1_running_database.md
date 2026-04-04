## Setting up the database

I've got some Python that downloads data from Financial Modelling Prep - great. It's too specific and it only downloads from one specific endpoint, but that's fine. The next thing is that I don't want to necessarily download that data everytime I want to use that data, so let's store it somewhere. The default to my mind for this is a database. The simplest 'database' I could go for would be a CSV file that I could then quickly load into a Pandas dataframe whenever I wanted to operate on it, and there's definitely merits to doing that here, but I could convert JSON to a CSV file and back to a dataframe in my sleep, so let's go with an actual database instead, as I might learn something.  

I'm going to go with a SQL database here for now. Given that this is currently time series data rather than relational, there's an argument to be made for using a database optimised for this, and I have been intrigued by the idea of learning something like KDB+, but there are a few good reasons not to:
1. I'm planning on having this app handle a few different things for me, not just time series data, I would prefer to have something that is also easy to use for those data, and 'normal' relational databases can handle both types.
2. This is not a high-frequency trading app, so something like KDB+ would be massive over-engineering for the stock data, and I would then need *another* database for any other data.
3. You should just use PostgreSQL, you do not have an app with 10 million users and millions of read/writes per second.

Additionally, if I really wanted to, I could just add TimescaleDB on as an extension to Postgres. Decision made.  

I've only installed and configured Postgres from scratch once, at work, about two years ago, and I cannot remember how to do it. Additionally, that time I just installed it raw on the server: here, I want to set it up in a Docker container, as I have an eye to eventually move this all into the cloud. Let's check the initial script for setting this up:  
```
#!/usr/bin/env bash

docker pull postgres
docker run --name postgres-db \
    -e POSTGRES_PASSWORD=mypassword \
    -e POSTGRES_USER=myuser \
    -e POSTGRES_DB=mydatabase \
    -p 5432:5432
    -v postgres-data:/var/lib/postgresql/data \
    -d postgres
```
Yes, those are the actual values I have written; no, I will not be leaving them in any script that is anywhere other than a my PC, and even then they will live there only temporarily. I start by pulling the image to make sure it's up to date; this can be changed if I want it pinned to a specific tag, but `latest` is fine for now. Three standard Postgres details are provided as environment variables to the image; then there is the port mapping to expose the normal Postgres port; following that is a volume mapping, so that the actual data for the database isn't stored within the ephemeral container but instead on the persistent file system of my computer; `-d` to detach the container (run it in the background); and, finally, the name (and tag) of the image we're using.  

I need to set up that postgres-data volume. Thinking about that, though, I think I would like setup and teardown commands for all of this for testing, which sounds like a job for a makefile to me. What do I want to have easy control over? I want to create the volume if it's not been torn down; and I want to run the Postgres container if it's not already doing so, with a prerequisite being the volume needs to have been created.  

At this stage, most of the process is quite simple. I won't elucidate all the commands, as they're quite basic. However, there were two points of interest. Firstly, I found out that you can ignore failures in a make command by leading the command with a `-`. This is useful for me, as I want to always stop and remove any containers with the same name as the container I'm trying to start prior to spinning up my Postgres container. Docker returns an error if it can't find the container you're referring to, so chaining the commands to stop and remove the container with the one to create it means that it *only* starts the container if there was a previously existing container with the same name; obviously, this is not intentional. Ignoring failures from the `docker stop` and `docker rm` commands solves this. The second item of interest to me was a reminder to keep the correct mental model in my head when dealing with containers. My first attempts to start the container appeared successful initially, but the container was exiting immediately with an error code. An inspection of the logs (once I stopped running the container with the autoremove option) showed the following:
```
Error: in 18+, these Docker images are configured to store database data in a
       format which is compatible with "pg_ctlcluster" (specifically, using
       major-version-specific directory names).  This better reflects how
       PostgreSQL itself works, and how upgrades are to be performed.

       See also https://github.com/docker-library/postgres/pull/1259

       Counter to that, there appears to be PostgreSQL data in:
         /var/lib/postgresql/data (unused mount/volume)

       This is usually the result of upgrading the Docker image without
       upgrading the underlying database using "pg_upgrade" (which requires both
       versions).

       The suggested container configuration for 18+ is to place a single mount
       at /var/lib/postgresql which will then place PostgreSQL data in a
       subdirectory, allowing usage of "pg_upgrade --link" without mount point
       boundary issues.

       See https://github.com/docker-library/postgres/issues/37 for a (long)
       discussion around this process, and suggestions for how to do so.
```
I immediately started checking the file system on my laptop, but couldn't see a `/var/lib/postgresql/data` folder. Of course, the error is *inside* the container, which has its own segregated file system. Once I changed the file path within the container to which I was mounting the volume, everything worked correctly, and I was able to connect to my new, containerised, definitely-not-productionised, database. Great success.
