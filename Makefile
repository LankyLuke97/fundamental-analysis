
start_postgres: stop_postgres create_volume
	docker pull postgres
	docker run --rm --name postgres-db \
	    -e POSTGRES_PASSWORD=mypassword \
	    -e POSTGRES_USER=myuser \
	    -e POSTGRES_DB=mydatabase \
	    -p 127.0.0.1:5433:5432 \
	    -v postgres_data:/var/lib/postgresql \
	    -d postgres

stop_postgres:
	@-docker stop postgres-db 
	@-docker rm postgres-db 

create_volume:
	docker volume create postgres_data

remove_volume:
	docker volume rm postgres_data
