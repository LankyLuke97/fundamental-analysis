
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

remove_volume: stop_postgres
	docker volume rm postgres_data

generate_schema:
	embar schema > db/schema.sql
	
generate_diff:
	docker pull postgres
	@-docker run --rm --name atlas-mig-db \
	    -e POSTGRES_PASSWORD=atlas \
	    -e POSTGRES_USER=atlas \
	    -e POSTGRES_DB=mig_db \
	    -p 127.0.0.1:5434:5432 \
	    -d postgres && sleep 5
	atlas migrate diff --config file://db/atlas.hcl --env local --dir file://db/migrations --to file://db/schema.sql --dev-url "postgres://atlas:atlas@localhost:5434/mig_db"

apply_diff:
	atlas migrate apply --config file://db/atlas.hcl --env local --dir file://db/migrations 
