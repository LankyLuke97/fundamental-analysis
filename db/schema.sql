CREATE TABLE IF NOT EXISTS "transactions" (
    "id" SERIAL PRIMARY KEY,
    "ticker" VARCHAR(15),
    "price" NUMERIC(19, 2),
    "fees" NUMERIC(19, 2),
    "currency" VARCHAR(3),
    "quantity" REAL,
    "datetime" TIMESTAMP,
    "notes" TEXT
);
