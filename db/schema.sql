CREATE TABLE IF NOT EXISTS "transaction" (
    "id" SERIAL PRIMARY KEY,
    "ticker" VARCHAR(15) NOT NULL,
    "price" NUMERIC(19, 2) NOT NULL,
    "fees" NUMERIC(19, 2) NOT NULL,
    "currency" VARCHAR(3) DEFAULT 'GBP',
    "quantity" REAL NOT NULL,
    "datetime" TIMESTAMP,
    "notes" TEXT DEFAULT ''
);
