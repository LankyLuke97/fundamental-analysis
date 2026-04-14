CREATE TABLE IF NOT EXISTS "transactions" (
    "id" SERIAL PRIMARY KEY,
    "ticker" VARCHAR(15) NOT NULL,
    "price" NUMERIC(19, 2) NOT NULL,
    "fees" NUMERIC(19, 2) NOT NULL,
    "currency" VARCHAR(3) DEFAULT 'GBP',
    "quantity" REAL NOT NULL,
    "datetime" TIMESTAMP DEFAULT '2026-04-11 14:22:03.664730',
    "notes" TEXT
);
