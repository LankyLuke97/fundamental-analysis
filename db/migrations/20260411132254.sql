-- Modify "transactions" table
ALTER TABLE "public"."transactions" ALTER COLUMN "ticker" SET NOT NULL, ALTER COLUMN "price" SET NOT NULL, ALTER COLUMN "fees" SET NOT NULL, ALTER COLUMN "currency" SET DEFAULT 'GBP', ALTER COLUMN "quantity" SET NOT NULL, ALTER COLUMN "datetime" SET DEFAULT '2026-04-11 14:22:03.66473';
