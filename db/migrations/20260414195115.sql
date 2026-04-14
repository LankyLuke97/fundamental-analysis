-- Create "transaction" table
CREATE TABLE "public"."transaction" (
  "id" serial NOT NULL,
  "ticker" character varying(15) NOT NULL,
  "price" numeric(19,2) NOT NULL,
  "fees" numeric(19,2) NOT NULL,
  "currency" character varying(3) NULL DEFAULT 'GBP',
  "quantity" real NOT NULL,
  "datetime" timestamp NULL DEFAULT '2026-04-14 20:51:14.067564',
  "notes" text NULL DEFAULT '',
  PRIMARY KEY ("id")
);
-- Drop "transactions" table
DROP TABLE "public"."transactions";
