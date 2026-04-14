-- Create "transactions" table
CREATE TABLE "public"."transactions" (
  "id" serial NOT NULL,
  "ticker" character varying(15) NOT NULL,
  "price" numeric(19,2) NOT NULL,
  "fees" numeric(19,2) NOT NULL,
  "currency" character varying(3) NULL DEFAULT 'GBP',
  "quantity" real NOT NULL,
  "datetime" timestamp NULL DEFAULT '2026-04-11 14:22:03.66473',
  "notes" text NULL,
  PRIMARY KEY ("id")
);
