-- Create "transactions" table
CREATE TABLE "public"."transactions" (
  "id" serial NOT NULL,
  "ticker" character varying(15) NULL,
  "price" numeric(19,2) NULL,
  "fees" numeric(19,2) NULL,
  "currency" character varying(3) NULL,
  "quantity" real NULL,
  "datetime" timestamp NULL,
  "notes" text NULL,
  PRIMARY KEY ("id")
);
