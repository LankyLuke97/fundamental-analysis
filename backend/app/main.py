# from database_manager import connect
from dotenv import dotenv_values

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.cash_flow import router as cash_flow_router
from app.db import database  # noqa: F401

app = FastAPI()
config = dotenv_values(".env")
# connection_string = f"{config['DB_PROTOCOL']}://{config['DB_USER']}:{config['DB_PASSWORD']}@{config['DB_HOST']}:{config['DB_PORT']}/{config['DB_NAME']}"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cash_flow_router)
