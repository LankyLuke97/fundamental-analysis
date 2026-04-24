from database_manager import connect
from dotenv import dotenv_values

from fastapi import FastAPI
from sqlmodel import select

from schema import Transaction

app = FastAPI()
config = dotenv_values('.env')
connection_string = f"{config['DB_PROTOCOL']}://{config['DB_USER']}:{config['DB_PASSWORD']}@{config['DB_HOST']}:{config['DB_PORT']}/{config['DB_NAME']}"

@app.get('/transactions')
async def about(ticker: str | None=None) -> list[Transaction]:
    with connect(echo=True, connection_string=connection_string) as session:
        statement = select(Transaction)
        if ticker: statement = statement.where(Transaction.ticker == ticker)
        return session.exec(statement).all()

