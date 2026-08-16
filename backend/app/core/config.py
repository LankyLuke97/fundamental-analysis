import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    def __init__(self):
        self._database_name = os.getenv("DATABASE_NAME", "test.db")

    @property
    def database_url(self):
        return f"sqlite:///{self._database_name}"


config = Config()
