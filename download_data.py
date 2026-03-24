import asyncio
from dotenv import dotenv_values
from httpx import AsyncClient
from typing import Any

config: dict[str, str | None] = {}

async def download_one(client: AsyncClient, url: str, headers: dict[str, str | None]={}, params: dict[str, str | int]={}) -> Any:
    try:
        response = await client.get(url, headers=headers, params=params)
    except:
        print(f"Failed to download from {response.url}")
        raise
    print(response.text)
    print(f"Downloaded successfully from {response.url}")
    return params

def download_many(tickers: list[str]) -> int:
    return asyncio.run(supervisor(tickers))

async def supervisor(tickers: list[str]) -> int:
    headers = {'apikey': config['API_FMP']}
    async with AsyncClient() as client:
        download_coroutines = [download_one(client, 'https://financialmodelingprep.com/stable/historical-price-eod/light', headers, {'symbol': ticker}) for ticker in tickers]
        res = await asyncio.gather(*download_coroutines, return_exceptions=True)
    print(res)

    return len(res)

def main() -> None:
    tickers = ['AAPL', 'GAW.L', 'META', 'BABA']
    download_many(tickers)

if __name__ == "__main__":
    config = dotenv_values(".env")
    main()

