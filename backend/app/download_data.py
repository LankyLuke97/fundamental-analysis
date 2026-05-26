import asyncio
from dotenv import dotenv_values
from httpx import AsyncClient, Headers
from typing import cast, Any, Mapping, Sequence, TypeAlias

config: dict[str, str] = {}
AsyncClientHeader: TypeAlias = Headers | Mapping[str, str] | Mapping[bytes, bytes] | Sequence[tuple[str, str]] | Sequence[tuple[bytes, bytes]] | None 

async def download_one(client: AsyncClient, url: str, headers: AsyncClientHeader, params: dict[str, str | int] | None=None) -> Any:
    headers = headers or {}
    params = params or {}
    try:
        response = await client.get(url, headers=headers, params=params)
    except:
        print(f"Failed to download from {url}")
        raise
    print(response)
    print(f"Downloaded successfully from {response.url}")
    return params

def download_many(base_endpoint: str, tickers: list[str]) -> int:
    return asyncio.run(supervisor(base_endpoint, tickers))

async def supervisor(base_endpoint: str, tickers: list[str]) -> int:
    headers = {'apikey': config['API_FMP']}
    async with AsyncClient() as client:
        download_coroutines = [download_one(client, base_endpoint, headers, {'symbol': ticker}) for ticker in tickers]
        res = await asyncio.gather(*download_coroutines, return_exceptions=True)
    print(res)

    return len(res)

def main() -> None:
    tickers = ['AAPL', 'GAW.L', 'META', 'BABA']
    download_many('https://financialmodelingprep.com/stable/historical-price-eod/light', tickers)

if __name__ == "__main__":
    config = cast(dict[str, str], dotenv_values(".env"))
    for k in config:
        if not config[k]: del config[k]
    main()

