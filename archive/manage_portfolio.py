from datetime import date
from pathlib import Path
import pandas as pd
import yfinance as yf

orders = pd.read_csv(
        'orders.csv',
        date_format={'date': '%Y-%m-%d'},
)
orders.sort_values(by='date', inplace=True, ignore_index=True)
tickers = set(orders['ticker'])
today = date.today()
for ticker in set(tickers):
    data = Path('data',f"{ticker}.csv")
    if not data.exists(): continue
    modify_date = date.fromtimestamp(int(data.stat().st_mtime))
    if modify_date == today:
        tickers.remove(ticker)
        print(f"Data for {ticker} last updated today; skip download")

if tickers:
    historical = yf.download(tickers, group_by='ticker')

    for ticker in tickers:
        if ticker not in historical:
            print(f"No data for {ticker}")
            continue
        data = Path('data',f"{ticker}.csv")
        data.parent.mkdir(parents=True, exist_ok=True)
        historical[ticker].to_csv(data)
