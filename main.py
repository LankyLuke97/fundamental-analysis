import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', '-r', 'requirements.txt'])

import csv
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv, dotenv_values
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import requests
import time
from typing import NamedTuple

class Record(NamedTuple):
    ticker: str
    weight: float
    present_value: float
    margin_of_safety: float
    current_price: float

load_dotenv()
fmp_key = os.getenv("FMP_KEY")

def request_data(endpoint, **parameters):
    request = requests.get(f'{endpoint}?{"".join([f"{key}={value}&" for key, value in parameters.items()])}apikey={fmp_key}')
    if request.status_code == 200:
        return request.content
    print(f'Request to endpoint {endpoint} unsuccessful: {request.status_code}')
    return []

def request_data_to_json(json_file_path, endpoint, **parameters):
    param_mapping = {'from_': 'from'}
    for param, map_to in param_mapping.items():
        if param in parameters: parameters[map_to] = parameters.pop(param)
    if not up_to_date(json_file_path):
        json_data = json.loads(request_data(endpoint, **parameters))
        os.makedirs(Path(json_file_path).parent, exist_ok=True)
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def up_to_date(filepath):
    time_stamp = datetime.fromtimestamp(os.path.getmtime(filepath)) if filepath.exists() else None
    return time_stamp and (time_stamp.year, time_stamp.month, time_stamp.day) == (datetime.today().year, datetime.today().month, datetime.today().day)

available_tickers_json = Path('data', 'available_tickers.json')
if not up_to_date(available_tickers_json):
    request_data_to_json(available_tickers_json, 'https://financialmodelingprep.com/api/v3/financial-statement-symbol-lists')

with open(available_tickers_json, 'r', encoding='utf-8') as f:
    available_tickers = json.load(f)

records = []
watchlist = sys.argv[1:]
if not watchlist:
    with open('watchlist.txt', 'r', encoding='utf-8') as f:
        watchlist = [ticker.strip() for ticker in f.readlines()]

for ticker in watchlist if watchlist else sys.argv[1:]:
    if ticker not in available_tickers:
        print(f'{ticker} has no associated financial statements at financial modeling prep.')
        continue
    income_statement = request_data_to_json(Path('data', ticker, 'income_statements.json'),
                                            f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}',
                                            period='annual',
                                            limit='11')
    balance_sheet = request_data_to_json(Path('data', ticker, 'balance_sheets.json'),
                                         f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}',
                                         period='annual',
                                         limit='11')
    metrics = request_data_to_json(Path('data', ticker, 'metrics.json'),
                                  f'https://financialmodelingprep.com/api/v3/key-metrics/{ticker}',
                                  period='annual',
                                  limit='11')
    eod = request_data_to_json(Path('data', ticker, 'end_of_day.json'),
                                  f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}',
                                  from_=datetime.strftime(datetime.today() - relativedelta(weeks=1), '%Y-%m-%d'),
                                  )
    
    key_info_income = ['date',
                       'revenue',
                       'netIncome',
                       'epsdiluted',
                      ]
    key_info_balance = ['date',
                        'cashAndCashEquivalents',
                        'totalCurrentAssets',
                        'totalCurrentLiabilities',
                        'totalNonCurrentLiabilities',
                        'totalEquity',
                        ]
    key_info_metrics = ['date',
                        'debtToEquity',
                        'peRatio',
                        'pbRatio',
                        'roic',
                        'dividendYield',
                        ]
    
    df = pd.merge(pd.DataFrame({key : [year[key] for year in income_statement] for key in key_info_income}),
                  pd.DataFrame({key : [year[key] for year in balance_sheet] for key in key_info_balance}),
                  on='date').set_index('date')
    
    for col in df.columns:
        current_value = df.iloc[0][col]
        growth_rate = np.power(current_value / df[col].iloc[1:], 1 / np.arange(1, len(df))) - 1
        df[f'{col}_cagr'] = np.concatenate(([0], growth_rate))
            
    df = df.merge(pd.DataFrame({key : [year[key] for year in metrics] for key in key_info_metrics}).set_index('date'), left_index=True, right_index=True)
    df['roic_calc'] = df.netIncome / (df.totalNonCurrentLiabilities + df.totalEquity)

    REFERENCE_RATE = 0.1
    years = [(y, w) for y, w in [(1, 0.17),(3, 0.22),(5,0.27),(10,0.34)] if y < len(df)]
    adjust = sum([w for _, w in years])
    years = [(y, w / adjust) for y, w in years]
    check = [('totalEquity',0.27),('epsdiluted',0.22),('revenue',0.17)]
    df.reset_index(inplace=True)
    weight = (df.loc[[year for year, _ in years], [f'{c}_cagr' for c, _ in check]] - REFERENCE_RATE).mul(
        pd.Series({year: weight for year, weight in years}),
        axis="index"
    ).mul(
        pd.Series({f'{c}_cagr': weight for c, weight in check}),
        axis=1
    ).sum().sum() + (df.roic.mean() * 0.34)

    present_value = 0
    if df.totalEquity_cagr.iloc[len(df)-1] > 0 and df.epsdiluted.iloc[0] > 0:
        COMPOUND_YEARS = 10
        DISCOUNT_RATE = 0.15
        MAX_GROWTH_RATE = 0.2
        compound_factor = np.power(1 + min(df.totalEquity_cagr.iloc[len(df)-1], MAX_GROWTH_RATE), COMPOUND_YEARS)
        discount_factor = np.power(1 + DISCOUNT_RATE, -COMPOUND_YEARS)
        future_value = df.epsdiluted.iloc[0] * compound_factor * df.peRatio.iloc[0]
        present_value = future_value * discount_factor
    
    df.to_csv(Path('data', '_analysis', f'{ticker}.csv'))
    records.append(Record(ticker=ticker, weight=weight, present_value=present_value, margin_of_safety=(present_value/2), current_price=eod['historical'][0]['adjClose']))

records = sorted(records, key=lambda record: record.weight, reverse=True)
with open('watchlist_analysis.csv', mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=Record._fields)
    writer.writeheader()
    for record in records:
        writer.writerow(record._asdict())