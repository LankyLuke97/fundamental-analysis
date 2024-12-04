import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', '-r', 'requirements.txt'])

from datetime import datetime
from dotenv import load_dotenv, dotenv_values
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import requests
import time

load_dotenv()
fmp_key = os.getenv("FMP_KEY")

if len(sys.argv) == 1:
    raise ValueError("Please input tickers for analysis.")

def request_data(endpoint, **parameters):
    request = requests.get(f'{endpoint}?{"".join([f"{key}={value}&" for key, value in parameters.items()])}apikey={fmp_key}')
    if request.status_code == 200:
        return request.content
    print(f'Request to endpoint {endpoint} unsuccessful: {request.status_code}')
    return []

def request_data_to_json(json_file_path, endpoint, **parameters):
    if not up_to_date(json_file_path):
        json_data = json.loads(request_data(endpoint, parameters=parameters))
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

for ticker in sys.argv[1:]:
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
    present_value = 0
    if df.totalEquity_cagr.iloc[len(df)-1] > 0 and df.epsdiluted.iloc[0] > 0:
        COMPOUND_YEARS = 10
        DISCOUNT_RATE = 0.15
        MAX_GROWTH_RATE = 0.2
        compound_factor = np.power(1 + min(df.totalEquity_cagr.iloc[len(df)-1], MAX_GROWTH_RATE), COMPOUND_YEARS)
        discount_factor = np.power(1 + DISCOUNT_RATE, -COMPOUND_YEARS)
        future_value = df.epsdiluted.iloc[0] * compound_factor * df.peRatio.iloc[0]
        present_value = future_value * discount_factor
    df.to_csv(Path('data', ticker, 'analysis.csv'))
    print(f'{ticker} true value: {present_value:2f}, margin of safety price: {(present_value / 2):.2f}')