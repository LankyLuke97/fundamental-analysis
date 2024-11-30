import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', '-r', 'requirements.txt'])

from datetime import datetime
from dotenv import load_dotenv, dotenv_values
import json
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
    json_data = json.loads(request_data(endpoint, parameters=parameters))
    os.makedirs(Path(json_file_path).parent, exist_ok=True)
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

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
    income_statement_json = Path('data', ticker, 'income_statements.json')
    if not up_to_date(income_statement_json):
        request_data_to_json(income_statement_json, f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}', period='annual', limit='11')
    with open(income_statement_json, 'r', encoding='utf-8') as f:
        income_statement = json.load(f)

    balance_sheet_json = Path('data', ticker, 'balance_sheets.json')
    if not up_to_date(balance_sheet_json):
        request_data_to_json(balance_sheet_json, f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}', period='annual', limit='11')
    with open(balance_sheet_json, 'r', encoding='utf-8') as f:
        balance_sheet = json.load(f)

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
    df = pd.merge(pd.DataFrame({key : [year[key] for year in income_statement] for key in key_info_income}),
                  pd.DataFrame({key : [year[key] for year in balance_sheet] for key in key_info_balance}),
                  on='date').set_index('date')
    df['roic'] = df.netIncome / (df.totalNonCurrentLiabilities + df.totalEquity)
    print(df)