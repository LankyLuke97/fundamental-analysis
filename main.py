import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', '-r', 'requirements.txt'])

from dotenv import load_dotenv, dotenv_values
import json
import os
import requests

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

available_tickers = json.loads(request_data('https://financialmodelingprep.com/api/v3/financial-statement-symbol-lists'))
for ticker in sys.argv[1:]:
    if ticker not in available_tickers:
        print(f'{ticker} has no associated financial statements at financial modeling prep.')
        continue
    income_statement = request_data(f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}', period='annual', limit='11', datatype='csv')
    with open(f'{ticker}.csv','wb') as file:
        file.write(income_statement)