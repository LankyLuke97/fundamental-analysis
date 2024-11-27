import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', '-r', 'requirements.txt'])

from dotenv import load_dotenv, dotenv_values
import os

load_dotenv()
print(os.getenv("FMP_KEY"))