"""Single command: isolate historical records, backtest, train and freeze forecasts.
No holdout evaluation is performed here.
"""
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--input',default='volunteer_data_en.csv')
a=p.parse_args()
source=Path(a.input).resolve()
if not source.is_file():p.error(f'CSV not found: {source}. Pass --input PATH')
subprocess.run([sys.executable,str(ROOT/'scripts/prepare.py'),'--input',str(source),'--out',str(ROOT/'data')],cwd=ROOT,check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/train_forecast.py'),'--train',str(ROOT/'data/train_only.csv'),'--out',str(ROOT/'outputs')],cwd=ROOT,check=True)
print('Forecast frozen. To evaluate, explicitly run scripts/evaluate_holdout.py after reviewing the lock.')
