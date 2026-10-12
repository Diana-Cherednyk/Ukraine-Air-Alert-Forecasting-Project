"""Explicit reveal step: verify frozen forecast checksum before reading holdout CSV."""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from pipeline import CUTOFF, END, hourly_targets, read_intervals
p=argparse.ArgumentParser()
p.add_argument('--raw',required=True)
p.add_argument('--forecast',default='outputs/blind_forecast.csv')
p.add_argument('--lock',default='outputs/forecast_lock.json')
p.add_argument('--out',default='outputs/holdout_metrics.json')
a=p.parse_args()
lock=json.loads(Path(a.lock).read_text())
if hashlib.sha256(Path(a.forecast).read_bytes()).hexdigest()!=lock['sha256']:
    raise ValueError('Forecast differs from frozen forecast: evaluation refused')
f=pd.read_csv(a.forecast)
f['hour_utc']=pd.to_datetime(f.hour_utc,utc=True)
if len(f)!=len(f.region.unique())*192 or f.hour_utc.min()!=CUTOFF or f.hour_utc.max()>=END:
    raise ValueError('Unexpected forecast shape or period')
# Only this separate script can read actual holdout observations.
raw=read_intervals(a.raw)
truth=hourly_targets(raw,CUTOFF,END,sorted(f.region.unique()))
m=f.merge(truth,on=['region','hour_utc'],validate='one_to_one')
result={}
for target in ('started','active'):
    y=m[target]; prob=m['prob_'+target]
    result[target]={'brier':float(brier_score_loss(y,prob)),
        'log_loss':float(log_loss(y,prob,labels=[0,1])),
        'prevalence':float(y.mean()),
        'roc_auc':float(roc_auc_score(y,prob)) if y.nunique()==2 else None}
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
Path(a.out).write_text(json.dumps({'forecast_sha256':lock['sha256'],'metrics':result},indent=2))
print(json.dumps(result,indent=2))
