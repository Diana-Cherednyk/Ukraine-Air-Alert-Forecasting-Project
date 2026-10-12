"""Train two probability models; run 3 strictly forward 8-day backtests.
Never accepts raw data or holdout observations.
"""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from pipeline import CUTOFF, END, TZ, features, hourly_targets, read_intervals

parser = argparse.ArgumentParser()
parser.add_argument('--train', default='data/train_only.csv')
parser.add_argument('--out', default='outputs')
a = parser.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
d = read_intervals(a.train)
if d.empty or d.started_at.max() >= CUTOFF or d.finished_at.max() > CUTOFF:
    raise ValueError('Training data empty or contains information after the cutoff')
regions = sorted(d.region.unique())
# Construct labels only for historical hours. Features are calendar-only and known in advance.
start = CUTOFF - pd.Timedelta(days=425)
panel = hourly_targets(d, start, CUTOFF, regions)
X = features(panel, regions)

def make_model():
    return HistGradientBoostingClassifier(max_iter=65, max_leaf_nodes=15,
        l2_regularization=5, learning_rate=.07, random_state=42)

metrics = {}
for target in ('started', 'active'):
    y = panel[target].to_numpy()
    folds = []
    # Each fold trains on times STRICTLY before its forecast origin.
    for days_back in (24, 16, 8):
        origin = CUTOFF - pd.Timedelta(days=days_back)
        finish = origin + pd.Timedelta(days=8)
        train = (panel.hour_utc < origin).to_numpy()
        valid = ((panel.hour_utc >= origin) & (panel.hour_utc < finish)).to_numpy()
        m = make_model(); m.fit(X.loc[train], y[train]); p = m.predict_proba(X.loc[valid])[:, 1]
        actual = y[valid]
        baseline = np.full(len(actual), y[train].mean())
        score = {'origin_utc': origin.isoformat(), 'end_utc': finish.isoformat(),
            'n': int(len(actual)), 'prevalence': float(actual.mean()),
            'brier': float(brier_score_loss(actual, p)),
            'baseline_brier': float(brier_score_loss(actual, baseline)),
            'log_loss': float(log_loss(actual, p, labels=[0, 1])),
            'roc_auc': float(roc_auc_score(actual, p)) if len(np.unique(actual)) == 2 else None}
        folds.append(score)
    metrics[target] = {'folds': folds, 'mean_brier': float(np.mean([v['brier'] for v in folds])),
        'mean_baseline_brier': float(np.mean([v['baseline_brier'] for v in folds]))}

future = hourly_targets(d.iloc[:0], CUTOFF, END, regions)
fX = features(future, regions)
for target in ('started', 'active'):
    model = make_model(); model.fit(X, panel[target].to_numpy())
    future['prob_' + target] = model.predict_proba(fX)[:, 1]
forecast = future[['region', 'hour_utc', 'prob_started', 'prob_active']].copy()
forecast['hour_kyiv'] = forecast.hour_utc.dt.tz_convert(TZ)
forecast_path = out / 'blind_forecast.csv'
if forecast_path.exists():
    raise FileExistsError('Blind forecast already frozen. Move outputs to a new run directory; never silently overwrite.')
forecast.to_csv(forecast_path, index=False)
sha = hashlib.sha256(forecast_path.read_bytes()).hexdigest()
(out / 'forecast_lock.json').write_text(json.dumps({'sha256': sha, 'cutoff_utc': CUTOFF.isoformat(),
    'end_utc': END.isoformat(), 'rows': len(forecast), 'region_count': len(regions)}, indent=2))
(out / 'backtest_metrics.json').write_text(json.dumps(metrics, indent=2))
print(json.dumps({'forecast_rows': len(forecast), 'forecast_sha256': sha, 'backtest': metrics}, indent=2))
