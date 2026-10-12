import argparse,hashlib,json
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pipeline import read_intervals,CUTOFF,END
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',default='data');a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
d=read_intervals(a.input)
# All rows starting at/after cutoff are excluded. Crossing intervals are censored.
train=d[d.started_at<CUTOFF].copy(); train['finished_at']=train.finished_at.clip(upper=CUTOFF)
train.to_csv(out/'train_only.csv',index=False)
# Restricted holdout is deliberately NOT created by this command.
manifest={'cutoff_utc':CUTOFF.isoformat(),'forecast_end_utc':END.isoformat(),
 'training_rows':len(train),'regions':sorted(train.region.unique().tolist()),
 'source_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
 'max_training_start':train.started_at.max().isoformat(),
 'max_training_finish':train.finished_at.max().isoformat()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
