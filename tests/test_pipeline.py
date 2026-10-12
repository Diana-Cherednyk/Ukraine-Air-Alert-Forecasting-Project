import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
from pipeline import hourly_targets,CUTOFF

def test_overlap_and_start():
    b=pd.Timestamp('2026-10-04 10:00',tz='UTC')
    d=pd.DataFrame({'region':['A'], 'started_at':[b+pd.Timedelta(minutes=20)],
                    'finished_at':[b+pd.Timedelta(hours=2,minutes=20)]})
    x=hourly_targets(d,b,b+pd.Timedelta(hours=4),['A'])
    assert x.started.tolist()==[1,0,0,0]
    assert x.active.tolist()==[1,1,1,0]

def test_cutoff_is_utc_correct():
    assert CUTOFF==pd.Timestamp('2026-10-04 21:00:00+00:00')

def test_no_end_boundary_overlap():
    b=pd.Timestamp('2026-10-04 10:00',tz='UTC')
    d=pd.DataFrame({'region':['A'], 'started_at':[b], 'finished_at':[b+pd.Timedelta(hours=1)]})
    x=hourly_targets(d,b,b+pd.Timedelta(hours=2),['A'])
    assert x.active.tolist()==[1,0]
