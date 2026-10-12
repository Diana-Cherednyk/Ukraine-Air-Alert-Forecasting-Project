import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

TZ='Europe/Kyiv'
CUTOFF=pd.Timestamp('2026-10-05 00:00',tz=TZ).tz_convert('UTC')
END=pd.Timestamp('2026-10-13 00:00',tz=TZ).tz_convert('UTC')

def read_intervals(path):
    d=pd.read_csv(path)
    required={'region','started_at','finished_at'}
    if not required.issubset(d): raise ValueError(f'Missing {required-set(d.columns)}')
    d['started_at']=pd.to_datetime(d.started_at,utc=True)
    d['finished_at']=pd.to_datetime(d.finished_at,utc=True)
    if (d.finished_at<d.started_at).any(): raise ValueError('Negative duration')
    return d

def hourly_targets(d, begin, end, regions=None):
    # All hour intervals are UTC, and represent [t,t+1h).
    times=pd.date_range(begin,end,freq='h',inclusive='left',tz='UTC')
    regions=sorted(regions if regions is not None else d.region.unique())
    n=len(times); out=[]
    for r in regions:
        part=d[d.region==r]
        starts=np.zeros(n,dtype=np.int16); diff=np.zeros(n+1,dtype=np.int32)
        for s,e in zip(part.started_at,part.finished_at):
            if s>=end or e<=begin: continue
            if begin<=s<end:
                k=int((s-begin)//pd.Timedelta(hours=1));starts[k]=1
            # strictly positive overlap: [s,e) with [hour,hour+1)
            lo=max(0,int((s-begin)//pd.Timedelta(hours=1)))
            hi=min(n,int(np.ceil((e-begin)/pd.Timedelta(hours=1))))
            if hi>lo:diff[lo]+=1;diff[hi]-=1
        out.append(pd.DataFrame({'region':r,'hour_utc':times,'started':starts,'active':(np.cumsum(diff[:-1])>0).astype('int8')}))
    return pd.concat(out,ignore_index=True)

def features(panel, region_names):
    local=panel.hour_utc.dt.tz_convert(TZ)
    hour=local.dt.hour.to_numpy();dow=local.dt.dayofweek.to_numpy();doy=local.dt.dayofyear.to_numpy()
    return pd.DataFrame({'region':pd.Categorical(panel.region,categories=region_names).codes,
        'hour_sin':np.sin(2*np.pi*hour/24),'hour_cos':np.cos(2*np.pi*hour/24),
        'week_sin':np.sin(2*np.pi*dow/7),'week_cos':np.cos(2*np.pi*dow/7),
        'year_sin':np.sin(2*np.pi*doy/365.25),'year_cos':np.cos(2*np.pi*doy/365.25),
        'weekend':(dow>=5).astype(int)})
