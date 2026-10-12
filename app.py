"""Streamlit dashboard for frozen regional probability forecasts."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Ukraine Air Alert Forecasting',layout='wide')
st.title('Ukraine · Air Alert Probability Forecasts')
st.warning('Research prototype only. Never use this model for personal safety decisions; follow official alerts.')
forecast=ROOT/'outputs/blind_forecast.csv'
if not forecast.exists():
    st.error('No forecast. Run python main.py --input volunteer_data_en.csv first.'); st.stop()
d=pd.read_csv(forecast)
d['hour_utc']=pd.to_datetime(d.hour_utc,utc=True)
d['hour_kyiv']=d.hour_utc.dt.tz_convert('Europe/Kyiv')
d['date']=d.hour_kyiv.dt.date
metric=st.sidebar.radio('Forecast target', ['prob_active','prob_started'],format_func=lambda x:'Alert active' if x=='prob_active' else 'New alert starts')
date=st.sidebar.selectbox('Kyiv date',sorted(d.date.unique()))
region=st.sidebar.selectbox('Region',sorted(d.region.unique()))
filtered=d[d.date==date].copy()
selected=filtered[filtered.region==region]
left,right=st.columns([1,1])
with left:
    st.subheader(f'{region} · hourly probability')
    st.plotly_chart(px.line(selected,x='hour_kyiv',y=metric,range_y=[0,1],markers=True),use_container_width=True)
with right:
    st.subheader('All regions · hour-by-hour heatmap')
    mat=filtered.assign(hour_local=filtered.hour_kyiv.dt.strftime('%H:%M')).pivot(index='region',columns='hour_local',values=metric)
    st.plotly_chart(px.imshow(mat,aspect='auto',zmin=0,zmax=1,color_continuous_scale='Viridis',labels={'color':'Probability'}),use_container_width=True)
st.subheader('Regional ranking for selected hour')
hours=sorted(filtered.hour_kyiv.dt.hour.unique())
hour=st.slider('Kyiv hour',min_value=min(hours),max_value=max(hours),value=min(hours))
rank=filtered[filtered.hour_kyiv.dt.hour==hour].sort_values(metric,ascending=False)
st.dataframe(rank[['region','hour_kyiv',metric]],hide_index=True,use_container_width=True)

st.subheader('Ukraine geographic map')
st.caption('Map is enabled when a verified oblast boundary GeoJSON is supplied at assets/oblasts.geojson. '
           'Its feature properties must include a region value matching the CSV. No unverified geography is invented.')
geo=ROOT/'assets/oblasts.geojson'
if geo.exists():
    geojson=json.loads(geo.read_text(encoding='utf-8'))
    # Requires feature.properties.region to match the input CSV names.
    features=geojson.get('features',[])
    keys={f.get('properties',{}).get('region') for f in features}
    if not set(rank.region).intersection(keys):
        st.error('GeoJSON region names do not match CSV identifiers; map not shown.')
    else:
        fig=px.choropleth_map(rank,geojson=geojson,locations='region',featureidkey='properties.region',
                              color=metric,color_continuous_scale='Viridis',range_color=[0,1],
                              map_style='carto-positron',zoom=5,center={'lat':49,'lon':31},opacity=.75)
        st.plotly_chart(fig,use_container_width=True)
else:
    st.info('Add verified oblasts.geojson to assets/ to enable the map. Charts and heatmap work without it.')
metrics=ROOT/'outputs/backtest_metrics.json'
if metrics.exists():
    st.subheader('Historical rolling-origin backtests')
    m=json.loads(metrics.read_text())
    for key,value in m.items():
        st.write(f'**{key}** · mean Brier {value["mean_brier"]:.4f} · constant baseline {value["mean_baseline_brier"]:.4f}')
        st.dataframe(pd.DataFrame(value['folds']),hide_index=True)
holdout=ROOT/'outputs/holdout_metrics.json'
if holdout.exists():
    st.subheader('Revealed holdout evaluation')
    st.json(json.loads(holdout.read_text()))
st.download_button('Download frozen predictions',data=forecast.read_bytes(),file_name='blind_forecast.csv',mime='text/csv')
