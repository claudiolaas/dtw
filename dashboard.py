#%%
from copy import deepcopy
from sqlite3 import Row
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import dateutil.relativedelta
import io
import yfinance as yf
from dtaidistance import dtw,ed
from dtaidistance import dtw_visualisation as dtwvis
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'
from plotly.subplots import make_subplots
from datetime import timedelta
from PIL import Image
def split(df,len_chunk,step):
    start = 0
    end = start + len_chunk
    dfs = []

    while end<len(df):
        dfs.append(df.iloc[start:end,:].copy(deep=True))
        start += step
        end = start + len_chunk
    dfs.append(df.tail(len_chunk).copy(deep=True))
    return dfs
#%%

#########
# Config
#########
title = "Past == Future"
pio.templates.default = "plotly_white"

#########
# Layout
#########
# Wide format
st.set_page_config(layout='wide', page_title=title, page_icon=":chart_with_upwards_trend:")
st.title(title)

######################
# Sidebar with filters
######################
with st.sidebar:
    with st.form("filters"):
        input_ticker = st.text_input("Ticker",'SPY')
        lookback = st.slider('How many days back?', 7, 60, 14)
        normalizer = st.radio("Normalizer",('Z-Score','Min-Max'))
        dist_measure = st.radio("Distance Measure",('Dynamic Time Warping','Euclidean Distance'))
        submitted = st.form_submit_button("Load data")

#############
# Query data
#############
# @st.cache(show_spinner=False)
def load_from_dwh(input_ticker):
    df = yf.Ticker(input_ticker).history(period="max")
    return df
# input_ticker = 'SPY'
# lookback = 15
# normalizer = 'Z-Score'
# dist_measure ='Dynamic Time Warping'



df = load_from_dwh(input_ticker)
df.index = df.index.date
df = df.reset_index()

dfs = split(df.iloc[:-lookback,:], len_chunk=lookback, step=2)
dists = []

if normalizer == 'Z-Score':
    nmz = zscore
else:
    nmz = min_max_scaler.fit_transform

if dist_measure == 'Dynamic Time Warping':    
    dm = dtw
else:
    dm = ed

seq1 = nmz(df.tail(lookback)['Close'].values.reshape(-1, 1))

for x in range(len(dfs)):

    seq2 = nmz((dfs[x]['Close'].values.reshape(-1, 1)))
    distance = dm.distance(seq1,seq2)

    dists.append(distance)


df_hist = dfs[np.argmin(dists)]
current =  df.tail(3*lookback)
current = current.reindex(pd.RangeIndex(current.index[0],current.index[-1]+2*lookback))

hist_start = df_hist.index[0] - (3*lookback-lookback)
hist_end = df_hist.index[-1] + 2*lookback
hist = df[hist_start:hist_end]

fig = make_subplots(rows=2)

fig.add_trace(go.Candlestick(x=current.index,
                open=current['Open'],
                high=current['High'],
                low=current['Low'],
                close=current['Close']),row=1,col=1)
fig.update(layout_xaxis1_rangeslider_visible=False)

fig.add_trace(go.Candlestick(x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close']),row=2,col=1)
fig.update(layout_xaxis2_rangeslider_visible=False)

fig.add_vline(x=df.tail(lookback).index[0], line_width=3, line_dash="dash", line_color="green",row=1,col=1)
fig.add_vline(x=df.tail(lookback).index[-1], line_width=3, line_dash="dash", line_color="green",row=1,col=1)

fig.add_vline(x=df_hist.index[0], line_width=3, line_dash="dash", line_color="green",row=2,col=1)
fig.add_vline(x=df_hist.index[-1], line_width=3, line_dash="dash", line_color="green",row=2,col=1)
fig.update_layout(
    xaxis1 = dict(tickmode = 'array',
        tickvals = [df.tail(lookback).index[0], df.tail(lookback).index[-1]],
        ticktext = [df.tail(lookback).head(1)['index'].values[0].strftime("%m/%d/%Y"),df.tail(lookback).tail(1)['index'].values[0].strftime("%m/%d/%Y")]
    ),
    xaxis2 = dict(tickmode = 'array',
        tickvals = [df_hist.index[0],df_hist.index[-1]],
        ticktext = [df_hist.head(1)['index'].values[0].strftime("%m/%d/%Y"),df_hist.tail(1)['index'].values[0].strftime("%m/%d/%Y")]
    ))
fig.update_layout(height=800)

st.plotly_chart(fig,use_container_width=True,height=800)

path = dtw.warping_path(seq1, nmz(df_hist['Close'].values.reshape(-1, 1))[:,0])

dtwvis.plot_warping(seq1[:,0], nmz(df_hist['Close'].values.reshape(-1, 1))[:,0], path, filename="warp.png")


st.header('Time warped comparison of closing prices')
st.image(Image.open('warp.png'),use_column_width=True)

# %%
