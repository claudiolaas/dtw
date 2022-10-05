#%%
from copy import deepcopy
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import dateutil.relativedelta
import io
import plotly.io as pio
import yfinance as yf
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from scipy.stats import zscore
import functions as f
#%%

#########
# Config
#########
title = "Past == Future"
pio.templates.default = "plotly_white"

date_picker_lookback_days = 0
from_date = datetime.date.today() - dateutil.relativedelta.relativedelta(days=date_picker_lookback_days+1)
to_date = datetime.date.today() - dateutil.relativedelta.relativedelta(days=1)

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
        lookback = st.st.slider('How many days back?', 7, 60, 14)
        submitted = st.form_submit_button("Load data")

#############
# Query data
#############
@st.cache(show_spinner=False)
def load_from_dwh(input_ticker):
    df = yf.Ticker(input_ticker).history(period="max")
    return df
#%%

df = load_from_dwh(input_ticker)

dfs = f.split(df.iloc[:-lookback,:], len_chunk=lookback, step=2)

dists = []

seq1 = zscore(df.tail(lookback)['Close'].values)

for x in range(len(dfs)):
    
    seq2 = zscore(dfs[x]['Close'].values)
    distance = dtw.distance(seq1,seq2)
    dists.append(distance)

np.argmin(dists)
#%%
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'
fig = go.Figure()


seq2 = zscore(dfs[np.argmin(dists)]['Close'].values)

fig.add_trace(go.Scatter(x=dfs[np.argmin(dists)].index, y=seq1))
fig.add_trace(go.Scatter(x=dfs[np.argmin(dists)].index, y=seq2))

fig.show()
#%%
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np

path = dtw.warping_path(seq1, seq2)
dtwvis.plot_warping(seq1, seq2, path, filename="warp.png")
#%%
# Result set is not empty
if len(df_input) > 0:
    st.subheader("Subheader")
    df_input 
    st.markdown('---')

    ###############
    # Data download
    ###############
    st.subheader("Data download")
    buffer = io.BytesIO()
    with st.spinner('Generating download file'):
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_input.to_excel(writer, sheet_name='Raw')
            writer.save()
            st.download_button(
                label='Download data as Excel',
                data=buffer,
                file_name="streamlit_download.xlsx",
                mime="application/vnd.ms-excel"
            )
else:
    # Result set is empty
    st.warning('No data for selected filters.')

