"""The team status page (status.py) as a Streamlit app, live, hosted in Hopsworks (Apps).

Hopsworks runs it from this repository (a Git-backed app on the fork's master, redeployed on
every push), in the project's `python-app-pipeline` environment. Inside Hopsworks the registry is
read with the app's own session; the W&B key comes from the Hopsworks secret WANDB_API_KEY. The
page is rebuilt every STATUS_REFRESH_SECONDS (default 120) without reloading the browser.

Locally, with the packages of requirements.txt plus streamlit:
    streamlit run status_app.py
"""

import os

import streamlit as st
import streamlit.components.v1 as components

import status

REFRESH = int(os.environ.get("STATUS_REFRESH_SECONDS", "120"))

st.set_page_config(page_title="Plasma Pirates Status", layout="wide")
st.markdown("<style>.block-container{padding:0.5rem 1rem 0 1rem;max-width:100%}"
            "header[data-testid='stHeader']{display:none}</style>", unsafe_allow_html=True)


@st.fragment(run_every=REFRESH)
def live_page():
    page, errors, _ = status.render(live=REFRESH)
    components.html(page, height=2800, scrolling=True)


live_page()
