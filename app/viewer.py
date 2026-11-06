"""
MedTRT Dashboard Entry Point (`app/viewer.py`).

    streamlit run app/viewer.py --server.port 8502

Results pages read `results/` at render time and never touch the GPU. Add a page here; the code for
each lives in `app/dashboard/pages.py`, and the CT workstation lives in `app/workstation.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
for path in (APP_DIR, APP_DIR.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dashboard import pages

st.set_page_config(page_title="MedTRT Dashboard", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

navigation = st.navigation(
    {
        "Results": [
            st.Page(pages.overview, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
            st.Page(pages.speedup, title="Speedup breakdown", icon=":material/stacked_bar_chart:", url_path="speedup"),
            st.Page(pages.latency, title="Latency", icon=":material/timer:", url_path="latency"),
            st.Page(pages.network, title="Network and sweep", icon=":material/memory:", url_path="network"),
            st.Page(pages.accuracy, title="Accuracy and agreement", icon=":material/target:", url_path="accuracy"),
            st.Page(pages.errors, title="Error analysis", icon=":material/troubleshoot:", url_path="errors"),
        ],
        "Tools": [
            st.Page("workstation.py", title="CT workstation", icon=":material/view_in_ar:", url_path="workstation"),
        ],
        "Reference": [
            st.Page(pages.claims, title="Claims vs results", icon=":material/fact_check:", url_path="claims"),
            st.Page(pages.method, title="Method and data", icon=":material/description:", url_path="method"),
        ],
    }
)
pages.shell()
navigation.run()
