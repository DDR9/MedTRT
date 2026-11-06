"""
MedTRT Dashboard Entry Point (`app/viewer.py`).

    streamlit run app/viewer.py --server.port 8502

Results pages read `results/` at render time and never touch the GPU. Add a page here; the code for
each lives in `app/dashboard/pages.py`, and the CT workstation lives in `app/workstation.py`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parent / "src"
for path in (APP_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Streamlit reloads only modules in this folder or on PYTHONPATH, which it reads on every run. `sys.path`
# does not count, so a direct `streamlit run` would otherwise keep `medtrt` as it was at server start.
pythonpath = os.environ.get("PYTHONPATH", "")
if str(SRC_DIR) not in pythonpath.split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (str(SRC_DIR), pythonpath)))

from dashboard import pages

st.set_page_config(page_title="MedTRT Dashboard", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

navigation = st.navigation(
    {
        "Results & Outcomes": [
            st.Page(pages.overview, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
            st.Page(pages.speedup, title="Speedup breakdown", icon=":material/stacked_bar_chart:", url_path="speedup"),
            st.Page(pages.latency, title="Latency", icon=":material/timer:", url_path="latency"),
            st.Page(pages.network, title="Network and sweep", icon=":material/memory:", url_path="network"),
            st.Page(pages.accuracy, title="Accuracy and agreement", icon=":material/target:", url_path="accuracy"),
            st.Page(pages.errors, title="Error analysis", icon=":material/troubleshoot:", url_path="errors"),
        ],
        "Engineering & Impact": [
            st.Page(pages.constructed, title="What we constructed", icon=":material/construction:", url_path="constructed"),
            st.Page(pages.implications, title="Implications and applications", icon=":material/hub:", url_path="implications"),
        ],
        "Tools": [
            st.Page("workstation.py", title="CT workstation", icon=":material/view_in_ar:", url_path="workstation"),
        ],
        "Procedures & Reference": [
            st.Page(pages.method, title="Procedures and references", icon=":material/description:", url_path="method"),
            st.Page(pages.claims, title="Claims vs results", icon=":material/fact_check:", url_path="claims"),
        ],
    }
)
pages.shell()
navigation.run()
