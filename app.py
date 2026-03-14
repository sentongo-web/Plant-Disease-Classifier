"""
app.py — Hugging Face Spaces Entry Point
=========================================

Hugging Face Spaces looks for a file called `app.py` in the root of the
repository to launch a Streamlit application.  This file is that entry
point.  It simply imports and runs the real application that lives in
app/streamlit_app.py so all the actual code stays organised in app/.

Why this separation?
--------------------
Hugging Face Spaces has a fixed convention: the root-level app.py is
the file it executes.  Our actual Streamlit code lives in app/streamlit_app.py
to keep the project structure clean.  Rather than move everything to the root
(which would break local development), we use this thin shim that adds the
project root to sys.path and then delegates to the real app module.

Running locally vs on HF Spaces
---------------------------------
  Local:   streamlit run app/streamlit_app.py
  HF:      streamlit run app.py        ← Spaces executes this automatically
"""

import sys
from pathlib import Path

# Ensure the project root is on the Python path so all `from src.xxx import`
# statements inside the app work correctly regardless of where Streamlit
# launches this file from.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Re-export the main() function so Streamlit can find and call it.
# Importing the module triggers the st.set_page_config() call at the top
# of streamlit_app.py, which is required to be the very first Streamlit
# command in the session.
from app.streamlit_app import main

main()
