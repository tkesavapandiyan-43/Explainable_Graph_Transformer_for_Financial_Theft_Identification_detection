#!/usr/bin/env bash
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null
python -m streamlit run app.py
