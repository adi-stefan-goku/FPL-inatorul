"""Porneste site-ul FPL-inatorul (apasa Run in PyCharm). Se deschide pe http://localhost:8502"""
import importlib.util
import os
import subprocess
import sys

if importlib.util.find_spec("streamlit") is None:
    print("Instalez streamlit (o singura data)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "streamlit"])

ROOT = os.path.dirname(os.path.abspath(__file__))
print("Pornesc site-ul pe http://localhost:8502 (opreste-l cu butonul rosu Stop)")
subprocess.call([sys.executable, "-m", "streamlit", "run", os.path.join(ROOT, "app.py"),
                 "--server.port", "8502"], cwd=ROOT)
