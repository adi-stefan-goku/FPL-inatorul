"""FPL-inatorul -- pastreaza o copie a predictiilor pentru fiecare etapa.

Ruleaza dupa fpl_model.py. Cat timp deadline-ul etapei nu a trecut, copia
data/arhiva/predictii_etapa_<N>.csv se rescrie cu cea mai noua predictie;
dupa deadline ramane exact ce am prezis inainte de meciuri (istoric cinstit).
"""
import os
import shutil
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
ARHIVA = os.path.join(DATA, "arhiva")
PRED = os.path.join(DATA, "predictii_etapa_urmatoare.csv")

pred = pd.read_csv(PRED)
gw = int(pred["etapa"].iloc[0])
gws = pd.read_csv(os.path.join(DATA, "gameweeks.csv"))
deadline = pd.to_datetime(gws.loc[gws["id"] == gw, "deadline_time"].iloc[0], utc=True)

if datetime.now(timezone.utc) < deadline:
    os.makedirs(ARHIVA, exist_ok=True)
    shutil.copy(PRED, os.path.join(ARHIVA, f"predictii_etapa_{gw}.csv"))
    print(f"Arhivat: predictii_etapa_{gw}.csv (deadline {deadline:%d %b %H:%M} UTC)")
else:
    print(f"Deadline-ul etapei {gw} a trecut -- nu suprascriu arhiva.")
