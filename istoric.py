"""FPL-inatorul -- compara predictiile arhivate (inainte de deadline) cu punctele reale."""
import glob
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
ARHIVA = os.path.join(DATA, "arhiva")


def _real_points():
    h = pd.read_csv(os.path.join(DATA, "player_history.csv"),
                    usecols=["element", "round", "total_points", "minutes"])
    return h.groupby(["round", "element"], as_index=False)[["total_points", "minutes"]].sum()


def load_comparisons():
    """Returneaza (rezumat pe etape, detalii pe jucator) doar pentru etapele terminate."""
    gws = pd.read_csv(os.path.join(DATA, "gameweeks.csv"))
    finished = set(gws.loc[gws["finished"].astype(str).str.lower() == "true", "id"].astype(int))
    avg_manager = dict(zip(gws["id"], gws["average_entry_score"]))
    real = _real_points()

    rows, details = [], []
    for path in sorted(glob.glob(os.path.join(ARHIVA, "predictii_etapa_*.csv"))):
        gw = int(re.search(r"predictii_etapa_(\d+)\.csv", path).group(1))
        if gw not in finished:
            continue
        p = pd.read_csv(path)
        r = real[real["round"] == gw].set_index("element")
        p["puncte_reale"] = p["element"].map(r["total_points"]).fillna(0)
        p["etapa"] = gw
        played = p[p["meciuri"] > 0]  # jucatori cu meci in etapa

        top10 = played.sort_values("puncte_prezise", ascending=False).head(10)
        owned10 = played.sort_values("selected_by_percent", ascending=False).head(10)
        cap = top10.iloc[0]
        rows.append({
            "Etapa": gw,
            "Top 10 FPL-inatorul": round(top10["puncte_reale"].mean(), 2),
            "Top 10 cei mai deținuți": round(owned10["puncte_reale"].mean(), 2),
            "Media tuturor jucătorilor": round(played["puncte_reale"].mean(), 2),
            "Căpitan recomandat": cap["web_name"],
            "Puncte căpitan": int(cap["puncte_reale"]),
            "Greșeala medie (pct)": round((played["puncte_prezise"] - played["puncte_reale"]).abs().mean(), 2),
            "Media managerilor FPL": avg_manager.get(gw),
        })
        details.append(top10[["etapa", "web_name", "team_name", "position", "adversar",
                              "puncte_prezise", "puncte_reale"]])

    summary = pd.DataFrame(rows).sort_values("Etapa", ascending=False) if rows else pd.DataFrame()
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return summary, detail


if __name__ == "__main__":
    s, d = load_comparisons()
    print(s.to_string(index=False))
    print(d.to_string(index=False))
