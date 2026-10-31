"""
FPL-inatorul -- Pasul 2: modelul de puncte prezise.

1. Descarca istoricul etapa-cu-etapa pe sezoanele 2022-23 -> 2025-26
   (arhiva publica github.com/vaastav/Fantasy-Premier-League), salvat in data/istoric/.
2. Adauga sezonul curent din data/ (descarcat de fpl_date.py -- ruleaza-l intai).
3. VALIDARE walk-forward (antrenez pe sezoanele trecute, testez pe urmatorul)
   -- modelul trebuie sa bata predictia naiva (media ultimelor 6 meciuri).
4. Antrenez pe tot si prezic etapa urmatoare -> data/predictii_etapa_urmatoare.csv
"""
import importlib.util
import os
import subprocess
import sys

for pkg, mod in [("pandas", "pandas"), ("scikit-learn", "sklearn"), ("streamlit", "streamlit")]:
    if importlib.util.find_spec(mod) is None:
        print(f"Instalez {pkg} (o singura data)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import urllib.request  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from features import add_features, add_previous_season, feature_columns, normalise  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
HIST = os.path.join(DATA, "istoric")
GITHUB = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
PAST_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
CURRENT = "2026-27"
HORIZON = 3  # cate etape inainte prezicem (pentru transferuri)
pd.set_option("display.width", 200)


def make_model():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=50, random_state=0)


def download_history():
    os.makedirs(HIST, exist_ok=True)
    for s in PAST_SEASONS:
        for name, url in [(f"merged_{s}.csv", f"{GITHUB}/{s}/gws/merged_gw.csv"),
                          (f"teams_{s}.csv", f"{GITHUB}/{s}/teams.csv")]:
            path = os.path.join(HIST, name)
            if not os.path.exists(path):  # sezoanele trecute nu se mai schimba -> cache
                print(f"  descarc {name}...")
                urllib.request.urlretrieve(url, path)


def load_all():
    frames = []
    for s in PAST_SEASONS:
        d = pd.read_csv(os.path.join(HIST, f"merged_{s}.csv"), on_bad_lines="skip")
        t = pd.read_csv(os.path.join(HIST, f"teams_{s}.csv"))
        d["team_id"] = d["team"].map(dict(zip(t["name"], t["id"])))
        d["season"] = s
        frames.append(d)
    players = pd.read_csv(os.path.join(DATA, "players.csv"))
    h = pd.read_csv(os.path.join(DATA, "player_history.csv"))
    h["team_id"] = h["element"].map(dict(zip(players["id"], players["team"])))
    h["name"] = h["element"].map(dict(zip(players["id"],
                                           players["first_name"] + " " + players["second_name"])))
    h["season"] = CURRENT
    h["GW"] = h["round"]
    frames.append(h)
    return pd.concat(frames, ignore_index=True), players


def future_rows(players):
    """Un rand 'gol' per jucator si meci din etapa urmatoare; shift(1) in features
    face ca predictia sa foloseasca doar meciurile deja jucate."""
    gws = pd.read_csv(os.path.join(DATA, "gameweeks.csv"))
    nxt = gws[gws["is_next"].astype(str).str.lower() == "true"]
    if nxt.empty:
        return pd.DataFrame(), None
    gw = int(nxt["id"].iloc[0])
    fx = pd.read_csv(os.path.join(DATA, "fixtures.csv"))
    fx = fx[fx["event"] == gw]
    rows = []
    for _, f in fx.iterrows():
        for team, opp, home in [(f["team_h"], f["team_a"], 1), (f["team_a"], f["team_h"], 0)]:
            for _, p in players[players["team"] == team].iterrows():
                rows.append({"element": p["id"], "name": f"{p['first_name']} {p['second_name']}",
                             "position": p["position"], "team_id": team, "opponent_team": opp,
                             "was_home": home, "fixture": f["id"], "kickoff_time": f["kickoff_time"],
                             "value": p["now_cost"], "season": CURRENT, "round": gw, "GW": gw,
                             "_future": 1})
    return pd.DataFrame(rows), gw


def later_gameweek_rows(nxt, gw):
    """Etapele gw+1 .. gw+HORIZON-1: forma jucatorului ramane cea de acum (din randul etapei
    urmatoare); se schimba doar adversarul, terenul si puterea adversarului."""
    fx = pd.read_csv(os.path.join(DATA, "fixtures.csv"))
    player_feat = nxt.sort_values("kickoff_time").drop_duplicates("element")
    team_str = (nxt.drop_duplicates("team_id")
                   .set_index("team_id")[["team_xg_for_l6", "team_xg_against_l6"]])
    rows = []
    for k in range(1, HORIZON):
        for _, f in fx[fx["event"] == gw + k].iterrows():
            for team, opp, home in [(f["team_h"], f["team_a"], 1), (f["team_a"], f["team_h"], 0)]:
                r = player_feat[player_feat["team_id"] == team].copy()
                if r.empty:
                    continue
                r["opponent_team"], r["was_home"], r["fixture"] = opp, home, f["id"]
                r["GW"] = r["round"] = gw + k
                r["kickoff_time"] = f["kickoff_time"]
                if opp in team_str.index:
                    r["opp_xg_for_l6"] = team_str.loc[opp, "team_xg_for_l6"]
                    r["opp_xg_against_l6"] = team_str.loc[opp, "team_xg_against_l6"]
                rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else nxt.head(0)


def validate(df, F):
    print("\n=== VALIDARE walk-forward (puncte pe jucator si etapa) ===")
    folds = [(PAST_SEASONS[:i], s) for i, s in enumerate(PAST_SEASONS[1:] + [CURRENT], 1)]
    out = []
    for train, test in folds:
        tr = df[df.season.isin(train)]
        te = df[df.season == test].copy()
        m = make_model().fit(tr[F], tr["total_points"])
        te["Model"] = m.predict(te[F])
        te["Naiv"] = te["total_points_l6"]
        te["pool"] = (te["minutes_l3"] >= 60).astype(int)
        agg = te.groupby(["GW", "element"], as_index=False).agg(
            Model=("Model", "sum"), Naiv=("Naiv", "sum"), pts=("total_points", "sum"),
            pool=("pool", "max"))
        pool = agg[agg.pool == 1]
        row = {"Test": test, "Etape": agg.GW.nunique()}
        for c in ["Model", "Naiv"]:
            row[f"MAE {c}"] = round((agg[c] - agg.pts).abs().mean(), 3)
            row[f"Top10 {c}"] = round(pool.sort_values(c, ascending=False)
                                      .groupby("GW").head(10).pts.mean(), 2)
            row[f"Capitan {c}"] = round(pool.sort_values(c, ascending=False)
                                        .groupby("GW").head(1).pts.mean(), 2)
        out.append(row)
    res = pd.DataFrame(out)
    print(res.to_string(index=False))
    wins = int((res["MAE Model"] < res["MAE Naiv"]).sum())
    print(f"Modelul bate naivul la MAE pe {wins}/{len(res)} sezoane, "
          f"la Top10 pe {int((res['Top10 Model'] > res['Top10 Naiv']).sum())}/{len(res)}, "
          f"la Capitan pe {int((res['Capitan Model'] > res['Capitan Naiv']).sum())}/{len(res)}.")
    res.to_csv(os.path.join(DATA, "validare_model.csv"), index=False)


def main():
    print("1/4 Date istorice...")
    download_history()
    raw, players = load_all()
    fut, gw = future_rows(players)
    raw["_future"] = 0
    if not fut.empty:
        raw = pd.concat([raw, fut], ignore_index=True)

    print("2/4 Calculez feature-urile (1-2 minute)...")
    df = add_features(add_previous_season(normalise(raw)))
    F = feature_columns()
    df[F] = df[F].fillna(0)
    played = df[(df["_future"] == 0) & (df["games_before"] >= 1)]

    print("3/4 Validare...")
    validate(played, F)

    if fut.empty:
        print("\nNu exista etapa urmatoare in calendar -- nu fac predictii.")
        return
    print(f"\n4/4 Antrenez pe tot si prezic Etapele {gw}-{gw + HORIZON - 1}...")
    model = make_model().fit(played[F], played["total_points"])
    nxt = df[df["_future"] == 1].copy()
    teams = pd.read_csv(os.path.join(DATA, "teams.csv"))
    short = dict(zip(teams["id"], teams["short_name"]))
    later = later_gameweek_rows(nxt, gw)
    allf = pd.concat([nxt, later], ignore_index=True)
    allf["pred"] = model.predict(allf[F])
    allf["adversar"] = allf["opponent_team"].map(short) + np.where(allf["was_home"] == 1, " (H)", " (A)")

    per_gw = (allf.groupby(["element", "GW"], as_index=False)
                  .agg(pred=("pred", "sum"), meciuri=("fixture", "count"),
                       adversar=("adversar", lambda s: ", ".join(s))))
    info = players[["id", "web_name", "team_name", "position", "price", "status",
                    "chance_of_playing_next_round", "selected_by_percent"]].rename(columns={"id": "element"})
    per_gw = per_gw.merge(info, on="element", how="left")
    per_gw["position"] = per_gw["position"].replace({"GKP": "GK"})
    # sansa de a juca (accidentari/suspendari) din FPL: 0-100, gol = disponibil.
    # Se aplica integral pe etapa urmatoare si pe jumatate pe cele de dupa (revin din accidentari).
    chance = pd.to_numeric(per_gw["chance_of_playing_next_round"], errors="coerce").fillna(100) / 100
    chance[per_gw["status"].isin(["u", "n", "s", "i"]) & per_gw["chance_of_playing_next_round"].isna()] = 0
    later_chance = np.where(per_gw["status"] == "u", 0, 0.5 + 0.5 * chance)
    per_gw["pts"] = per_gw["pred"] * np.where(per_gw["GW"] == gw, chance, later_chance)

    wide = per_gw.pivot_table(index="element", columns="GW", values="pts", aggfunc="sum").fillna(0)
    wide.columns = [f"pct_etapa_{int(c)}" for c in wide.columns]
    opp = per_gw.pivot_table(index="element", columns="GW", values="adversar", aggfunc="first")
    opp.columns = [f"adversar_etapa_{int(c)}" for c in opp.columns]
    pred = info.copy()
    pred["position"] = pred["position"].replace({"GKP": "GK"})
    pred = pred.merge(wide, on="element", how="left").merge(opp, on="element", how="left")
    pts_cols = [c for c in pred.columns if c.startswith("pct_etapa_")]
    pred[pts_cols] = pred[pts_cols].fillna(0).round(2)
    pred["puncte_prezise"] = pred.get(f"pct_etapa_{gw}", 0)
    pred["puncte_3_etape"] = pred[pts_cols].sum(axis=1).round(2)
    pred["adversar"] = pred.get(f"adversar_etapa_{gw}", "").fillna("-")
    pred["meciuri"] = pred["element"].map(per_gw[per_gw.GW == gw].set_index("element")["meciuri"]).fillna(0).astype(int)
    pred = pred.sort_values("puncte_prezise", ascending=False)
    out = pred[["web_name", "team_name", "position", "price", "adversar", "meciuri",
                "puncte_prezise", "puncte_3_etape"] + pts_cols +
               [c for c in pred.columns if c.startswith("adversar_etapa_")] +
               ["chance_of_playing_next_round", "status", "selected_by_percent", "element"]]
    out.insert(0, "etapa", gw)
    out.to_csv(os.path.join(DATA, "predictii_etapa_urmatoare.csv"), index=False)

    cols = ["web_name", "team_name", "position", "price", "adversar", "puncte_prezise"]
    print(f"\n=== TOP 10 CAPITAN -- Etapa {gw} ===")
    print(out[cols].head(10).to_string(index=False))
    for pos in ["GK", "DEF", "MID", "FWD"]:
        print(f"\n--- Top 5 {pos} ---")
        print(out[out.position == pos][cols].head(5).to_string(index=False))
    print(f"\nSalvat: {os.path.join(DATA, 'predictii_etapa_urmatoare.csv')}")


if __name__ == "__main__":
    main()
