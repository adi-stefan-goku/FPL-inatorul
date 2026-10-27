"""Feature-uri comune pentru modelul de puncte FPL (folosit la antrenare si la predictie)."""
import numpy as np
import pandas as pd

STAT_COLS = ["total_points", "minutes", "starts", "expected_goals", "expected_assists",
             "expected_goals_conceded", "bps", "bonus", "ict_index", "saves",
             "goals_scored", "assists", "clean_sheets"]
POSITIONS = ["GK", "DEF", "MID", "FWD"]


def normalise(df):
    """Aduce datele (arhiva GitHub sau API-ul FPL) la acelasi format."""
    df = df.copy()
    df["position"] = df["position"].replace({"GKP": "GK"})
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    df["was_home"] = df["was_home"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    for c in STAT_COLS + ["value"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)
    if "starts" not in df or df["starts"].sum() == 0:
        df["starts"] = (df["minutes"] >= 60).astype(int)
    return df


def team_strength(df):
    """xG marcat/primit pe echipa si meci, din randurile jucatorilor."""
    t = (df.groupby(["season", "fixture", "team_id"], as_index=False)
           .agg(kickoff_time=("kickoff_time", "first"), xg_for=("expected_goals", "sum"),
                opp=("opponent_team", "first")))
    opp = t[["season", "fixture", "team_id", "xg_for"]].rename(
        columns={"team_id": "opp", "xg_for": "xg_against"})
    t = t.merge(opp, on=["season", "fixture", "opp"], how="left")
    t = t.sort_values("kickoff_time")
    g = t.groupby(["season", "team_id"])
    for c in ["xg_for", "xg_against"]:
        t[f"team_{c}_l6"] = g[c].transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    return t


def add_features(df):
    """Feature-uri DOAR din meciurile anterioare (shift(1)) -- fara scurgere de informatie."""
    df = df.sort_values(["season", "element", "kickoff_time"]).copy()
    g = df.groupby(["season", "element"])
    for c in STAT_COLS:
        sh = g[c].shift(1)
        df[f"{c}_l3"] = sh.groupby([df["season"], df["element"]]).transform(
            lambda s: s.rolling(3, min_periods=1).mean())
        df[f"{c}_l6"] = sh.groupby([df["season"], df["element"]]).transform(
            lambda s: s.rolling(6, min_periods=1).mean())
        df[f"{c}_season"] = sh.groupby([df["season"], df["element"]]).transform(
            lambda s: s.expanding().mean())
    df["games_before"] = g.cumcount()
    df["minutes_l1"] = g["minutes"].shift(1)
    df["points_l1"] = g["total_points"].shift(1)

    ts = team_strength(df)
    own = ts[["season", "fixture", "team_id", "team_xg_for_l6", "team_xg_against_l6"]]
    df = df.merge(own, on=["season", "fixture", "team_id"], how="left")
    opp = ts[["season", "fixture", "team_id", "team_xg_for_l6", "team_xg_against_l6"]].rename(
        columns={"team_id": "opponent_team", "team_xg_for_l6": "opp_xg_for_l6",
                 "team_xg_against_l6": "opp_xg_against_l6"})
    df = df.merge(opp, on=["season", "fixture", "opponent_team"], how="left")
    for p in POSITIONS:
        df[f"pos_{p}"] = (df["position"] == p).astype(int)
    df["price"] = df["value"] / 10
    return df


def feature_columns():
    cols = []
    for c in STAT_COLS:
        cols += [f"{c}_l3", f"{c}_l6", f"{c}_season"]
    cols += ["minutes_l1", "points_l1", "prev_ppg", "prev_mpg", "prev_xgi90",
             "games_before", "team_xg_for_l6", "team_xg_against_l6", "opp_xg_for_l6",
             "opp_xg_against_l6", "was_home", "price"] + [f"pos_{p}" for p in POSITIONS]
    return cols


SEASON_ORDER = ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]


def add_previous_season(df, past=None):
    """Statistici din sezonul trecut (dupa nume): puncte/meci, minute/meci, xGI la 90."""
    agg = (df.groupby(["season", "name"], as_index=False)
             .agg(pts=("total_points", "sum"), mins=("minutes", "sum"), n=("fixture", "count"),
                  xgi=("expected_goals", "sum"), xa=("expected_assists", "sum")))
    if past is not None:
        agg = pd.concat([agg, past], ignore_index=True)
    agg["prev_ppg"] = agg["pts"] / agg["n"].clip(lower=1)
    agg["prev_mpg"] = agg["mins"] / agg["n"].clip(lower=1)
    agg["prev_xgi90"] = (agg["xgi"] + agg["xa"]) / agg["mins"].clip(lower=90) * 90
    nxt = {a: b for a, b in zip(SEASON_ORDER[:-1], SEASON_ORDER[1:])}
    agg["season"] = agg["season"].map(nxt)
    agg = agg.dropna(subset=["season"]).drop_duplicates(["season", "name"])
    return df.merge(agg[["season", "name", "prev_ppg", "prev_mpg", "prev_xgi90"]],
                    on=["season", "name"], how="left")
