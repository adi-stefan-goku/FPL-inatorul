"""
Planificatorul de chip-uri: in ce etapa merita Bench Boost, Triple Captain, Free Hit si Wildcard.

Pentru fiecare etapa din orizontul predictiilor (pct_etapa_<n>) calculeaza cat aduce fiecare chip
in plus fata de a nu-l folosi, pe lotul actual al utilizatorului:
  - Bench Boost    = punctele prezise ale celor 4 rezerve
  - Triple Captain = punctele prezise ale capitanului (al treilea multiplicator)
  - Free Hit       = cea mai buna echipa posibila pentru O etapa (buget = valoarea lotului + banca)
                     minus echipa ta in acea etapa
  - Wildcard       = cel mai bun lot pastrat pe urmatoarele N etape minus lotul tau pe aceleasi etape
Optimizarile de lot sunt programe liniare cu numere intregi (scipy.optimize.milp).
Presupunere: lotul ramane cel de acum (fara transferuri intre timp) -- e o estimare, nu o garantie.
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

SQUAD = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
CHIP_NAMES = {"bboost": "Bench Boost", "3xc": "Triple Captain", "freehit": "Free Hit", "wildcard": "Wildcard"}


def gw_columns(pred):
    cols = [c for c in pred.columns if c.startswith("pct_etapa_")]
    return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))


def fixture_notes(fixtures, teams, gws):
    """Pentru fiecare etapa: echipe cu 2 meciuri (DGW) si echipe fara meci (blank)."""
    short = dict(zip(teams["id"], teams["short_name"]))
    out = {}
    fx = fixtures.dropna(subset=["event"])
    for g in gws:
        f = fx[fx["event"] == g]
        counts = pd.concat([f["team_h"], f["team_a"]]).value_counts()
        double = sorted(short.get(t, str(t)) for t, n in counts.items() if n >= 2)
        blank = sorted(short.get(t, str(t)) for t in teams["id"] if t not in counts.index)
        out[g] = {"double": double, "blank": blank, "matches": len(f)}
    return out


def _xi_and_bench(pts_by_pos):
    """pts_by_pos: {poz: [(puncte, nume), ...]} pentru lotul de 15. Intoarce (XI, capitan, banca)."""
    srt = {p: sorted(v, key=lambda x: -x[0]) for p, v in pts_by_pos.items()}
    best = None
    for d in range(3, 6):
        for m in range(2, 6):
            f = 10 - d - m
            if not 1 <= f <= 3:
                continue
            need = {"GK": 1, "DEF": d, "MID": m, "FWD": f}
            if any(len(srt.get(p, [])) < n for p, n in need.items()):
                continue
            xi = [x for p, n in need.items() for x in srt[p][:n]]
            total = sum(x[0] for x in xi)
            if best is None or total > best[0]:
                bench = [x for p in srt for x in srt[p][need.get(p, 0):]]
                best = (total, xi, bench)
    if best is None:
        return 0.0, ("-", 0.0), 0.0
    total, xi, bench = best
    cap = max(xi, key=lambda x: x[0])
    return total, (cap[1], cap[0]), sum(x[0] for x in bench)


def _optimal_squad(pool, pts_matrix, budget, max_vars=220):
    """Cel mai bun lot de 15 (pret <= buget, max 3/club) care maximizeaza suma pe etape a
    (XI optim + capitan). pts_matrix: array (n_jucatori, n_etape). Intoarce (valoare, indici)."""
    n, G = pts_matrix.shape
    keep = np.argsort(-pts_matrix.sum(axis=1))[:max_vars]
    # pastram si cei mai ieftini jucatori pe pozitie, ca sa se poata completa banca
    for pos in SQUAD:
        idx = np.where(pool["position"].values == pos)[0]
        keep = np.union1d(keep, idx[np.argsort(pool["price"].values[idx])[:6]])
    pool = pool.iloc[keep].reset_index(drop=True)
    P = pts_matrix[keep]
    n = len(pool)
    nv = n + 2 * n * G  # x_i, s_ig, c_ig
    c = np.zeros(nv)
    for g in range(G):
        c[n + g * n: n + (g + 1) * n] = -P[:, g]
        c[n + n * G + g * n: n + n * G + (g + 1) * n] = -P[:, g]
    rows, lo, hi = [], [], []

    def add(row, a, b):
        rows.append(row)
        lo.append(a)
        hi.append(b)

    price = pool["price"].values
    r = np.zeros(nv); r[:n] = price; add(r, 0, budget)
    pos = pool["position"].values
    for p, k in SQUAD.items():
        r = np.zeros(nv); r[:n] = (pos == p); add(r, k, k)
    for club in pool["team_name"].unique():
        r = np.zeros(nv); r[:n] = (pool["team_name"].values == club); add(r, 0, MAX_PER_CLUB)
    for g in range(G):
        s0, c0 = n + g * n, n + n * G + g * n
        r = np.zeros(nv); r[s0:s0 + n] = 1; add(r, 11, 11)
        for p in SQUAD:
            r = np.zeros(nv); r[s0:s0 + n] = (pos == p); add(r, XI_MIN[p], XI_MAX[p])
        r = np.zeros(nv); r[c0:c0 + n] = 1; add(r, 1, 1)
        for i in range(n):
            r = np.zeros(nv); r[s0 + i] = 1; r[i] = -1; add(r, -1, 0)      # s <= x
            r = np.zeros(nv); r[c0 + i] = 1; r[s0 + i] = -1; add(r, -1, 0)  # c <= s
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi),
               integrality=np.ones(nv), bounds=Bounds(0, 1), options={"time_limit": 20})
    if res.x is None:
        return None, []
    chosen = np.where(res.x[:n] > 0.5)[0]
    return -res.fun, pool.iloc[chosen]


def plan(pred, squad_elements, bank, team_value=None, chips_available=None, wildcard_window=5):
    """Intoarce (tabel pe etape, recomandari). chips_available: {cod_chip: ultima_etapa_permisa}."""
    cols = gw_columns(pred)
    gws = [int(c.rsplit("_", 1)[1]) for c in cols]
    squad = pred[pred.element.isin(squad_elements)]
    if team_value is None or team_value <= 0:
        team_value = squad["price"].sum()
    budget = team_value + bank
    pool = pred[pred.status.fillna("a") != "u"].reset_index(drop=True)

    rows = []
    for col, g in zip(cols, gws):
        by_pos = {}
        for r in squad.itertuples():
            by_pos.setdefault(r.position, []).append((float(getattr(r, col)), r.web_name))
        xi_pts, (cap_name, cap_pts), bench_pts = _xi_and_bench(by_pos)
        fh_val, _ = _optimal_squad(pool, pool[[col]].to_numpy(dtype=float), budget)
        own = xi_pts + cap_pts
        rows.append({"GW": g, "Your XI (pts)": round(own, 1), "Captain": cap_name,
                     "Bench Boost": round(bench_pts, 1), "Triple Captain": round(cap_pts, 1),
                     "Free Hit": round(max((fh_val or own) - own, 0), 1)})
    table = pd.DataFrame(rows)

    # Wildcard: lotul optim pastrat pe urmatoarele `wildcard_window` etape vs lotul tau
    wildcard_window = min(wildcard_window, len(cols))
    wc = []
    for start in range(len(cols) - wildcard_window + 1):
        win = cols[start:start + wildcard_window]
        best, _ = _optimal_squad(pool, pool[win].to_numpy(dtype=float), budget, max_vars=150)
        own = 0.0
        for col in win:
            by_pos = {}
            for r in squad.itertuples():
                by_pos.setdefault(r.position, []).append((float(getattr(r, col)), r.web_name))
            x, (_, cp), _ = _xi_and_bench(by_pos)
            own += x + cp
        wc.append(round(max((best or own) - own, 0), 1))
    table["Wildcard"] = wc + [np.nan] * (len(cols) - len(wc))

    recs = []
    chips_available = chips_available or {}
    for code, label in CHIP_NAMES.items():
        if code not in chips_available:
            recs.append({"chip": label, "status": "used", "best_gw": None, "gain": None})
            continue
        last = chips_available[code]
        ok = table[table.GW <= last].dropna(subset=[label]) if last else table.dropna(subset=[label])
        if ok.empty:
            recs.append({"chip": label, "status": "outside", "best_gw": None, "gain": None, "last": last})
            continue
        best = ok.loc[ok[label].idxmax()]
        recs.append({"chip": label, "status": "ok", "best_gw": int(best.GW), "gain": float(best[label]),
                     "last": last, "median": float(ok[label].median()), "window": wildcard_window})
    return table, recs


def chips_available_for(chips_csv, chips_used, next_gw):
    """{cod_chip: ultima etapa in care poate fi folosit} pentru chip-urile inca disponibile.
    Foloseste ferestrele din chips.csv (start_event/stop_event); daca lipsesc, presupune doua jumatati
    de sezon (1-19 si 20-38), ca in regulile FPL din 2025-26."""
    windows = []
    if chips_csv is not None and os.path.exists(chips_csv):
        ch = pd.read_csv(chips_csv)
        if {"name", "start_event", "stop_event"} <= set(ch.columns):
            windows = [(r.name, int(r.start_event), int(r.stop_event)) for r in ch.itertuples()]
    if not windows:
        windows = [(c, s, e) for c in CHIP_NAMES for s, e in [(1, 19), (20, 38)]]
    used = chips_used or []
    avail = {}
    for name, start, stop in windows:
        if not start <= next_gw <= stop:
            continue
        if any(u_name == name and start <= u_ev <= stop for u_name, u_ev in used):
            continue
        avail[name] = stop
    return avail
