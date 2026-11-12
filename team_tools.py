"""Logica pentru echipa utilizatorului: echipa de start optima, capitan, transferuri."""
import itertools
import json
import urllib.request

import pandas as pd

BASE_URL = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (FPL-inatorul personal project)"}
FORMATION_LIMITS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
SQUAD_SIZE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3


def get_json(path):
    req = urllib.request.Request(f"{BASE_URL}/{path}", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


class TeamNotFound(Exception):
    pass


class NoPicksYet(Exception):
    pass


def fetch_team(entry_id, gw):
    """Lotul de 15 jucatori + banii din banca pentru etapa gw (ultima etapa cu deadline trecut)."""
    import urllib.error
    try:
        entry = get_json(f"entry/{entry_id}/")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise TeamNotFound(entry_id) from exc
        raise
    try:
        picks = get_json(f"entry/{entry_id}/event/{gw}/picks/")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:  # echipa noua, inca n-a trecut niciun deadline
            raise NoPicksYet(entry_id) from exc
        raise
    hist = picks.get("entry_history", {})
    return {
        "nume_echipa": entry.get("name"),
        "manager": f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip(),
        "puncte_totale": entry.get("summary_overall_points"),
        "loc_general": entry.get("summary_overall_rank"),
        "banca": hist.get("bank", 0) / 10,
        "elemente": [p["element"] for p in picks["picks"]],
        "chip_activ": picks.get("active_chip"),
    }


def best_xi(squad):
    """Cele 11 cu cele mai multe puncte prezise, respectand formatiile permise.
    squad: DataFrame cu coloanele position, puncte_prezise."""
    squad = squad.sort_values("puncte_prezise", ascending=False)
    best, best_pts = None, -1e9
    for d in range(3, 6):
        for m in range(2, 6):
            f = 10 - d - m
            if not 1 <= f <= 3:
                continue
            need = {"GK": 1, "DEF": d, "MID": m, "FWD": f}
            parts = [squad[squad.position == pos].head(n) for pos, n in need.items()]
            if any(len(p) < n for p, n in zip(parts, need.values())):
                continue
            xi = pd.concat(parts)
            pts = xi.puncte_prezise.sum()
            if pts > best_pts:
                best, best_pts = xi, pts
    if best is None:
        return squad.head(0), squad, 0.0
    bench = squad.drop(best.index)
    gk_bench = bench[bench.position == "GK"]
    bench = pd.concat([gk_bench, bench[bench.position != "GK"]])
    return best, bench, best_pts


FORMATIONS = [(d, m, 10 - d - m) for d in range(3, 6) for m in range(2, 6) if 1 <= 10 - d - m <= 3]


def _fast_score(by_pos):
    """Ca xi_score, dar pe liste simple: by_pos = {pozitie: [puncte, ...]}."""
    srt = {p: sorted(v, reverse=True) for p, v in by_pos.items()}
    if not srt.get("GK"):
        return 0.0
    best = None
    for d, m, f in FORMATIONS:
        need = {"DEF": d, "MID": m, "FWD": f}
        if any(len(srt.get(p, [])) < n for p, n in need.items()):
            continue
        chosen = [srt["GK"][0]] + [x for p, n in need.items() for x in srt[p][:n]]
        total = sum(chosen) + max(chosen)
        best = total if best is None or total > best else best
    return 0.0 if best is None else best


def xi_score(squad):
    """Puncte prezise ale echipei de start optime + capitanul (dublat)."""
    by_pos = {}
    for p, x in zip(squad.position, squad.puncte_prezise.astype(float)):
        by_pos.setdefault(p, []).append(x)
    return _fast_score(by_pos)


def transfer_suggestions(squad, pool, bank, n_transfers=1, top_k=8):
    """Cele mai bune transferuri: maximizeaza punctele echipei de start + capitan.
    Pretul de vanzare e aproximat cu pretul curent (API-ul public nu da pretul exact)."""
    sq = squad.to_dict("records")
    for r in sq:
        r["puncte_prezise"] = float(r["puncte_prezise"])
    base_pos = {}
    for r in sq:
        base_pos.setdefault(r["position"], []).append(r["puncte_prezise"])
    base = _fast_score(base_pos)
    clubs = {}
    for r in sq:
        clubs[r["team_name"]] = clubs.get(r["team_name"], 0) + 1
    owned = {r["element"] for r in sq}
    candidates = pool[~pool.element.isin(owned) & (pool.puncte_prezise > 0)]
    cand = (candidates.sort_values("puncte_prezise", ascending=False)
            .groupby("position").head(40).to_dict("records"))

    def score_after(outs, ins):
        by_pos = {p: list(v) for p, v in base_pos.items()}
        for o in outs:
            by_pos[o["position"]].remove(o["puncte_prezise"])
        for i in ins:
            by_pos.setdefault(i["position"], []).append(float(i["puncte_prezise"]))
        return _fast_score(by_pos)

    def clubs_ok(outs, ins):
        c = dict(clubs)
        for o in outs:
            c[o["team_name"]] -= 1
        for i in ins:
            c[i["team_name"]] = c.get(i["team_name"], 0) + 1
        return max(c.values()) <= MAX_PER_CLUB

    singles = []
    for out in sq:
        for inn in cand:
            if inn["position"] != out["position"]:
                continue
            money = bank + out["price"] - inn["price"]
            if money < -1e-9 or not clubs_ok([out], [inn]):
                continue
            singles.append({"iese": out["web_name"], "intra": inn["web_name"],
                            "intra_echipa": inn["team_name"], "pozitie": out["position"],
                            "pret_iese": out["price"], "pret_intra": inn["price"],
                            "banca_ramasa": round(money, 1),
                            "castig": round(score_after([out], [inn]) - base, 2),
                            "_out": out, "_in": inn})
    if not singles:
        return pd.DataFrame()
    singles.sort(key=lambda r: r["castig"], reverse=True)
    if n_transfers == 1:
        return (pd.DataFrame(singles).drop(columns=["_out", "_in"])
                .head(top_k).reset_index(drop=True))

    # 2 transferuri: combinam cele mai bune transferuri simple (jucatori diferiti)
    pairs = []
    for a, b in itertools.combinations(singles[:40], 2):
        if a["_out"]["element"] == b["_out"]["element"] or a["_in"]["element"] == b["_in"]["element"]:
            continue
        outs, ins = [a["_out"], b["_out"]], [a["_in"], b["_in"]]
        money = bank + sum(o["price"] for o in outs) - sum(i["price"] for i in ins)
        if money < -1e-9 or not clubs_ok(outs, ins):
            continue
        pairs.append({"transfer_1": f"{a['iese']} -> {a['intra']}",
                      "transfer_2": f"{b['iese']} -> {b['intra']}",
                      "banca_ramasa": round(money, 1),
                      "castig": round(score_after(outs, ins) - base, 2)})
    pairs = pd.DataFrame(pairs)
    if pairs.empty:
        return pairs
    return pairs.sort_values("castig", ascending=False).head(top_k).reset_index(drop=True)
