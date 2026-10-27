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


def fetch_team(entry_id, gw):
    """Lotul de 15 jucatori + banii din banca pentru etapa gw (ultima etapa cu deadline trecut)."""
    entry = get_json(f"entry/{entry_id}/")
    picks = get_json(f"entry/{entry_id}/event/{gw}/picks/")
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


def xi_score(squad):
    """Puncte prezise ale echipei de start optime + capitanul (dublat)."""
    xi, _, pts = best_xi(squad)
    if xi.empty:
        return 0.0
    return pts + xi.puncte_prezise.max()


def transfer_suggestions(squad, pool, bank, n_transfers=1, top_k=8):
    """Cele mai bune transferuri: maximizeaza punctele echipei de start + capitan.
    Pretul de vanzare e aproximat cu pretul curent (API-ul public nu da pretul exact)."""
    base = xi_score(squad)
    owned = set(squad.element)
    candidates = pool[~pool.element.isin(owned) & (pool.puncte_prezise > 0)]
    # limitam candidatii la cei mai buni pe pozitie si pret, ca sa fie rapid
    cand = (candidates.sort_values("puncte_prezise", ascending=False)
            .groupby("position").head(40))

    def valid(new_squad, money_left):
        if money_left < -1e-9:
            return False
        if new_squad.team_name.value_counts().max() > MAX_PER_CLUB:
            return False
        return True

    singles = []
    for _, out in squad.iterrows():
        same = cand[cand.position == out.position]
        for _, inn in same.iterrows():
            money = bank + out.price - inn.price
            new = pd.concat([squad[squad.element != out.element], inn.to_frame().T])
            if not valid(new, money):
                continue
            new["puncte_prezise"] = new["puncte_prezise"].astype(float)
            gain = xi_score(new) - base
            singles.append({"iese": out.web_name, "intra": inn.web_name,
                            "intra_echipa": inn.team_name, "pozitie": out.position,
                            "pret_iese": out.price, "pret_intra": inn.price,
                            "banca_ramasa": round(money, 1), "castig": round(gain, 2),
                            "_out": out.element, "_in": inn.element})
    singles = pd.DataFrame(singles)
    if singles.empty:
        return singles
    singles = singles.sort_values("castig", ascending=False)
    if n_transfers == 1:
        return singles.drop(columns=["_out", "_in"]).head(top_k).reset_index(drop=True)

    # 2 transferuri: combinam cele mai bune transferuri simple (jucatori diferiti)
    top = singles.head(40).to_dict("records")
    pairs = []
    for a, b in itertools.combinations(top, 2):
        if a["_out"] == b["_out"] or a["_in"] == b["_in"]:
            continue
        outs, ins = {a["_out"], b["_out"]}, {a["_in"], b["_in"]}
        new = pd.concat([squad[~squad.element.isin(outs)], pool[pool.element.isin(ins)]])
        money = bank + squad[squad.element.isin(outs)].price.sum() - pool[pool.element.isin(ins)].price.sum()
        if not valid(new, money):
            continue
        new["puncte_prezise"] = new["puncte_prezise"].astype(float)
        pairs.append({"transfer_1": f"{a['iese']} -> {a['intra']}",
                      "transfer_2": f"{b['iese']} -> {b['intra']}",
                      "banca_ramasa": round(money, 1),
                      "castig": round(xi_score(new) - base, 2)})
    pairs = pd.DataFrame(pairs)
    if pairs.empty:
        return pairs
    return pairs.sort_values("castig", ascending=False).head(top_k).reset_index(drop=True)
