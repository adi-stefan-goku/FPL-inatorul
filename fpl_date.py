"""
FPL-inatorul -- Pasul 1: descarca datele din API-ul public Fantasy Premier League.

Doar biblioteca standard Python (fara pip install), ca sa mearga direct
in orice proiect PyCharm.

Ce face:
  1. bootstrap-static  -> jucatori, echipe, etape (gameweeks)
  2. fixtures          -> toate meciurile sezonului (cu dificultate FDR)
  3. element-summary   -> istoric etapa-cu-etapa pentru fiecare jucator
                          (sezonul curent) + totaluri pe sezoanele trecute

Salveaza totul in folderul data/ ca fisiere CSV:
  players.csv, teams.csv, gameweeks.csv, fixtures.csv,
  player_history.csv, player_past_seasons.csv
"""

import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://fantasy.premierleague.com/api"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HEADERS = {"User-Agent": "Mozilla/5.0 (FPL-inatorul personal project)"}
MAX_WORKERS = 6  # nu bombardam API-ul


def get_json(path, retries=3):
    url = f"{BASE_URL}/{path}"
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Nu am putut descarca {url}: {exc}") from exc
            time.sleep(2 * attempt)


def write_csv(filename, rows, columns=None):
    path = os.path.join(DATA_DIR, filename)
    if not rows:
        print(f"  ATENTIE: {filename} -- 0 randuri, nu scriu fisierul")
        return
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _flat(row.get(k)) for k in columns})
    print(f"  {filename}: {len(rows)} randuri")


def _flat(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    start = time.time()

    print("1/3 Descarc bootstrap-static (jucatori, echipe, etape)...")
    boot = get_json("bootstrap-static/")
    players = boot["elements"]
    teams = boot["teams"]
    events = boot["events"]
    positions = {p["id"]: p["singular_name_short"] for p in boot["element_types"]}
    team_names = {t["id"]: t["name"] for t in teams}

    for p in players:
        p["position"] = positions.get(p["element_type"])
        p["team_name"] = team_names.get(p["team"])
        p["price"] = p["now_cost"] / 10  # API-ul da pretul *10

    write_csv("players.csv", players)
    write_csv("teams.csv", teams)
    write_csv("gameweeks.csv", events)

    print("2/3 Descarc fixtures (meciurile sezonului)...")
    fixtures = get_json("fixtures/")
    for fx in fixtures:
        fx["team_h_name"] = team_names.get(fx["team_h"])
        fx["team_a_name"] = team_names.get(fx["team_a"])
    write_csv("fixtures.csv", fixtures)

    print(f"3/3 Descarc istoricul pentru {len(players)} jucatori "
          f"(dureaza 1-3 minute)...")
    history_rows, past_rows, failed = [], [], []

    def fetch(player):
        return player, get_json(f"element-summary/{player['id']}/")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(fetch, p) for p in players]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                player, summary = fut.result()
            except RuntimeError as exc:
                failed.append(str(exc))
                continue
            name = player["web_name"]
            for h in summary.get("history", []):
                h["web_name"] = name
                h["position"] = player["position"]
                h["team_name"] = player["team_name"]
                history_rows.append(h)
            for s in summary.get("history_past", []):
                s["element"] = player["id"]
                s["web_name"] = name
                past_rows.append(s)
            if i % 100 == 0:
                print(f"    {i}/{len(players)} jucatori...")

    history_rows.sort(key=lambda r: (r["element"], r["round"], r["kickoff_time"] or ""))
    write_csv("player_history.csv", history_rows)
    write_csv("player_past_seasons.csv", past_rows)

    # ---- rezumat ----
    current = next((e for e in events if e["is_current"]), None)
    nxt = next((e for e in events if e["is_next"]), None)
    print("\n================ REZUMAT ================")
    print(f"Jucatori: {len(players)} | Echipe: {len(teams)} | Meciuri: {len(fixtures)}")
    if current:
        print(f"Etapa curenta: {current['name']} (terminata: {current['finished']})")
    if nxt:
        print(f"Urmatoarea etapa: {nxt['name']} -- deadline {nxt['deadline_time']}")
    print(f"Randuri istoric etapa-cu-etapa: {len(history_rows)}")
    print(f"Randuri sezoane trecute: {len(past_rows)}")

    top = sorted(players, key=lambda p: p["total_points"], reverse=True)[:10]
    print("\nTop 10 jucatori dupa puncte totale sezonul asta:")
    for p in top:
        print(f"  {p['web_name']:<18} {p['team_name']:<16} {p['position']:<4} "
              f"£{p['price']:<5} {p['total_points']} pct")

    if failed:
        print(f"\nATENTIE: {len(failed)} jucatori nu s-au putut descarca:")
        for msg in failed[:5]:
            print("  ", msg)
        sys.exit(1)

    print(f"\nGata in {time.time() - start:.0f} secunde. Fisiere in: {DATA_DIR}")


if __name__ == "__main__":
    main()
