"""
FPL-inatorul Weekly -- genereaza continutul newsletter-ului din predictiile modelului.

Ruleaza dupa fpl_model.py (in GitHub Actions sau local). Scrie un fisier Markdown in
data/newsletter/GW<etapa>.md, gata de copiat in Substack. Textul e doar scheletul cu
cifre -- Adrian adauga 2-3 randuri proprii la inceput, ca sa sune a om, nu a robot.
"""
import os

import pandas as pd

import istoric

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(DATA, "newsletter")
SITE = "https://fpl-inator.streamlit.app"
STATUS = {"d": "doubtful", "i": "injured", "s": "suspended", "u": "unavailable", "n": "not in squad"}



def _fixtures(r, gw_cols):
    return ", ".join(str(r[c]) for c in gw_cols if isinstance(r[c], str))


def build():
    pred = istoric.fix_venue(pd.read_csv(os.path.join(DATA, "predictii_etapa_urmatoare.csv")))
    gw = int(pred["etapa"].iloc[0])
    gws = pd.read_csv(os.path.join(DATA, "gameweeks.csv"))
    dl = gws.loc[gws.id == gw, "deadline_time"]
    deadline = (pd.to_datetime(dl.iloc[0]).tz_convert("Europe/London").strftime("%A %d %B, %H:%M")
                if not dl.empty else "")
    players_file = os.path.join(DATA, "players.csv")
    players = pd.read_csv(players_file) if os.path.exists(players_file) else pd.DataFrame()

    gw_cols = sorted(c for c in pred.columns if c.startswith("adversar_etapa_"))
    avail = pred[pred.status.fillna("a") == "a"]
    L = [f"# FPL-inatorul Weekly — Gameweek {gw}", ""]
    if deadline:
        L += [f"*Deadline: {deadline} (UK time)*", ""]
    L += ["<!-- Scrie aici 2-3 randuri ale tale: cum ti-a mers echipa, ce ai observat. -->", ""]

    # --- track record ---
    summary, _ = istoric.load_comparisons()
    if not summary.empty:
        last = summary.sort_values("Etapa").iloc[-1]
        ours, owned = last["Top 10 FPL-inatorul"], last["Top 10 cei mai deținuți"]
        verdict = "beat" if ours > owned else ("matched" if ours == owned else "lost to")
        L += [f"## How the model did in GW{int(last['Etapa'])}", "",
              f"Our top 10 picks averaged **{ours:.1f} pts** each, vs **{owned:.1f}** for the 10 most-owned "
              f"players — we {verdict} the template this week. "
              f"Recommended captain {last['Căpitan recomandat']} scored {int(last['Puncte căpitan'])} pts.",
              "",
              f"Season so far ({len(summary)} GWs): **{summary['Top 10 FPL-inatorul'].mean():.2f}** vs "
              f"**{summary['Top 10 cei mai deținuți'].mean():.2f}** for the most-owned. "
              f"Every week, good or bad, is on the site.", ""]

    # --- captains ---
    L += ["## Captain picks", ""]
    for i, r in enumerate(avail.sort_values("puncte_prezise", ascending=False).head(5).itertuples(), 1):
        L.append(f"{i}. **{r.web_name}** ({r.team_name}) — {r.adversar} — {r.puncte_prezise:.1f} pts")
    L.append("")

    # --- best by position over 3 GWs ---
    L += ["## Best picks for the next 3 gameweeks", ""]
    names = {"GK": "Goalkeepers", "DEF": "Defenders", "MID": "Midfielders", "FWD": "Forwards"}
    for pos, label in names.items():
        top = avail[avail.position == pos].sort_values("puncte_3_etape", ascending=False).head(3)
        L.append(f"**{label}:** " + " · ".join(
            f"{r.web_name} (£{r.price:.1f}m, {r.puncte_3_etape:.1f} pts — {_fixtures(r, gw_cols)})"
            for _, r in top.iterrows()))
    L.append("")

    # --- differentials ---
    diff = avail[avail.selected_by_percent < 10].sort_values("puncte_3_etape", ascending=False).head(5)
    L += ["## Differentials (under 10% owned)", ""]
    L += [f"- **{r.web_name}** ({r.team_name}, £{r.price:.1f}m, {r.selected_by_percent:.1f}% owned) — "
          f"{r.puncte_3_etape:.1f} pts over 3 GWs" for r in diff.itertuples()]
    L.append("")

    # --- budget ---
    cheap = avail[((avail.position.isin(["DEF", "GK"])) & (avail.price <= 4.6)) |
                  ((avail.position == "MID") & (avail.price <= 5.5)) |
                  ((avail.position == "FWD") & (avail.price <= 6.0))]
    L += ["## Budget enablers", ""]
    L += [f"- **{r.web_name}** ({r.position}, {r.team_name}, £{r.price:.1f}m) — {r.puncte_3_etape:.1f} pts over 3 GWs"
          for r in cheap.sort_values("puncte_3_etape", ascending=False).head(4).itertuples()]
    L.append("")

    # --- injury watch among popular players ---
    flagged = pred[(pred.status.fillna("a") != "a") & (pred.selected_by_percent >= 5)]
    if not flagged.empty:
        news = dict(zip(players["id"], players["news"])) if "news" in players else {}
        L += ["## Injury watch (popular players)", ""]
        for r in flagged.sort_values("selected_by_percent", ascending=False).head(6).itertuples():
            chance = "" if pd.isna(r.chance_of_playing_next_round) else f"{int(r.chance_of_playing_next_round)}% · "
            note = news.get(r.element)
            L.append(f"- **{r.web_name}** ({r.selected_by_percent:.1f}% owned) — {chance}{STATUS.get(r.status, r.status)}"
                     + (f": {note}" if isinstance(note, str) and note else ""))
        L.append("")

    # --- transfer trends ---
    if {"transfers_in_event", "transfers_out_event"} <= set(players.columns):
        tin = players.sort_values("transfers_in_event", ascending=False).head(3)
        tout = players.sort_values("transfers_out_event", ascending=False).head(3)
        L += ["## What other managers are doing", "",
              "Most bought: " + ", ".join(f"{r.web_name} ({int(r.transfers_in_event):,})" for r in tin.itertuples()),
              "", "Most sold: " + ", ".join(f"{r.web_name} ({int(r.transfers_out_event):,})" for r in tout.itertuples()),
              ""]

    L += ["---", "",
          f"Want picks for **your** team? Enter your FPL ID at [{SITE.replace('https://', '')}]({SITE}) "
          "for your best XI, captain and transfers. Free, no sign-up.", "",
          "Know someone who plays FPL? Forward this to them — they can subscribe at "
          "[fplinator.substack.com](https://fplinator.substack.com).", "",
          "*Predictions are estimates from a statistical model, not guarantees. "
          "Unofficial, not affiliated with the Premier League.*"]

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"GW{gw}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"Newsletter scris: {path}")
    return path


if __name__ == "__main__":
    build()
