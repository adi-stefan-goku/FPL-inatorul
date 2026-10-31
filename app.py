"""FPL-inatorul -- web app (Streamlit). Local: porneste_site.py. Online: Streamlit Community Cloud."""
import os

import pandas as pd
import streamlit as st

import istoric
import team_tools as tt

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
PRED_FILE = os.path.join(DATA, "predictii_etapa_urmatoare.csv")

st.set_page_config(page_title="FPL-inatorul", page_icon="⚽", layout="wide")

# ---------------------------------------------------------------- translations
T = {
    "en": {
        "pos": {"GK": "Goalkeeper", "DEF": "Defender", "MID": "Midfielder", "FWD": "Forward"},
        "no_data": "No predictions available yet.",
        "preds_for": "Predictions for **Gameweek {gw}**",
        "horizon": "Transfer horizon", "h3": "Next 3 gameweeks", "h1": "Next gameweek only",
        "updated": "Data updated: {t} · refreshes automatically twice a day",
        "disclaimer": "Unofficial tool, not affiliated with the Premier League or Fantasy Premier League. "
                      "Predictions are estimates, not guarantees.",
        "tabs": ["🧑‍💼 My team", "📈 Top players", "✅ Our track record", "🔬 How good is the model"],
        "my_team": "My team", "id_label": "Your FPL team ID",
        "id_help": "On fantasy.premierleague.com → Points: the number in the web address after /entry/",
        "id_digits": "The ID must contain digits only.", "reading": "Reading your team from FPL...",
        "read_fail": "Could not read team {id} (gameweek {gw}): {e}",
        "team": "Team", "total": "Total points", "rank": "Overall rank", "bank": "In the bank",
        "squad_note": "Squad as of Gameweek {c}. Transfers you already made for Gameweek {gw} are not "
                      "public until the deadline.",
        "best_xi": "Recommended starting XI — Gameweek {gw}",
        "captain": "**Captain: {c}** ({cp:.1f} pts) · Vice: {v} ({vp:.1f} pts) · "
                   "Predicted total (captain doubled): **{t:.1f} pts**",
        "bench": "**Bench** (in substitution order)",
        "availability": "⚠️ Availability issues: ",
        "transfers": "Recommended transfers",
        "transfer_note": "Gain = extra predicted points for your starting XI + captain ({h}). Selling price is "
                         "approximated by the current price (the public FPL API doesn't show your exact "
                         "selling price).",
        "how_many": "How many transfers?", "searching": "Finding the best transfers (10-30 sec)...",
        "no_transfer": "No transfer found that improves your team.",
        "hit_note": "If you only have one free transfer, the second one costs -4 points: only worth it if the "
                    "total gain is clearly above 4.",
        "small_gain": "The best gain is only {g:.1f} pts — within the noise. Consider rolling your transfer to "
                      "next week (free transfers stack up to 5).",
        "top": "Top players — Gameweek {gw}", "position": "Position", "all": "All",
        "max_price": "Max price (£m)", "sort": "Sort by", "s1": "Next gameweek", "s3": "Next 3 gameweeks",
        "hist_title": "Our track record",
        "hist_intro": "Every gameweek's predictions are saved automatically **before the deadline** and never "
                      "changed afterwards. After the matches we compare them with the real points — and we show "
                      "every gameweek, including the bad ones.",
        "hist_empty": "No gameweek with saved predictions has finished yet. Check back after the first one.",
        "m_top10": "FPL-inatorul top 10 (pts/player)",
        "m_top10_help": "Average real points of the 10 players with the highest predicted points.",
        "m_owned": "Top 10 most owned", "m_owned_help": "Comparison: picking the 10 most popular FPL players.",
        "m_all": "Average of all players",
        "hist_caption": "Averages over {n} {w}. ", "gw_one": "gameweek", "gw_many": "gameweeks",
        "hist_small": "A single gameweek proves nothing — watch how the number develops.",
        "real_pts_axis": "Real points / player", "gw_label": "GW {g}",
        "hist_detail": "Details per gameweek", "pick_gw": "Gameweek",
        "summary_cols": {"Etapa": "Gameweek", "Top 10 FPL-inatorul": "FPL-inatorul top 10",
                         "Top 10 cei mai deținuți": "Top 10 most owned",
                         "Media tuturor jucătorilor": "Average of all players",
                         "Căpitan recomandat": "Recommended captain", "Puncte căpitan": "Captain points",
                         "Greșeala medie (pct)": "Average error (pts)"},
        "model_title": "How good is the model",
        "model_intro": "The model is trained on every match of the last 4 seasons (form, minutes, xG, xA, bonus, "
                       "team and opponent strength) and tested **on seasons it has never seen**. We compare it "
                       "with the simple method: average points over the last 6 matches.",
        "val_cols": {"Test": "Test season", "Etape": "Gameweeks", "MAE Model": "MAE model",
                     "Top10 Model": "Top10 model", "Capitan Model": "Captain model", "MAE Naiv": "MAE simple",
                     "Top10 Naiv": "Top10 simple", "Capitan Naiv": "Captain simple"},
        "model_notes": """
- **MAE** = average error in points per player (lower = better).
- **Top10** = average real points of the 10 players each method picks every gameweek.
- **Captain** = real points of each method's #1 pick. Here the model is **not proven yet** (a single player per gameweek is very random).
- Predictions for 2 and 3 gameweeks ahead keep current form and only change the opponent — they are estimates, not separately validated.
""",
        "cols": {"player": "Player", "club": "Club", "opp": "Opponent", "pts_gw": "Pts GW{gw}",
                 "pts3": "Pts next 3", "price": "Price", "out": "Out", "in": "In", "price_out": "Price out",
                 "price_in": "Price in", "bank_after": "Bank after", "gain": "Gain (pts)",
                 "t1": "Transfer 1", "t2": "Transfer 2", "owned": "Owned %", "gw": "GW",
                 "pred": "Predicted points", "real": "Real points"},
    },
}


@st.cache_data
def load_predictions(mtime):
    return istoric.fix_venue(pd.read_csv(PRED_FILE))


@st.cache_data
def load_gameweeks(mtime):
    return pd.read_csv(os.path.join(DATA, "gameweeks.csv"))


@st.cache_data(ttl=600, show_spinner=False)
def cached_team(entry_id, gw):
    return tt.fetch_team(entry_id, gw)


@st.cache_data(show_spinner=False)
def cached_transfers(elements, bank, n, metric, mtime):
    pool = load_predictions(mtime).copy()
    pool["puncte_prezise"] = pool[metric]
    squad = pool[pool.element.isin(elements)]
    return tt.transfer_suggestions(squad, pool, bank, n_transfers=n)


@st.cache_data
def load_history(mtime):
    return istoric.load_comparisons()


def show_table(df, cols, names):
    st.dataframe(df[cols].rename(columns=names), hide_index=True, width="stretch")


L = T["en"]
POS, C = L["pos"], L["cols"]

if not os.path.exists(PRED_FILE):
    st.error(L["no_data"])
    st.stop()

mtime = os.path.getmtime(PRED_FILE)
pred = load_predictions(mtime)
gws = load_gameweeks(mtime)
gw = int(pred["etapa"].iloc[0])
gw_row = gws[gws.id == gw]
deadline = (pd.to_datetime(gw_row["deadline_time"].iloc[0]).tz_convert("Europe/London")
            if not gw_row.empty else None)
current_gw = gw - 1

# ---------------- sidebar ----------------
st.sidebar.title("⚽ FPL-inatorul")
st.sidebar.caption(L["preds_for"].format(gw=gw) +
                   (f" · deadline {deadline:%a %d %b, %H:%M} (UK)" if deadline is not None else ""))
horizon = st.sidebar.radio(L["horizon"], [L["h3"], L["h1"]])
metric = "puncte_3_etape" if horizon == L["h3"] else "puncte_prezise"
st.sidebar.divider()
updated = pd.Timestamp(mtime, unit="s", tz="UTC").tz_convert("Europe/London")
st.sidebar.caption(L["updated"].format(t=f"{updated:%d %b %H:%M}"))
st.sidebar.caption(L["disclaimer"])

tab_team, tab_top, tab_hist, tab_model = st.tabs(L["tabs"])
base_cols = {"web_name": C["player"], "team_name": C["club"], "adversar": C["opp"],
             "puncte_prezise": C["pts_gw"].format(gw=gw), "puncte_3_etape": C["pts3"], "price": C["price"]}

# ---------------- My team ----------------
with tab_team:
    st.header(L["my_team"])
    entry_id = st.text_input(L["id_label"], help=L["id_help"])
    if entry_id:
        if not entry_id.strip().isdigit():
            st.error(L["id_digits"])
            st.stop()
        try:
            with st.spinner(L["reading"]):
                team = cached_team(int(entry_id), current_gw)
        except Exception as exc:  # noqa: BLE001
            st.error(L["read_fail"].format(id=entry_id, gw=current_gw, e=exc))
            st.stop()

        squad = pred[pred.element.isin(team["elemente"])].copy()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(L["team"], team["nume_echipa"] or "-")
        c2.metric(L["total"], team["puncte_totale"])
        c3.metric(L["rank"], f"{team['loc_general']:,}" if team["loc_general"] else "-")
        c4.metric(L["bank"], f"£{team['banca']:.1f}m")
        st.caption(L["squad_note"].format(c=current_gw, gw=gw))

        xi, bench, xi_pts = tt.best_xi(squad)
        cap = xi.sort_values("puncte_prezise", ascending=False)
        st.subheader(L["best_xi"].format(gw=gw))
        if len(cap) >= 2:
            st.success(L["captain"].format(c=cap.iloc[0].web_name, cp=cap.iloc[0].puncte_prezise,
                                           v=cap.iloc[1].web_name, vp=cap.iloc[1].puncte_prezise,
                                           t=xi_pts + cap.iloc[0].puncte_prezise))
        order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
        xi = xi.assign(_o=xi.position.map(order)).sort_values(["_o", "puncte_prezise"], ascending=[True, False])
        xi["_pos"] = xi.position.map(POS)
        bench = bench.copy()
        bench["_pos"] = bench.position.map(POS)
        cols = ["_pos", "web_name", "team_name", "adversar", "puncte_prezise", "puncte_3_etape", "price"]
        names = {"_pos": L["position"], **base_cols}
        show_table(xi, cols, names)
        st.markdown(L["bench"])
        show_table(bench, cols, names)
        injured = squad[squad.status.isin(["i", "s", "u", "d", "n"])]
        if not injured.empty:
            st.warning(L["availability"] + ", ".join(
                f"{r.web_name} ({'' if pd.isna(r.chance_of_playing_next_round) else str(int(r.chance_of_playing_next_round)) + '% '}{r.status})"
                for r in injured.itertuples()))

        st.subheader(L["transfers"])
        st.caption(L["transfer_note"].format(h=horizon.lower()))
        n = int(st.radio(L["how_many"], ["1", "2"], horizontal=True))
        with st.spinner(L["searching"]):
            sugg = cached_transfers(tuple(team["elemente"]), team["banca"], n, metric, mtime)
        if sugg.empty:
            st.info(L["no_transfer"])
        else:
            if n == 1:
                show_table(sugg, ["iese", "intra", "intra_echipa", "pret_iese", "pret_intra",
                                  "banca_ramasa", "castig"],
                           {"iese": C["out"], "intra": C["in"], "intra_echipa": C["club"],
                            "pret_iese": C["price_out"], "pret_intra": C["price_in"],
                            "banca_ramasa": C["bank_after"], "castig": C["gain"]})
            else:
                show_table(sugg, ["transfer_1", "transfer_2", "banca_ramasa", "castig"],
                           {"transfer_1": C["t1"], "transfer_2": C["t2"],
                            "banca_ramasa": C["bank_after"], "castig": C["gain"]})
                st.caption(L["hit_note"])
            best = sugg["castig"].iloc[0]
            if best < 1:
                st.info(L["small_gain"].format(g=best))

# ---------------- Top players ----------------
with tab_top:
    st.header(L["top"].format(gw=gw))
    c1, c2, c3 = st.columns(3)
    pos = c1.selectbox(L["position"], [L["all"]] + list(POS.values()))
    max_price = c2.slider(L["max_price"], 4.0, float(pred.price.max()), float(pred.price.max()), 0.5)
    sort_by = c3.radio(L["sort"], [L["s1"], L["s3"]], horizontal=True)
    view = pred[pred.price <= max_price].copy()
    if pos != L["all"]:
        view = view[view.position.map(POS) == pos]
    view = view.sort_values("puncte_prezise" if sort_by == L["s1"] else "puncte_3_etape",
                            ascending=False).head(30)
    view["_pos"] = view.position.map(POS)
    gw_cols = [c for c in pred.columns if c.startswith("adversar_etapa_")]
    show_table(view, ["web_name", "team_name", "_pos", "price", "puncte_prezise", "puncte_3_etape"]
               + gw_cols + ["selected_by_percent"],
               {**base_cols, "_pos": L["position"], "selected_by_percent": C["owned"],
                **{c: f"{C['gw']} {c.split('_')[-1]}" for c in gw_cols}})

# ---------------- Track record ----------------
with tab_hist:
    st.header(L["hist_title"])
    st.markdown(L["hist_intro"])
    summary, detail = load_history(mtime)
    if summary.empty:
        st.info(L["hist_empty"])
    else:
        n = len(summary)
        c1, c2, c3 = st.columns(3)
        c1.metric(L["m_top10"], f"{summary['Top 10 FPL-inatorul'].mean():.2f}", help=L["m_top10_help"])
        c2.metric(L["m_owned"], f"{summary['Top 10 cei mai deținuți'].mean():.2f}", help=L["m_owned_help"])
        c3.metric(L["m_all"], f"{summary['Media tuturor jucătorilor'].mean():.2f}")
        st.caption(L["hist_caption"].format(n=n, w=L["gw_one"] if n == 1 else L["gw_many"])
                   + (L["hist_small"] if n < 5 else ""))

        sc = L["summary_cols"]
        series = ["Top 10 FPL-inatorul", "Top 10 cei mai deținuți", "Media tuturor jucătorilor"]
        chart = summary.set_index("Etapa")[series].sort_index().rename(columns=sc)
        chart.index = chart.index.map(lambda g: L["gw_label"].format(g=g))
        st.bar_chart(chart, stack=False, y_label=L["real_pts_axis"])

        show_table(summary, ["Etapa"] + series + ["Căpitan recomandat", "Puncte căpitan",
                                                  "Greșeala medie (pct)"], sc)

        st.subheader(L["hist_detail"])
        pick = st.selectbox(L["pick_gw"], summary["Etapa"].tolist(),
                            format_func=lambda g: L["gw_label"].format(g=g))
        d = detail[detail["etapa"] == pick].copy()
        d["_pos"] = d["position"].map(POS)
        show_table(d, ["web_name", "team_name", "_pos", "adversar", "puncte_prezise", "puncte_reale"],
                   {"web_name": C["player"], "team_name": C["club"], "_pos": L["position"],
                    "adversar": C["opp"], "puncte_prezise": C["pred"], "puncte_reale": C["real"]})

# ---------------- About the model ----------------
with tab_model:
    st.header(L["model_title"])
    st.markdown(L["model_intro"])
    val_file = os.path.join(DATA, "validare_model.csv")
    if os.path.exists(val_file):
        st.dataframe(pd.read_csv(val_file).rename(columns=L["val_cols"]), hide_index=True, width="stretch")
    st.markdown(L["model_notes"])
