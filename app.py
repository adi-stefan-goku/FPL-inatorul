"""FPL-inatorul -- site-ul (Streamlit). Porneste-l cu porneste_site.py."""
import os
import subprocess
import sys

import pandas as pd
import streamlit as st

import team_tools as tt

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
PRED_FILE = os.path.join(DATA, "predictii_etapa_urmatoare.csv")
POS_RO = {"GK": "Portar", "DEF": "Fundaș", "MID": "Mijlocaș", "FWD": "Atacant"}

st.set_page_config(page_title="FPL-inatorul", page_icon="⚽", layout="wide")


@st.cache_data
def load_predictions(mtime):
    return pd.read_csv(PRED_FILE)


@st.cache_data
def load_gameweeks(mtime):
    return pd.read_csv(os.path.join(DATA, "gameweeks.csv"))


@st.cache_data(ttl=600, show_spinner=False)
def cached_team(entry_id, gw):
    return tt.fetch_team(entry_id, gw)


@st.cache_data(show_spinner=False)
def cached_transfers(elements, bank, n, metric, mtime):
    pred = load_predictions(mtime)
    pool = pred.copy()
    pool["puncte_prezise"] = pool[metric]
    squad = pool[pool.element.isin(elements)]
    return tt.transfer_suggestions(squad, pool, bank, n_transfers=n)


def show_table(df, cols, names):
    st.dataframe(df[cols].rename(columns=names), hide_index=True, width="stretch")


if not os.path.exists(PRED_FILE):
    st.error("Nu exista predictii. Ruleaza intai fpl_date.py si fpl_model.py.")
    st.stop()

mtime = os.path.getmtime(PRED_FILE)
pred = load_predictions(mtime)
gws = load_gameweeks(mtime)
gw = int(pred["etapa"].iloc[0])
gw_row = gws[gws.id == gw]
deadline = pd.to_datetime(gw_row["deadline_time"].iloc[0]).tz_convert("Europe/London") if not gw_row.empty else None
current_gw = gw - 1

# ---------------- sidebar ----------------
st.sidebar.title("⚽ FPL-inatorul")
st.sidebar.caption(f"Predicții pentru **Etapa {gw}**" +
                   (f" · deadline {deadline:%a %d %b, %H:%M}" if deadline is not None else ""))
horizon = st.sidebar.radio("Orizont pentru transferuri",
                           ["Următoarele 3 etape", "Doar etapa următoare"])
metric = "puncte_3_etape" if horizon.startswith("Următoarele") else "puncte_prezise"
st.sidebar.divider()
if st.sidebar.button("🔄 Actualizează datele (2-3 min)"):
    with st.spinner("Descarc datele noi din FPL și recalculez predicțiile..."):
        for script in ["fpl_date.py", "fpl_model.py"]:
            r = subprocess.run([sys.executable, os.path.join(ROOT, script)], cwd=ROOT,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                st.sidebar.error(f"{script} a eșuat:\n{r.stderr[-800:]}")
                st.stop()
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"Date actualizate: {pd.Timestamp(mtime, unit='s', tz='UTC').tz_convert('Europe/London'):%d %b %H:%M}")

tab_team, tab_top, tab_model = st.tabs(["🧑‍💼 Echipa mea", "📈 Top jucători", "🔬 Cât de bun e modelul"])

# ---------------- Echipa mea ----------------
with tab_team:
    st.header("Echipa mea")
    entry_id = st.text_input("ID-ul echipei tale FPL",
                             help="Pe fantasy.premierleague.com → Points: numărul din adresă, după /entry/")
    if entry_id:
        if not entry_id.strip().isdigit():
            st.error("ID-ul trebuie să fie doar cifre.")
            st.stop()
        try:
            with st.spinner("Citesc echipa din FPL..."):
                team = cached_team(int(entry_id), current_gw)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Nu am putut citi echipa {entry_id} (etapa {current_gw}): {exc}")
            st.stop()

        squad = pred[pred.element.isin(team["elemente"])].copy()
        squad_m = squad.copy()
        squad_m["puncte_prezise"] = squad_m[metric]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Echipa", team["nume_echipa"] or "-")
        c2.metric("Puncte totale", team["puncte_totale"])
        c3.metric("Loc general", f"{team['loc_general']:,}" if team["loc_general"] else "-")
        c4.metric("Bani în bancă", f"£{team['banca']:.1f}m")
        st.caption(f"Lotul e cel din Etapa {current_gw}. Transferurile pe care le-ai făcut deja pentru "
                   f"Etapa {gw} nu se văd public până la deadline.")

        # echipa de start pentru etapa urmatoare (mereu pe etapa urmatoare)
        xi, bench, xi_pts = tt.best_xi(squad)
        cap = xi.sort_values("puncte_prezise", ascending=False)
        st.subheader(f"Echipa de start recomandată -- Etapa {gw}")
        if len(cap) >= 2:
            st.success(f"**Căpitan: {cap.iloc[0].web_name}** ({cap.iloc[0].puncte_prezise:.1f} pct) · "
                       f"Vice: {cap.iloc[1].web_name} ({cap.iloc[1].puncte_prezise:.1f} pct) · "
                       f"Total prezis (cu căpitan dublat): **{xi_pts + cap.iloc[0].puncte_prezise:.1f} pct**")
        order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
        xi = xi.assign(_o=xi.position.map(order)).sort_values(["_o", "puncte_prezise"], ascending=[True, False])
        xi["Poziție"] = xi.position.map(POS_RO)
        cols = ["Poziție", "web_name", "team_name", "adversar", "puncte_prezise", "puncte_3_etape", "price"]
        names = {"web_name": "Jucător", "team_name": "Echipa", "adversar": "Adversar",
                 "puncte_prezise": f"Pct Etapa {gw}", "puncte_3_etape": "Pct 3 etape", "price": "Preț"}
        show_table(xi, cols, names)
        st.markdown("**Rezerve** (în ordinea de intrare)")
        bench["Poziție"] = bench.position.map(POS_RO)
        show_table(bench, cols, names)
        injured = squad[squad.status.isin(["i", "s", "u", "d", "n"])]
        if not injured.empty:
            st.warning("⚠️ Probleme de disponibilitate: " + ", ".join(
                f"{r.web_name} ({'' if pd.isna(r.chance_of_playing_next_round) else str(int(r.chance_of_playing_next_round)) + '%'} {r.status})"
                for r in injured.itertuples()))

        st.subheader("Transferuri recomandate")
        st.caption(f"Câștigul = puncte prezise în plus pentru echipa de start + căpitan "
                   f"({horizon.lower()}). Prețul de vânzare e aproximat cu prețul curent "
                   f"(API-ul public nu arată prețul tău exact de vânzare).")
        n = int(st.radio("Câte transferuri?", ["1", "2"], horizontal=True))
        with st.spinner("Caut cele mai bune transferuri (10-30 sec)..."):
            sugg = cached_transfers(tuple(team["elemente"]), team["banca"], n, metric, mtime)
        if sugg.empty:
            st.info("Nu am găsit niciun transfer care să-ți îmbunătățească echipa.")
        else:
            if n == 1:
                show_table(sugg, ["iese", "intra", "intra_echipa", "pret_iese", "pret_intra",
                                  "banca_ramasa", "castig"],
                           {"iese": "Iese", "intra": "Intră", "intra_echipa": "Echipa",
                            "pret_iese": "Preț iese", "pret_intra": "Preț intră",
                            "banca_ramasa": "Bancă după", "castig": "Câștig (pct)"})
            else:
                show_table(sugg, ["transfer_1", "transfer_2", "banca_ramasa", "castig"],
                           {"transfer_1": "Transfer 1", "transfer_2": "Transfer 2",
                            "banca_ramasa": "Bancă după", "castig": "Câștig (pct)"})
                st.caption("Dacă ai un singur transfer gratuit, al doilea costă -4 puncte: "
                           "merită doar dacă câștigul total e clar peste 4.")
            best = sugg["castig"].iloc[0]
            if best < 1:
                st.info(f"Cel mai bun câștig e doar {best:.1f} pct -- în limita zgomotului. "
                        "Poți păstra transferul pentru săptămâna viitoare (se adună până la 5).")

# ---------------- Top jucatori ----------------
with tab_top:
    st.header(f"Top jucători -- Etapa {gw}")
    c1, c2, c3 = st.columns(3)
    pos = c1.selectbox("Poziție", ["Toate"] + list(POS_RO.values()))
    max_price = c2.slider("Preț maxim (£m)", 4.0, float(pred.price.max()), float(pred.price.max()), 0.5)
    sort_by = c3.radio("Sortează după", ["Etapa următoare", "3 etape"], horizontal=True)
    view = pred[pred.price <= max_price].copy()
    if pos != "Toate":
        view = view[view.position.map(POS_RO) == pos]
    view = view.sort_values("puncte_prezise" if sort_by == "Etapa următoare" else "puncte_3_etape",
                            ascending=False).head(30)
    view["Poziție"] = view.position.map(POS_RO)
    gw_cols = [c for c in pred.columns if c.startswith("adversar_etapa_")]
    show_table(view, ["web_name", "team_name", "Poziție", "price", "puncte_prezise", "puncte_3_etape"]
               + gw_cols + ["selected_by_percent"],
               {"web_name": "Jucător", "team_name": "Echipa", "price": "Preț",
                "puncte_prezise": f"Pct Etapa {gw}", "puncte_3_etape": "Pct 3 etape",
                "selected_by_percent": "Deținut %",
                **{c: "Etapa " + c.split("_")[-1] for c in gw_cols}})

# ---------------- Despre model ----------------
with tab_model:
    st.header("Cât de bun e modelul")
    st.markdown("""
Modelul e antrenat pe toate meciurile din ultimele 4 sezoane (forma, minute, xG, xA,
bonus, puterea echipei și a adversarului) și testat **pe sezoane pe care nu le-a văzut**.
Îl comparăm cu metoda simplă: media punctelor din ultimele 6 meciuri.
""")
    val_file = os.path.join(DATA, "validare_model.csv")
    if os.path.exists(val_file):
        st.dataframe(pd.read_csv(val_file), hide_index=True, width="stretch")
    st.markdown("""
- **MAE** = greșeala medie în puncte per jucător (mai mic = mai bine).
- **Top10** = câte puncte au făcut în medie cei 10 jucători aleși de metodă în fiecare etapă.
- **Căpitan** = câte puncte a făcut jucătorul nr. 1 ales. Aici modelul **nu e încă dovedit**
  (un singur jucător pe etapă e foarte aleator).
- Predicțiile pentru etapele 2 și 3 înainte păstrează forma de acum și schimbă doar adversarul --
  sunt o estimare, nevalidate separat.
""")
