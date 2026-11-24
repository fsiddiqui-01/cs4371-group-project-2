"""
SafeStay Seattle: the Task 5 web app.   Run:  streamlit run app.py

Streamlit runs this whole file again on EVERY click or message. Anything that must
survive (the conversation, the fields collected so far) lives in st.session_state.
Slow data loading is wrapped in st.cache_data so it runs once, not on every rerun.
"""
import calendar
import pandas as pd
import streamlit as st

from task5 import assistant as A
from task5 import backend as B
from task5.llm_temp import load_env

load_env()                                   # API keys come from .env / environment, never the UI
st.set_page_config(page_title="SafeStay Seattle", page_icon="🧳", layout="wide")

# Cache teammates' data functions so repeat questions don't reload anything.
# (search_hotels is NOT cached here: its own disk cache decides live vs cached prices.)
for name in ["outlook", "best_months", "places", "all_attractions", "all_hotels", "nearby", "map_hotels"]:
    if not hasattr(getattr(B, name), "clear"):           # wrap only once per server
        setattr(B, name, st.cache_data(show_spinner=False)(getattr(B, name)))

# ----------------------------------------------------------------- state that survives reruns
if "chat" not in st.session_state:
    st.session_state.chat = []               # what is shown: {"role", "content", "tables", "sources"}
    st.session_state.state = A.new_state()   # what the assistant knows (fields, intent, pending question)


def md(text):
    """Streamlit treats $...$ as maths; escape dollar signs so prices display correctly."""
    return text.replace("$", "\\$")


def show(m):
    with st.chat_message(m["role"]):
        st.markdown(md(m["content"]))
        for title, df in m.get("tables", []):
            st.caption(title)
            st.dataframe(df, hide_index=True, width="stretch")
        if m.get("sources"):
            st.caption("**Sources:** " + md(" · ".join(m["sources"])))


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧳 SafeStay Seattle")
    st.caption("Plan a stay in Seattle using open city data. Prices are searched and compared, never booked.")
    if st.button("🔄 New conversation", width="stretch"):
        st.session_state.chat = []
        st.session_state.state = A.new_state()
        st.rerun()
    st.session_state.state["live"] = st.toggle("Live hotel prices (up to 20 hotels)", value=False,
                                               help="Off = use cached prices, labelled with their age.")
    st.subheader("Details collected so far")
    c = st.session_state.state["collected"]
    st.json({"intent": st.session_state.state["intent"], **{k: v for k, v in c.items()}}, expanded=False)
    if B.STUBBED:
        st.warning("Placeholder data still in use for: " + ", ".join(B.STUBBED))

chat_tab, map_tab = st.tabs(["💬 Chat", "🗺️ Map"])

# ----------------------------------------------------------------- chat page
with chat_tab:
    if not st.session_state.chat:
        st.info('Try: "I\'m going to Seattle for three nights from 10 July 2027, 2 guests, '
                'budget $180 a night. What\'s the cheapest hotel in a calmer area?"')
    for m in st.session_state.chat:
        show(m)

    prompt = st.chat_input("Ask about a trip to Seattle…")
    if prompt:
        user_msg = {"role": "user", "content": prompt}
        st.session_state.chat.append(user_msg)
        show(user_msg)
        with st.spinner("Working on it…"):
            r = A.handle(prompt, st.session_state.state)
        bot_msg = {"role": "assistant", "content": r["text"], "tables": r["tables"], "sources": r["sources"]}
        st.session_state.chat.append(bot_msg)
        st.rerun()                           # redraw so the sidebar shows the updated fields

# ----------------------------------------------------------------- map page
with map_tab:
    col1, col2 = st.columns(2)
    month = col1.selectbox("Month", list(range(1, 13)), index=6, format_func=lambda m: calendar.month_name[m])
    budget = col2.number_input("Max price per night (USD)", min_value=50, max_value=1000, value=200, step=10)

    with st.spinner("Loading hotels and places…"):
        hotels, meta = B.map_hotels(budget)          # saved Task 4 prices: no API calls on every click
        top10, top5 = B.places(month, budget)

    in_budget = hotels[hotels.within_budget]
    pts = pd.concat([
        pd.DataFrame({"lat": in_budget.lat, "lon": in_budget.lon, "color": "#1f77b4", "size": 40}),
        pd.DataFrame({"lat": top10.lat, "lon": top10.lon, "color": "#e4572e", "size": 70}),
    ])
    st.map(pts, latitude="lat", longitude="lon", color="color", size="size")
    st.caption(f"🔵 {len(in_budget)} hotels within \\${budget} a night (saved prices for {meta['dates']})  ·  "
               f"🔴 top 10 places to visit in {calendar.month_name[month]}  ·  Map data © OpenStreetMap contributors")

    st.subheader(f"Top 10 places to visit in {calendar.month_name[month]}")
    st.dataframe(top10.drop(columns=["lat", "lon"]), hide_index=True, width="stretch")
    st.subheader("Top 5 neighbourhoods")
    st.dataframe(top5, hide_index=True, width="stretch")
    st.caption("**Sources:** " + md(
        f"Attractions: © OpenStreetMap contributors (Overpass API). "
        f"Reported incidents: Seattle Police Department, data.seattle.gov, 2022-2024 (Tasks 1-2). "
        f"{A._price_source(meta)}"))
