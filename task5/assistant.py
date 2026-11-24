"""
Task 5: the natural-language assistant (no Streamlit code here, so Task 7 can test it directly).

One call to handle(message, state) does the whole pipeline:

  1. EXTRACT   the LLM turns the message into JSON (intent + the 7 required fields + near)
  2. VALIDATE  plain Python checks types and ranges (bad values become follow-up questions)
  3. DECIDE    decline (not Seattle / off-topic), or merge into the fields collected so far
               and ask a follow-up if something required is still missing
  4. RUN       call the teammates' functions (Task 2, 4, 6, 8) for this intent
  5. ANSWER    send ONLY those results to the LLM, which writes the reply
  6. CHECK     every number in the reply must appear in the data, else retry / show tables only

`state` is a plain dict the app keeps in st.session_state, so it survives Streamlit reruns.
"""
import calendar
import json
import re
import time
from datetime import date, timedelta

import pandas as pd

from task5 import backend as B

CITY = "Seattle"
FIELDS = ["city", "check_in", "nights", "guests", "max_price_per_night", "month", "preferences"]
INTENTS = ["outlook", "hotels", "nearby", "best_month", "places", "out_of_scope"]

# Which fields each kind of request needs before it can be answered.
REQUIRED = {
    "outlook": ["month"],
    "hotels": ["check_in", "nights", "guests", "max_price_per_night"],
    "nearby": ["near", "check_in", "nights", "guests", "max_price_per_night"],
    "best_month": [],
    "places": ["month"],
}
PURPOSE = {
    "outlook": "show reported incidents by neighbourhood",
    "hotels": "search hotel prices",
    "nearby": "find hotels near that spot",
    "places": "recommend places and neighbourhoods",
}
ASK_TEXT = {
    "check_in": "your check-in date (for example 12 March 2027)",
    "nights": "how many nights you're staying",
    "guests": "how many guests",
    "max_price_per_night": "your maximum price per night in US dollars",
    "month": "which month you're interested in",
    "near": "which hotel or landmark you'd like to stay near",
}
CAPABILITIES = ("I can help with trips to Seattle: reported incidents by neighbourhood for a month, "
                "hotel prices (cheapest options and nearby alternatives), the best month to visit, "
                "and places to visit.")


def new_state():
    """A fresh conversation (the app's 'New conversation' button calls this)."""
    return {"collected": {f: None for f in FIELDS} | {"preferences": [], "near": None},
            "intent": None, "pending": None, "turns": []}


# ------------------------------------------------------------------ 1. extract
def date_hints(today=None):
    """Concrete dates for relative phrases, given to the LLM (and reused by Task 7 labels)."""
    today = today or date.today()
    this_sat = today + timedelta(days=(5 - today.weekday()) % 7 or 7)
    first_next = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
    return {"today": today.isoformat(), "weekday": today.strftime("%A"),
            "tomorrow": (today + timedelta(days=1)).isoformat(),
            "this_weekend": this_sat.isoformat(), "next_weekend": (this_sat + timedelta(days=7)).isoformat(),
            "next_month": first_next.month}


def extract_prompt(today=None):
    h = date_hints(today)
    return f"""You turn a traveller's message into JSON for SafeStay, a trip planner that only covers Seattle.
Today is {h['weekday']} {h['today']}.
Return one JSON object with exactly these keys:
- "intent": one of
    "outlook" (which neighbourhoods had more or fewer reported incidents in a month),
    "hotels" (find or compare hotel prices, cheapest hotel, booking requests),
    "nearby" (hotels near a named hotel or landmark, or cheaper hotels nearby),
    "best_month" (best month or time of year to visit),
    "places" (attractions, things to do, best neighbourhoods to stay in or explore),
    "out_of_scope" (anything else: forecasts, restaurants, flights, general chat, other topics),
  or null if the message only answers the assistant's last question.
- "city": the city the user names, as written, else null. Do not assume a city.
- "check_in": "YYYY-MM-DD" only if the user gives a specific day. "tomorrow" = {h['tomorrow']},
  "this weekend" = {h['this_weekend']}, "next weekend" = {h['next_weekend']}. A day without a year means its
  next future date. A month alone ("in July") is NOT a check-in date: set month and leave check_in null.
- "nights": whole number of nights, or null.
- "guests": number of guests ("two adults" = 2, "just me" = 1), or null.
- "max_price_per_night": nightly budget in US dollars as a number ("under $150 a night" = 150), or null.
- "month": 1-12 if a month is named ("next month" = {h['next_month']}) or check_in is given, else null.
- "near": the hotel or landmark the user wants to stay near, else null.
- "preferences": list of short strings for other wishes (e.g. "museums", "compare Booking.com and Agoda"), else [].
Use null for anything the user did not say. Never guess. Reply with the JSON object only."""


def extract(message, state, today=None):
    """Steps 1 and 2. Returns (clean_fields, problems) or (None, ...) if the JSON was unusable."""
    user = (f"Fields collected so far: {json.dumps(state['collected'])}\n"
            f"Assistant's last question: {state['pending'] or 'none'}\n"
            f"New user message: {message}")
    msgs = [{"role": "system", "content": extract_prompt(today)}, {"role": "user", "content": user}]
    for _ in range(2):                                   # retry once if the JSON is broken
        raw = B.parse_json(B.chat(msgs, json_mode=True))
        if isinstance(raw, dict):
            return validate(raw, today)
        msgs = msgs + [{"role": "user", "content": "That was not valid JSON. Reply with the JSON object only."}]
    return None, ["unreadable"]


# ------------------------------------------------------------------ 2. validate
def _int(v):
    try:
        f = float(str(v).replace("$", "").replace(",", ""))
        return int(f) if f == int(f) else None
    except (TypeError, ValueError):
        return None


def validate(raw, today=None):
    """Check every field in code. Wrong values are dropped and explained (-> follow-up question)."""
    today = today or date.today()
    out, problems = {}, []

    intent = raw.get("intent")
    out["intent"] = intent if intent in INTENTS else None

    city = raw.get("city")
    out["city"] = city.strip() if isinstance(city, str) and city.strip() else None

    out["check_in"] = None
    if raw.get("check_in"):
        try:
            d = date.fromisoformat(str(raw["check_in"])[:10])
            if d <= today:
                problems.append(f"the check-in date {d.isoformat()} has already passed")
            elif d > today + timedelta(days=335):
                problems.append(f"{d.isoformat()} is too far ahead (prices exist only about 11 months out)")
            else:
                out["check_in"] = d.isoformat()
        except ValueError:
            problems.append("I couldn't read the check-in date")

    for f, lo, hi in [("nights", 1, 30), ("guests", 1, 10)]:
        v = _int(raw.get(f))
        out[f] = v if v is not None and lo <= v <= hi else None
        if raw.get(f) is not None and out[f] is None:
            problems.append(f"{f} must be a whole number from {lo} to {hi}")

    try:
        p = float(str(raw.get("max_price_per_night")).replace("$", "").replace(",", ""))
        out["max_price_per_night"] = p if 20 <= p <= 3000 else None
    except (TypeError, ValueError):
        out["max_price_per_night"] = None

    m = raw.get("month")
    if isinstance(m, str) and m.strip().lower() in [n.lower() for n in calendar.month_name if n]:
        m = [n.lower() for n in calendar.month_name].index(m.strip().lower())
    m = _int(m)
    out["month"] = m if m and 1 <= m <= 12 else None

    near = raw.get("near")
    out["near"] = near.strip() if isinstance(near, str) and near.strip() else None

    prefs = raw.get("preferences") or []
    if isinstance(prefs, str):
        prefs = [prefs]
    out["preferences"] = [str(p).strip() for p in prefs if str(p).strip()]
    return out, problems


# ------------------------------------------------------------------ 3. decide
def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def merge(state, new):
    """New non-empty values overwrite; empty ones never erase what we already know."""
    c = state["collected"]
    for k in ["city", "check_in", "nights", "guests", "max_price_per_night", "month", "near"]:
        if new.get(k) is not None:
            c[k] = new[k]
    for p in new.get("preferences", []):
        if p not in c["preferences"]:
            c["preferences"].append(p)
    if new.get("intent") and new["intent"] != "out_of_scope":
        state["intent"] = new["intent"]
    if c["check_in"]:                                    # a check-in date fixes the month
        c["month"] = int(c["check_in"][5:7])
    if c["city"] is None:                                # Seattle is the only city we cover
        c["city"] = CITY


def is_seattle(city):
    return city is None or re.sub(r"[^a-z]", "", city.lower()).startswith("seattle")


# ------------------------------------------------------------------ 4. run the tasks
def _resolve_near(name, hotels, month):
    """Turn 'Pike Place Market' or a hotel name into one of our hotels. Returns (row, where)."""
    key = name.lower()
    names = hotels.name.str.lower()
    hit = hotels[names.str.contains(re.escape(key)) | names.apply(lambda n: n in key)]
    if not hit.empty:
        return hit.iloc[0], hit.iloc[0]["name"]
    att = B.all_attractions(month or 1)
    a = att[att.name.str.lower().str.contains(re.escape(key)) | att.name.str.lower().apply(lambda n: n in key)]
    if a.empty:
        return None, None
    lat, lon = a.iloc[0].lat, a.iloc[0].lon
    d = ((hotels.lat - lat) ** 2 + ((hotels.lon - lon) * 0.67) ** 2)   # rough: just to pick the closest
    return hotels.loc[d.idxmin()], a.iloc[0]["name"]


def _price_source(meta):
    return (f"Hotel prices: {meta['mode']}, retrieved {str(meta['retrieved_at'])[:16].replace('T', ' ')} UTC "
            f"({meta['age_hours']} hours old). {meta.get('source', '')}")


def run_tools(intent, c, live=False):
    """Call the right teammate functions. Returns (facts for the LLM, tables to show, sources)."""
    month_name = calendar.month_name[c["month"]] if c.get("month") else None
    incidents_src = ("Reported incidents: Seattle Police Department, data.seattle.gov, 2022-2024, "
                     "processed with Spark (Tasks 1-2).")

    if intent == "outlook":
        df = B.outlook(c["month"])
        return ({"month": month_name, "neighbourhood_outlook": df.to_dict("records")},
                [(f"Reported incidents in {month_name} (monthly average 2022-2024)", df)], [incidents_src])

    if intent == "best_month":
        df = B.best_months()
        return ({"months_ranked": df.to_dict("records")},
                [("Best months to visit (lower score is better)", df)],
                ["Temperatures: city_temperature.csv, Seattle 2020-2024 (Homework 2/4).", incidents_src])

    if intent == "places":
        top10, top5 = B.places(c["month"], c.get("max_price_per_night"))
        t10 = top10.drop(columns=["lat", "lon"])
        return ({"month": month_name, "nightly_budget": c.get("max_price_per_night"),
                 "top_places": t10.to_dict("records"), "top_neighbourhoods": top5.to_dict("records")},
                [(f"Top places to visit in {month_name}", t10), ("Top neighbourhoods", top5)],
                ["Attractions: © OpenStreetMap contributors (Overpass API).", incidents_src])

    if intent == "hotels":
        df, meta = B.search_hotels(c["check_in"], c["nights"], c["guests"], c["max_price_per_night"], c["month"],
                                   live=live)
        # The 4 booking sites with the most prices (all sites are kept in Person 2's t4_hotels.csv)
        price_cols = df[[x for x in df.columns if x.startswith("price_")]].notna().sum() \
            .sort_values(ascending=False).index[:4].tolist()
        show = df[df.within_budget].head(10)
        note = None
        if show.empty:
            show, note = df.head(5), "No hotel is within the budget; these are the cheapest overall."
        show = show[["name", "neighbourhood", *price_cols, "cheapest_price", "cheapest_site", "total"]]
        facts = {"check_in": c["check_in"], "nights": c["nights"], "guests": c["guests"],
                 "nightly_budget": c["max_price_per_night"], "note": note,
                 "cheapest_within_budget": meta["cheapest"],
                 "cheapest_within_budget_outside_top5_incident_areas": meta["cheapest_outside_top5"],
                 "top5_incident_neighbourhoods_that_month": meta["top5_neighbourhoods"],
                 "hotels": show.to_dict("records")}
        return facts, [("Hotels and prices per night by booking site", show)], [_price_source(meta), incidents_src]

    if intent == "nearby":
        df, meta = B.search_hotels(c["check_in"], c["nights"], c["guests"], c["max_price_per_night"], c["month"],
                                   live=live)
        hotel, where = _resolve_near(c["near"], df, c["month"])
        if hotel is None:
            return None, [], []
        # distances from Person 2's nearby(); prices from THIS search (the user's dates)
        nb = B.nearby(hotel["hotel_key"], km=2, check_in=c["check_in"])[["hotel_key", "name", "neighbourhood",
                                                                            "distance_km"]]
        nb = nb.merge(df[["hotel_key", "cheapest_price", "cheapest_site"]], on="hotel_key", how="left")
        nb = nb.sort_values(["cheapest_price", "distance_km"], na_position="last").reset_index(drop=True)
        nb["within_budget"] = nb["cheapest_price"] <= c["max_price_per_night"]
        me = {k: hotel[k] for k in ["name", "neighbourhood", "cheapest_price", "cheapest_site"]}
        me["total"] = None if pd.isna(hotel["cheapest_price"]) else hotel["cheapest_price"] * c["nights"]
        facts = {"near": where, "check_in": c["check_in"], "nights": c["nights"],
                 "nightly_budget": c["max_price_per_night"], "chosen_hotel": me,
                 "hotels_within_2_km": nb.drop(columns=["hotel_key"]).to_dict("records")}
        return (facts, [(f"Hotels within 2 km of {hotel['name']} (cheapest first)", nb.drop(columns=["hotel_key"]))],
                [_price_source(meta), "Distances: haversine formula (Task 4)."])
    raise ValueError(intent)


# ------------------------------------------------------------------ 5 + 6. answer and check
ANSWER_SYSTEM = """You are SafeStay's assistant for travellers to Seattle. Write a short, friendly answer
(at most 150 words) to the user's request using ONLY the facts in the JSON data.
Rules:
- Every number you write must be copied exactly from the data. Do not calculate, round or estimate,
  do not count items, and do not write numbers as words.
- Do not invent hotels, places, neighbourhoods or prices.
- Say "reported incidents". Never call a place safe, unsafe or dangerous, and do not judge the people who live there.
- Prices are per night in US dollars unless the field is "total". You only search and compare prices:
  never say anything was booked or reserved.
- If the data contains a note (for example no hotel within budget), say it.
- Do not list sources; the app shows them under your answer."""


def write_answer(request_text, fields, facts):
    """Ask the LLM for the reply, then number-check it. Returns (text or None, passed, bad)."""
    data = json.dumps(facts, default=str)
    allowed = data + " " + json.dumps(fields, default=str) + " " + request_text
    msgs = [{"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"User request: {request_text}\nRequest fields: {json.dumps(fields)}\n"
                                        f"Data (JSON):\n{data}"}]
    bad = []
    for _ in range(3):                                   # first try + 2 retries
        text = B.chat(msgs)
        ok, bad = B.numbers_ok(text, allowed)
        if ok and len(text.split()) >= 8:
            return text, True, []
        msgs = msgs + [{"role": "assistant", "content": text},
                       {"role": "user", "content": f"These are not in the data: {bad}. Rewrite using only "
                                                   "numbers copied exactly from the data, as digits."}]
    return None, False, bad


# ------------------------------------------------------------------ the whole pipeline
def handle(message, state, today=None):
    """Process one user message. Returns a reply dict the app (or Task 7) can display/record:
    kind ("answer" | "ask" | "decline" | "error"), text, tables, sources, fields, intent,
    number_check (True/False/None), seconds, llm_calls."""
    t0, calls0 = time.time(), B.COUNTER["llm_calls"]
    reply = {"kind": None, "text": "", "tables": [], "sources": [], "number_check": None}

    def done(kind, text, **kw):
        reply.update(kind=kind, text=text, **kw)
        reply.update(fields=dict(state["collected"]), intent=state["intent"],
                     seconds=round(time.time() - t0, 2), llm_calls=B.COUNTER["llm_calls"] - calls0)
        return reply

    try:
        new, problems = extract(message, state, today)
    except Exception as e:                               # LLM down, no key, network error...
        return done("error", f"The language model is not reachable right now ({type(e).__name__}). "
                             "Please try again in a moment.")
    if new is None:
        return done("error", "Sorry, I couldn't understand that. Could you rephrase it?")

    # Decline: another city, or a topic we don't cover.
    if not is_seattle(new["city"]):
        return done("decline", f"Sorry, SafeStay only covers Seattle, so I can't help with {new['city']}. "
                               + CAPABILITIES)
    if new["intent"] == "out_of_scope" or (new["intent"] is None and state["intent"] is None):
        return done("decline", "Sorry, that's outside what I can help with. " + CAPABILITIES)

    merge(state, new)
    state["turns"].append(message)
    intent, c = state["intent"], state["collected"]

    booking_note = ("I can search and compare prices, but I can't make reservations or take payment. "
                    if re.search(r"\b(book|reserve|pay|card)\b", message, re.I) else "")

    # Ask: something required is missing or was invalid.
    missing = [f for f in REQUIRED[intent] if not c.get(f)]
    if problems or missing:
        parts = []
        if problems:
            parts.append("It looks like " + _join(problems) + ".")
        if missing:
            parts.append(f"To {PURPOSE[intent]}, could you tell me {_join([ASK_TEXT[f] for f in missing])}?")
        state["pending"] = " ".join(parts)
        return done("ask", booking_note + state["pending"])

    # Run the tasks.
    facts, tables, sources = run_tools(intent, c, live=state.get("live", False))
    if facts is None:                                    # nearby: the place wasn't found
        c["near"] = None
        state["pending"] = ("I couldn't find that hotel or landmark in my list. "
                            "Which hotel or landmark would you like to stay near?")
        return done("ask", state["pending"])

    request_text = " ".join(state["turns"])
    text, ok, bad = write_answer(request_text, {k: c[k] for k in FIELDS}, facts)
    if not ok:
        text = ("Here are the results from our data. (The written summary failed the number check, "
                "so only the tables are shown.)")
    sources = sources + ["Answer written by openai/gpt-oss-120b on Groq from these tables only"
                         + (" and number-checked in code." if ok else ".")]
    if B.STUBBED:
        sources.append("Placeholder data still used for: " + ", ".join(B.STUBBED) + ".")
    state["pending"], state["turns"] = None, []
    return done("answer", booking_note + text, tables=tables, sources=sources,
                number_check=ok, failed_numbers=bad)
