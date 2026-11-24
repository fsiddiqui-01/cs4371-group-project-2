"""
Offline test of the assistant logic with a FAKE LLM (no API key, no internet needed).
Run from the repo root:   python -m task5.test_offline

The fake LLM returns the JSON we tell it to, so this checks OUR code: validation, merging
across turns, follow-up questions, declines, routing to the right task, and the number check.
"""
import json
from datetime import date

from task5 import assistant as A
from task5 import backend as B

TODAY = date(2026, 12, 1)
queue = []          # extraction replies to hand out, in order


def fake_chat(messages, json_mode=False, use_cache=True):
    B.COUNTER["llm_calls"] += 1
    if json_mode:
        return json.dumps(queue.pop(0))
    if "Data (JSON)" in messages[1]["content"] and len(messages) == 2:
        # first answer attempt: deliberately include a made-up number to test the retry
        return "There were 999.9 reported incidents somewhere, which is a fake figure for this test."
    data = json.loads(messages[1]["content"].split("Data (JSON):\n", 1)[1])
    first = json.dumps(data)
    import re
    n = re.findall(r"\d+\.\d+", first)[0]
    return f"Here is a short summary from the data: one value in the table is {n}, see the table below."


B.chat = fake_chat
# Offline: use the placeholder hotels instead of Person 2's live Xotelo search.
from task5 import stubs as S
for name in ["search_hotels", "nearby", "all_hotels", "map_hotels"]:
    setattr(B, name, getattr(S, name))


def blank(**kw):
    d = {"intent": None, "city": None, "check_in": None, "nights": None, "guests": None,
         "max_price_per_night": None, "month": None, "near": None, "preferences": []}
    d.update(kw)
    return d


def run(title, turns):
    print(f"\n=== {title}")
    state = A.new_state()
    for msg, extracted in turns:
        queue.append(extracted)
        r = A.handle(msg, state, today=TODAY)
        print(f"USER: {msg}\n  -> {r['kind']}: {r['text'][:150]}")
        print(f"     intent={r['intent']} number_check={r['number_check']} llm_calls={r['llm_calls']}")
        for t, df in r["tables"]:
            print(f"     table: {t} ({len(df)} rows)")
    return state, r


# 1. complete hotel request
_, r = run("hotel search", [("Cheapest hotel in Seattle for two nights from 12 March, two adults, under $150 a night.",
                             blank(intent="hotels", city="Seattle", check_in="2027-03-12", nights=2, guests=2,
                                   max_price_per_night=150, month=3))])
assert r["kind"] == "answer" and r["number_check"] is True and r["llm_calls"] == 3   # extract + bad + good

# 2. missing fields, then the user answers (merge across turns)
s, r = run("missing fields", [
    ("Book me something nice in Seattle.", blank(intent="hotels", city="Seattle")),
    ("10 April 2027, 2 nights, just me, max $200", blank(check_in="2027-04-10", nights=2, guests=1,
                                                         max_price_per_night=200, month=4))])
assert r["kind"] == "answer" and s["collected"]["guests"] == 1 and s["intent"] == "hotels"

# 3. Miami must be declined
_, r = run("other city", [("What's the safest hotel in Miami?", blank(intent="hotels", city="Miami"))])
assert r["kind"] == "decline"

# 4. off-topic must be declined
_, r = run("off topic", [("Write me a poem about rain", blank(intent="out_of_scope"))])
assert r["kind"] == "decline"

# 5. past date -> follow-up question
_, r = run("past date", [("Hotel checking in 1 Nov 2026 for 2 nights, 2 guests, $150",
                          blank(intent="hotels", check_in="2026-11-01", nights=2, guests=2, max_price_per_night=150))])
assert r["kind"] == "ask" and "passed" in r["text"]

# 6. places without a month -> ask, then answer
_, r = run("places", [("Which neighbourhood is best to stay in for museums, under $200 a night?",
                       blank(intent="places", max_price_per_night=200, preferences=["museums"])),
                      ("August", blank(month=8))])
assert r["kind"] == "answer" and len(r["tables"]) == 2

# 7. nearby a landmark
_, r = run("nearby", [("Hotels near Pike Place Market for 3 nights from 2027-01-08, 2 guests, under $250",
                       blank(intent="nearby", near="Pike Place Market", check_in="2027-01-08", nights=3,
                             guests=2, max_price_per_night=250))])
assert r["kind"] == "answer"

# 8. outlook and best month
_, r = run("outlook", [("Which neighbourhoods saw incidents rise the most in December?", blank(intent="outlook", month=12))])
assert r["kind"] == "answer"
_, r = run("best month", [("When is the best month to visit Seattle?", blank(intent="best_month", city="Seattle"))])
assert r["kind"] == "answer"

# 9. number check catches words and wrong numbers
assert B.numbers_ok("about three hundred incidents", "300")[0] is False
assert B.numbers_ok("$1,250 total for 2 nights", '{"total": 1250, "nights": 2}')[0] is True
print("\nALL OFFLINE TESTS PASSED")
