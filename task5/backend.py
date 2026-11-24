"""
The ONE place where the app gets teammates' functions.

Anything not delivered yet comes from task5/stubs.py (placeholder data). When a teammate pushes
their real function, change only the matching line below and remove its name from STUBBED.
"""
# Person 1 (Task 2). Real one, once wrapped as a function:  from task2.task2 import outlook
from task5.stubs import outlook

# Person 2 (Task 4): REAL, through a small adapter that renames their columns.
from task5.person2_adapter import search_hotels, nearby, all_hotels, map_hotels

# Person 2 (Task 8): still in progress.  Real one later, e.g.:  from person2.places import places
from task5.stubs import places, all_attractions

# Person 3 (Task 6). Real one:  from task6.best_months import best_months
from task5.stubs import best_months

# Person 3 (shared LLM code). Real one:  from llm import chat, parse_json, numbers_ok
from task5.llm_temp import chat, parse_json, numbers_ok, COUNTER

# Which parts are still placeholders (shown as a warning in the app).
STUBBED = ["outlook (Task 2)", "places (Task 8)", "best_months (Task 6)", "LLM helper (temporary)"]
