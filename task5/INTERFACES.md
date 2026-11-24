# What the web app needs from each teammate

The app (Person 4) already runs on placeholder versions of these functions in `task5/stubs.py`.
When your real function returns **the same name, arguments and columns**, Person 4 changes one
import line in `task5/backend.py` and your part is live in the app. Extra columns are fine;
missing or renamed ones break the app.

## Person 1: Task 2

`outlook(month: int) -> DataFrame` with 15 rows (5 most incidents, 5 largest seasonal increase, 5 fewest)

| column | example |
|---|---|
| list | most_incidents / largest_seasonal_increase / fewest_incidents |
| neighborhood | CAPITOL HILL |
| avg_incidents, avg_persons, avg_property | 472.3, 64.3, 323.3 |
| share_of_city_pct | 7.3 |
| seasonal_ratio | 1.23 |

These are already the columns of `t2_outlook_july.csv`. The only change needed in `task2.py`:
put the table code in `def outlook(month):` that returns `rows`, and move the command-line and LLM
part under `if __name__ == "__main__":` so importing the file doesn't run it.

## Person 2: Task 4 (DONE: plugged in through task5/person2_adapter.py)

Person 2's `search_hotels()` and `find_nearby_hotels()` are used as they are; the adapter only renames
columns. Please keep their argument names and output columns stable, or tell Person 4 before changing them.
The format below is what the adapter produces for the app.

`search_hotels(check_in: str, nights: int, guests: int, budget: float, month: int = None, live: bool = False) -> (DataFrame, dict)`

DataFrame, one row per hotel: `hotel_key, name, lat, lon, neighbourhood, price_<site>` (one column per
booking site, e.g. `price_booking_com`), `cheapest_price, cheapest_site, total, within_budget`

dict (meta): `retrieved_at` (ISO time), `mode` ("live" or "cached"), `age_hours`,
`cheapest` and `cheapest_outside_top5` (each a dict with hotel_key, name, neighbourhood,
cheapest_price, cheapest_site, total, or None), `top5_neighbourhoods` (list), `source` (text)

`live=True` = the demo's live search, limited to about 20 hotels.

`nearby(hotel_key: str, km: float = 2, check_in: str = None) -> DataFrame`
columns `hotel_key, name, neighbourhood, distance_km, cheapest_price, cheapest_site`, cheapest first.

`all_hotels(check_in: str = None) -> DataFrame` every hotel with the same columns as search_hotels
(used for the map and to find a hotel by name).

Optional for Task 7: `API_COUNTER = {"api_calls": 0}` that `cache.get_json` adds 1 to for every real
(non-cached) request, exposed in `task5/backend.py`.

## Person 2: Task 8

`places(month: int, budget: float = None) -> (top10 DataFrame, top5 DataFrame)`

top10: `rank, name, lat, lon, kind, has_wikipedia, neighbourhood, avg_incidents_month,
hotels_within_1km, cheapest_nearby_hotel, cheapest_nearby_price, score`

top5: `rank, neighbourhood, attractions, avg_incidents_month, hotels_in_budget, median_price, score`

`all_attractions(month: int) -> DataFrame` all attractions with at least `name, lat, lon`
(used to find landmarks such as "Pike Place Market" for nearby searches).

## Person 3: Task 6 and shared LLM code

`best_months() -> DataFrame` one row per month:
`month, month_name, avg_temp_f, temp_distance_from_70, avg_incidents, score, rank` (rank 1 = best).
Say in the report whether a higher or lower score is better; the app currently assumes lower.

`chat(messages: list, json_mode: bool = False) -> str` one Groq request, temperature 0, cached.
`parse_json(text) -> dict | None`
`numbers_ok(text, data_text) -> (bool, list_of_bad_numbers)`
plus `COUNTER = {"llm_calls": 0, "llm_cache_hits": 0}`.
`task5/llm_temp.py` has working versions of these you can reuse or replace.
