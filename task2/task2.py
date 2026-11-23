import os, re, sys, json, time, calendar, requests
from pathlib import Path
import pandas as pd

MONTHS = {n.lower(): i for i, n in enumerate(calendar.month_name) if n}
arg = sys.argv[1].lower()
m = int(arg) if arg.isdigit() else MONTHS[arg]
mname = calendar.month_name[m].lower()

d = pd.read_parquet("neighbourhood_month.parquet")
hoods = sorted(d["neighborhood"].unique())
full = pd.MultiIndex.from_product([hoods, [2022, 2023, 2024], range(1, 13)],
                                  names=["neighborhood", "year", "month"])
d = d.set_index(["neighborhood", "year", "month"]).reindex(full).reset_index()
for c in ["total", "persons", "property"]:
    d[c] = d[c].fillna(0)

sel = d[d.month == m].groupby("neighborhood")[["total", "persons", "property"]].mean()
sel.columns = ["avg_incidents", "avg_persons", "avg_property"]
sel["share_of_city_pct"] = sel.avg_incidents / sel.avg_incidents.sum() * 100
sel["seasonal_ratio"] = sel.avg_incidents / d.groupby("neighborhood")["total"].mean()

parts = [sel.nlargest(5, "avg_incidents").assign(list="most_incidents"),
         sel.nlargest(5, "seasonal_ratio").assign(list="largest_seasonal_increase"),
         sel.nsmallest(5, "avg_incidents").assign(list="fewest_incidents")]
rows = pd.concat(parts).reset_index()
rows["avg_incidents"] = rows.avg_incidents.round(1)
rows["avg_persons"] = rows.avg_persons.round(1)
rows["avg_property"] = rows.avg_property.round(1)
rows["share_of_city_pct"] = rows.share_of_city_pct.round(1)
rows["seasonal_ratio"] = rows.seasonal_ratio.round(2)
rows = rows[["list", "neighborhood", "avg_incidents", "avg_persons", "avg_property",
             "share_of_city_pct", "seasonal_ratio"]]
rows.to_csv(f"t2_outlook_{mname}.csv", index=False)
csv_text = rows.to_csv(index=False)
print(rows.to_string(index=False))

def nums(s):
    s = re.sub(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?!\d)", lambda mo: mo.group(0).replace(",", ""), s)
    return {round(float(x), 2) for x in re.findall(r"\d+(?:\.\d+)?", s)}

allowed = {round(float(v), 2) for c in rows.columns if c not in ("list", "neighborhood") for v in rows[c]}
allowed |= {2022.0, 2023.0, 2024.0}

SYSTEM = ("You write short travel summaries from a data table. Use ONLY numbers that appear in the table, "
          "exactly as written. Do not calculate new numbers, do not state counts or rankings as digits, "
          "and do not round. Say 'reported incidents', never 'dangerous' or 'safe'. Do not judge people "
          "or neighbourhoods. Write 120 words or fewer in plain, friendly language. "
          "avg_incidents is the average monthly count for this month across 2022-2024. "
          "seasonal_ratio above 1 means incidents rise in this month compared with the yearly average. "
          "avg_persons and avg_property are average monthly counts of reported incidents classed as crimes "
          "against persons and crimes against property, not numbers of people. Never write 'persons' alone: "
          "say 'person-crime reports' and 'property reports'. For a zero, say 'no reports of that type'. Never write a digit that is not copied exactly from the table: no thresholds like 'above 1' or '300-plus', no counts of neighbourhoods, no ranges. Describe ratios in words (rises, falls) and give numbers only as exact table values. Do not make claims about 'all' or 'every' neighbourhood. Mention only the neighbourhoods in the table. Do not make statements about the city as a whole, such as 'citywide'.")

key = os.environ["GROQ_API_KEY"]
Path("cache/llm").mkdir(parents=True, exist_ok=True)

def ask(messages, tag):
    f = Path(f"cache/llm/t2_{mname}_{tag}.json")
    if f.exists():
        return json.loads(f.read_text())["text"]
    for tries in range(6):
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"}, timeout=120,
            json={"model": "openai/gpt-oss-120b", "temperature": 0,
                  "max_completion_tokens": 4000, "messages": messages})
        if r.status_code != 429:
            break
        wait = float(r.headers.get("retry-after", 20)) + 2
        print(f"rate limited, waiting {wait:.0f}s")
        time.sleep(wait)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()
    f.write_text(json.dumps({"retrieved": time.time(), "text": text}))
    time.sleep(1)
    return text

user = (f"Month: {calendar.month_name[m]}. Table (CSV):\n{csv_text}\n"
        "Write the traveler-friendly summary.")
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
summary, ok = None, False
for attempt in range(4):
    text = ask(msgs, f"a{attempt}")
    bad = sorted(nums(text) - allowed) or ([] if len(text.split()) >= 30 else ["too short or empty"])
    print(f"attempt {attempt}: numbers not in data = {bad}")
    if bad:
        print("   text was:", text.replace("\n", " ")[:700])
    if not bad:
        summary, ok = text, True
        break
    msgs = msgs + [{"role": "assistant", "content": text},
                   {"role": "user", "content": f"These numbers are not in the table: {bad}. "
                    "Rewrite using only numbers copied from the table."}]

src = ("Source: Seattle Police Department incident reports (data.seattle.gov), 2022-2024, "
       "processed with Spark. Summary written by gpt-oss-120b and number-checked in code.")
with open(f"t2_summary_{mname}.md", "w") as f:
    f.write(f"# Neighbourhood outlook: {calendar.month_name[m]}\n\n")
    if ok:
        f.write(summary + "\n\n")
    else:
        f.write("_Summary failed the number check, table shown only._\n\n")
    f.write(rows.to_markdown(index=False))
    f.write("\n\n" + src + "\n")
print("summary passed number check:", ok)
