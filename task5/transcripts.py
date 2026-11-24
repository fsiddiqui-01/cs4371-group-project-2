"""
Task 5 output: record complete conversations into t5_transcripts.md.

Run from the repo root (needs GROQ_API_KEY):   python -m task5.transcripts
Includes a missing-field conversation and an out-of-scope request, as the assignment requires.
You can also copy real conversations from the web app into the file by hand.
"""
from datetime import datetime
from pathlib import Path

from task5 import assistant as A

ROOT = Path(__file__).resolve().parent.parent

CONVERSATIONS = [
    ("Hotel search with a budget",
     ["I'm going to Seattle for three nights from 10 July 2027 with a budget of $180 a night, 2 guests. "
      "Which areas had more reported incidents in July, and what is the cheapest hotel in a calmer area?"]),
    ("Missing field: the assistant asks a follow-up",
     ["Book me something nice in Seattle.", "Checking in 10 April 2027 for 2 nights, just me, max $200 a night."]),
    ("Out of scope: another city",
     ["What's the safest hotel in Miami?"]),
    ("Best month to visit",
     ["When is the best month to visit Seattle?"]),
    ("Places to visit, month asked for",
     ["Which neighbourhood is best to stay in for museums, under $200 a night?", "August"]),
    ("Nearby alternatives",
     ["Hotels near the Space Needle under $220 a night for 2 nights from 8 August 2027, 2 adults."]),
]


def table_md(df, limit=10):
    return df.head(limit).to_markdown(index=False)


def main():
    out = [f"# Task 5 transcripts\n\nRecorded {datetime.now():%Y-%m-%d %H:%M} with the SafeStay assistant "
           "(main LLM: openai/gpt-oss-120b on Groq).\n"]
    for title, turns in CONVERSATIONS:
        print("recording:", title)
        state = A.new_state()
        out.append(f"\n## {title}\n")
        for msg in turns:
            r = A.handle(msg, state)
            out.append(f"**User:** {msg}\n")
            out.append(f"**Assistant** _({r['kind']}; number check: {r['number_check']}; "
                       f"{r['llm_calls']} LLM calls; {r['seconds']} s)_: {r['text']}\n")
            for t, df in r["tables"]:
                out.append(f"_{t}_\n\n{table_md(df)}\n")
            if r["sources"]:
                out.append("Sources: " + " · ".join(r["sources"]) + "\n")
            out.append(f"<details><summary>Fields after this turn</summary>\n\n`{r['fields']}`\n\n</details>\n")
    (ROOT / "t5_transcripts.md").write_text("\n".join(out))
    print("Wrote t5_transcripts.md")


if __name__ == "__main__":
    main()
