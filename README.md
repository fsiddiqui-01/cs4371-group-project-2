# CS4371 Group Project 2 — SafeStay Seattle

An LLM-assisted travel planner using Seattle open data.

## Team Assignments
- Person 1: Spark Data Pipeline and Neighborhood Analysis
- Person 2: Hotel Search and Attractions
- Person 3: LLM Evaluation and Best Months
- Person 4: Streamlit Website and Testing


## Tasks 1 and 2 (Seattle incidents, Spark)
Needs Java 17 and Python 3.9+.
python3 -m pip install requests pandas pyarrow pyspark tabulate

Environment variables (see .env.example): SOCRATA_APP_TOKEN, GROQ_API_KEY

python3 ingest.py            # downloads 2022-2024 reports to cache/incidents
python3 to_raw.py            # cache -> raw.parquet
python3 clean_summarize.py   # cleaning, categories, neighbourhood_month.parquet, t1_summary.txt
python3 task2.py july        # outlook CSV and summary for any month name or number

## Task 5 web app and Task 7 evaluation (Person 4)
Needs Python 3.9+ and the packages in requirements.txt:
python3 -m pip install -r requirements.txt

Put GROQ_API_KEY in a file called .env (copy .env.example). The app reads it on the server side; it is never shown on screen.

streamlit run app.py          # opens http://localhost:8501 (Chat and Map pages)
python3 -m task5.test_offline # checks the assistant logic with a fake LLM (no key needed)
python3 -m task5.transcripts  # records conversations into t5_transcripts.md
python3 -m task7.evaluate     # runs the 23 hand-labelled tests -> t7_evaluation.csv, task7/t7_results.md

Teammates' functions are plugged in through task5/backend.py. Until they exist, the app uses
placeholder versions from task5/stubs.py and shows a warning. task5/INTERFACES.md lists the
function names and columns each person's code must return.
