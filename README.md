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
