
import json
import requests
from datetime import date, timedelta

# First hotel from your collected dataset
HOTEL_KEY = "g60878-d258824"

# Use future booking dates
check_in = date.today() + timedelta(days=30)
check_out = check_in + timedelta(days=3)

print("Hotel:", HOTEL_KEY)
print("Check-in:", check_in)
print("Check-out:", check_out)

response = requests.get(
    "https://data.xotelo.com/api/rates",
    params={
        "hotel_key": HOTEL_KEY,
        "chk_in": check_in.isoformat(),
        "chk_out": check_out.isoformat(),
    },
    timeout=30,
)

print("HTTP status:", response.status_code)
response.raise_for_status()

data = response.json()

print(json.dumps(data, indent=2)[:5000])
