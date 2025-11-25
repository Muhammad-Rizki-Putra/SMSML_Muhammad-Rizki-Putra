import requests
import random
import time

url = "http://localhost:5000/predict"

print(f"Starting inference test to {url}...")

while True:
    # Data dummy
    payload = [
        {
            "tenure": random.randint(1, 72),
            "MonthlyCharges": random.uniform(20.0, 118.0),
            "TotalCharges": random.uniform(20.0, 8000.0)
        }
    ]

    try:

        response = requests.post(url, json=payload)

   
        if response.status_code == 200:
            print(f"Success! Prediction: {response.json()}")
        else:
            print(f"Failed: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Error: {e}")

    time.sleep(1)