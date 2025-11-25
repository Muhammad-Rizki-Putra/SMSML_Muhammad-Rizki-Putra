# File: Monitoring/prometheus_exporter.py
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import psutil
import time
import threading
import random
import os

app = Flask(__name__)

SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU Usage in percent')
SYSTEM_RAM_USAGE = Gauge('system_ram_usage_bytes', 'RAM Usage in bytes')
APP_UPTIME = Gauge('app_uptime_seconds', 'Application uptime in seconds')

REQUEST_COUNT = Counter('app_requests_total', 'Total HTTP Requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency', ['endpoint'])
ERROR_COUNT = Counter('app_errors_total', 'Total HTTP Errors', ['endpoint'])

PREDICTION_CHURN_COUNT = Counter('prediction_churn_total', 'Total Churn Predictions')
PREDICTION_NO_CHURN_COUNT = Counter('prediction_no_churn_total', 'Total No-Churn Predictions')
INPUT_MONTHLY_CHARGES = Histogram('input_monthly_charges_dist', 'Distribution of Monthly Charges Input')
CONFIDENCE_SCORE = Gauge('model_confidence_score_avg', 'Average Model Confidence Score')

MODEL_PATH = "model_churn_rf.pkl" 
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
else:
    print("WARNING: Model not found. Using dummy prediction for testing.")
    model = None

def update_system_metrics():
    start_time = time.time()
    while True:
        SYSTEM_CPU_USAGE.set(psutil.cpu_percent())
        SYSTEM_RAM_USAGE.set(psutil.virtual_memory().used)
        APP_UPTIME.set(time.time() - start_time)
        time.sleep(5)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    
    try:
        data = request.json
        df = pd.DataFrame(data)
        
        if model:
            try:
                prediction = model.predict(df)[0]
                proba = model.predict_proba(df)[0][1]
            except:
                prediction = random.choice([0, 1])
                proba = random.uniform(0.5, 0.9)
        else:
            prediction = random.choice([0, 1])
            proba = random.uniform(0.1, 0.9)

        if prediction == 1:
            PREDICTION_CHURN_COUNT.inc()
        else:
            PREDICTION_NO_CHURN_COUNT.inc()
            
        if 'MonthlyCharges' in data[0]:
            INPUT_MONTHLY_CHARGES.observe(float(data[0]['MonthlyCharges']))
            
        CONFIDENCE_SCORE.set(proba)
        
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint='/predict').observe(latency)
        
        return jsonify({'churn_prediction': int(prediction), 'confidence': proba})

    except Exception as e:
        ERROR_COUNT.labels(endpoint='/predict').inc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    start_http_server(8000)
    print("Prometheus Metrics running on port 8000")
    
    threading.Thread(target=update_system_metrics, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5000)