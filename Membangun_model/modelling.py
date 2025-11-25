import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib
import os
import sys

DAGSHUB_URI = "https://dagshub.com/Muhammad-Rizki-Putra/CustomerChurn.mlflow"
mlflow.set_tracking_uri(DAGSHUB_URI)
mlflow.set_experiment("Telco-Churn-Production")

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'telco_churn_clean.csv')
    
    if not os.path.exists(file_path):
        file_path = os.path.join(current_dir, '..', 'preprocessing', 'telco_churn_clean.csv')
        
    if not os.path.exists(file_path):
        print("Dataset tidak ditemukan.")
        sys.exit(1)
        
    return pd.read_csv(file_path)

def train_production():
    print("Loading data...")
    df = load_data()

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="Production_Run_Final"):
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc) 
        
        model_filename = "model_churn_rf.pkl"
        joblib.dump(rf, model_filename)
        mlflow.log_artifact(model_filename)
        
        print(f"Training Selesai. Accuracy: {acc}")
        print("Model & Metrics berhasil di-upload manual sebagai backup.")

if __name__ == "__main__":
    train_production()