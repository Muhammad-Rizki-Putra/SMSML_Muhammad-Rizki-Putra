import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import shutil
import os
import sys
import joblib

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
    
    with mlflow.start_run(run_name="Production_Model_Structure"):
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        print(f"Training Selesai. Accuracy: {acc}")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("training_confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("training_confusion_matrix.png")

        print("Generating Standard MLflow Model Structure...")
        if os.path.exists("model_temp"):
            shutil.rmtree("model_temp")
            
        mlflow.sklearn.save_model(rf, "model_temp")
        
        print("Uploading artifacts to DagsHub...")
        mlflow.log_artifacts("model_temp", artifact_path="model")
        
        joblib.dump(rf, "model_churn_rf.pkl")
        
        print("Success! Struktur folder 'model' lengkap sudah terupload.")
        
        if os.path.exists("model_temp"): shutil.rmtree("model_temp")
        if os.path.exists("training_confusion_matrix.png"): os.remove("training_confusion_matrix.png")

if __name__ == "__main__":
    train_production()