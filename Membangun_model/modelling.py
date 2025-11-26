import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg') # Penting agar tidak error GUI
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import joblib
import os
import sys
import shutil

# --- CONFIG ---
DAGSHUB_URI = "https://dagshub.com/Muhammad-Rizki-Putra/CustomerChurn.mlflow"
mlflow.set_tracking_uri(DAGSHUB_URI)
mlflow.set_experiment("Telco-Churn-Production")

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'telco_churn_clean.csv')
    
    if not os.path.exists(file_path):
        file_path = os.path.join(current_dir, '..', 'preprocessing', 'telco_churn_clean.csv')
        
    if not os.path.exists(file_path):
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
    
    with mlflow.start_run(run_name="Production_With_Matrix"):
        rf.fit(X_train, y_train)
        
        # Hitung Prediksi
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        
        # --- BAGIAN MEMBUAT CONFUSION MATRIX (MANUAL) ---
        print("Generating Confusion Matrix...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Simpan gambar lokal lalu upload
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")
        
        # --- BAGIAN MEMBUAT FOLDER MODEL LENGKAP ---
        print("Generating Model Folder...")
        if os.path.exists("model_temp_folder"):
            shutil.rmtree("model_temp_folder")
            
        mlflow.sklearn.save_model(rf, "model_temp_folder")
        mlflow.log_artifacts("model_temp_folder", artifact_path="model")
        
        # Simpan file lokal untuk Docker
        joblib.dump(rf, "model_churn_rf.pkl")
        
        print("Selesai. Matrix dan Model Folder berhasil di-upload.")
        
        # Cleanup
        if os.path.exists("model_temp_folder"): shutil.rmtree("model_temp_folder")
        if os.path.exists("confusion_matrix.png"): os.remove("confusion_matrix.png")

if __name__ == "__main__":
    train_production()