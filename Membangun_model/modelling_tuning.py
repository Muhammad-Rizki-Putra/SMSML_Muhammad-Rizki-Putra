import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import time
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os
import sys
import joblib 

DAGSHUB_URI = "https://dagshub.com/Muhammad-Rizki-Putra/CustomerChurn.mlflow"
mlflow.set_tracking_uri(DAGSHUB_URI)
mlflow.set_experiment("Telco-Churn-Advanced-Experiment")

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'preprocessing', 'telco_churn_clean.csv')
    
    print(f"Mencari data di: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {file_path}")
        
    return pd.read_csv(file_path)

def train_and_log():
    df = load_data()
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20], 
        'min_samples_split': [2, 5]
    }
    
    print("Starting MLflow Run...")
    
    with mlflow.start_run(run_name="RandomForest_GridSearch_Advance"):
        
        start_time = time.time()
        
        print("Performing Grid Search (Sequential)...")
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=1)
        grid_search.fit(X_train, y_train)
        
        end_time = time.time()
        duration = end_time - start_time
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        print("Logging metrics to DagsHub...")
        
        for param, value in best_params.items():
            mlflow.log_param(param, value)
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc_score", roc_auc)      
        mlflow.log_metric("training_duration", duration) 
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("confusion_matrix.png")
        plt.close()
        
        mlflow.log_artifact("confusion_matrix.png")
        
        print("Saving model to artifact...")
        model_filename = "model_churn_rf.pkl"
        joblib.dump(best_model, model_filename)
        
        mlflow.log_artifact(model_filename)
        
        if os.path.exists("confusion_matrix.png"): os.remove("confusion_matrix.png")
        if os.path.exists(model_filename): os.remove(model_filename)
        
        print(f"Run Completed! Accuracy: {acc:.4f}")
        print(f"Check your DagsHub Artifacts: {DAGSHUB_URI}")

if __name__ == "__main__":
    train_and_log()