import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def train():
    if os.path.exists("telco_churn_clean.csv"):
        df = pd.read_csv("telco_churn_clean.csv")
    else:
        print("Dataset tidak ditemukan!")
        return

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)
    
    acc = accuracy_score(y, rf.predict(X))
    print(f"Model Trained. Accuracy: {acc}")
    
    joblib.dump(rf, "model_churn.pkl")
    print("Model saved to model_churn.pkl")

if __name__ == "__main__":
    train()