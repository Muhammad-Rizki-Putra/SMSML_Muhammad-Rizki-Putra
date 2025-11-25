import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_path, output_path):
    print("Loading data...")
    df = pd.read_csv(input_path)
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn']) 
    
    df = pd.get_dummies(df, drop_first=True)
    
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Preprocessing Done!")

if __name__ == "__main__":
    import sys
    input_csv = 'preprocessing/WA_Fn-UseC_-Telco-Customer-Churn.csv' # Sesuaikan nama file raw
    output_csv = 'preprocessing/telco_churn_clean.csv'

    if len(sys.argv) > 2:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]

    preprocess_data(input_csv, output_csv)

