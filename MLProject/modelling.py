import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import argparse

# 1. Aktifkan MLflow Autologging
mlflow.sklearn.autolog()

def train_and_log_model(data_path, n_estimators, random_state):
    # Set nama eksperimen di MLflow jika tidak dijalankan dari MLflow Project (mlflow run)
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment("Heart_Disease_Classification")

    
    # 2. Memuat dataset hasil preprocessing
    # Cek jika path tidak ditemukan, coba cari di fallback paths
    if not os.path.exists(data_path):
        fallback_paths = [
            "heart_preprocessing.csv",
            "preprocessing/heart_preprocessing.csv",
            "../preprocessing/heart_preprocessing.csv",
            "../../preprocessing/heart_preprocessing.csv"
        ]
        found = False
        for path in fallback_paths:
            if os.path.exists(path):
                data_path = path
                found = True
                break
        if not found:
            print(f"File data {data_path} tidak ditemukan. Pastikan script preprocessing sudah dijalankan!")
            return
        
    print(f"Memuat data dari: {data_path}")
    df = pd.read_csv(data_path)
    
    # Memisahkan kembali fitur dan target untuk training
    X = df.drop(columns=['target'])
    y = df['target']
    
    print("Memulai pelatihan model dengan MLflow Autolog...")
    
    # 3. Jalankan MLflow tracking
    with mlflow.start_run(run_name="Random_Forest_Baseline") as run:
        # Menggunakan model Random Forest Classifier
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X, y)
        
        # Prediksi sederhana untuk memastikan model berjalan baik
        predictions = model.predict(X)
        acc = accuracy_score(y, predictions)
        
        print(f"Pelatihan Selesai! Akurasi Training: {acc:.4f}")
        print(f"Run ID: {run.info.run_id}")
        print("Buka terminal lalu ketik 'mlflow ui' untuk melihat hasilnya di browser.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="preprocessing/heart_preprocessing.csv", help="Path ke file heart_preprocessing.csv")
    parser.add_argument("--n_estimators", type=int, default=100, help="Jumlah estimators untuk RandomForestClassifier")
    parser.add_argument("--random_state", type=int, default=42, help="Random state untuk RandomForestClassifier")
    args = parser.parse_args()
    
    train_and_log_model(args.data_path, args.n_estimators, args.random_state)