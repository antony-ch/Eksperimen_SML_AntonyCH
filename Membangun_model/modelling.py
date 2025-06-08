# Membangun_model/modelling.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os
import logging

import dagshub # Pastikan dagshub diimport
dagshub.init(repo_owner='antony-ch',
             repo_name='Eksperimen_SML_AntonyCH',
             mlflow=True)

# BLOK BERMASALAH SEBELUMNYA (import mlflow dan with mlflow.start_run(): ...) TELAH DIHAPUS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Konfigurasi MLflow untuk DagsHub ---
# Baris ini masih dipertahankan untuk memastikan MLflow_TRACKING_URI diatur
# meskipun dagshub.init() juga melakukan hal serupa. Ini tidak akan menyebabkan konflik.
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/antony-ch/Eksperimen_SML_AntonyCH.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'antony-ch' # Ganti dengan username DagsHub Anda
# UNTUK KEAMANAN: Jangan hardcode password/PAT di sini.
# Atur sebagai variabel lingkungan sebelum menjalankan script, misal:
# export MLFLOW_TRACKING_PASSWORD='<your_dagshub_pat>' (Linux/macOS)
# $env:MLFLOW_TRACKING_PASSWORD='<your_dagshub_pat>' (PowerShell Windows)
# Atau, yang paling disarankan untuk development lokal: `dagshub login` di terminal sebelum menjalankan script.

logging.info("MLflow tracking URI diatur ke DagsHub (untuk Basic model).")

def load_data(features_path, target_path):
    """Memuat dataset yang sudah diproses."""
    logging.info(f"Memuat fitur dari: {features_path}")
    X = pd.read_csv(features_path)
    logging.info(f"Memuat target dari: {target_path}")
    y = pd.read_csv(target_path).squeeze()
    return X, y

def train_basic_models(X_train, X_test, y_train, y_test):
    """Melatih beberapa model dasar dengan MLflow autologging."""
    models = {
        "Logistic Regression (Basic)": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree (Basic)": DecisionTreeClassifier(random_state=42),
        "Random Forest (Basic)": RandomForestClassifier(random_state=42)
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            logging.info(f"Memulai run MLflow dengan autologging untuk: {name}")
            mlflow.autolog() # Mengaktifkan autologging MLflow

            model.fit(X_train, y_train)

            logging.info(f"Model {name} berhasil dilatih.")
            logging.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
            logging.info(f"Autologging telah mencatat parameter dan metrik.")
        logging.info(f"Run MLflow untuk {name} selesai.")


if __name__ == "__main__":
    logging.info("Memulai script modelling.py (Basic Model Training)...")

    # Path ke dataset yang sudah dipreproses (relatif terhadap folder Membangun_model)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, 'namadataset_preprocessing')
    features_file = os.path.join(data_folder, 'telco_customer_churn_features_preprocessed.csv')
    target_file = os.path.join(data_folder, 'telco_customer_churn_target_preprocessed.csv')

    # Load data
    X, y = load_data(features_file, target_file)
    logging.info(f"Dimensi data loaded: X={X.shape}, y={y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info("Data berhasil dibagi menjadi training dan testing set.")

    # Train basic models
    train_basic_models(X_train, X_test, y_train, y_test)

    logging.info("Proses pelatihan model dasar dan logging MLflow (autolog) selesai.")