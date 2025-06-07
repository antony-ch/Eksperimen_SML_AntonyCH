# automate_Preprocessing_TelcoChurn.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import zipfile # Untuk menangani file ZIP jika dataset berasal dari sumber yang sama

def preprocess_telco_churn_data(raw_data_path, output_dir='namadataset_preprocessing'):
    """
    Melakukan preprocessing pada dataset Telco Customer Churn.

    Args:
        raw_data_path (str): Path ke file CSV dataset mentah.
                              Jika berupa file ZIP dari Kaggle, pastikan sudah diekstrak.
        output_dir (str): Direktori tempat menyimpan data yang sudah dipreprocessing.

    Returns:
        tuple: (DataFrame fitur yang sudah diproses, Series target yang sudah diproses)
    """
    print(f"Memulai preprocessing untuk data dari: {raw_data_path}")

    # Memuat Dataset
    try:
        # Jika raw_data_path adalah file ZIP, ekstrak dulu
        if raw_data_path.endswith('.zip'):
            print(f"Mengekstrak file ZIP: {raw_data_path}")
            with zipfile.ZipFile(raw_data_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(raw_data_path)) # Ekstrak ke direktori yang sama
            # Asumsi nama file CSV di dalam ZIP
            csv_filename = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
            df = pd.read_csv(os.path.join(os.path.dirname(raw_data_path), csv_filename))
        else:
            df = pd.read_csv(raw_data_path)
        print("Dataset berhasil dimuat.")
    except Exception as e:
        print(f"Error memuat dataset: {e}")
        raise

    # 1. Menghapus kolom 'customerID'
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        print("Kolom 'customerID' telah dihapus.")

    # 2. Mengatasi missing values di 'TotalCharges'
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    print("Missing values di 'TotalCharges' telah diisi dengan median setelah konversi ke numerik.")

    # 3. Mengubah 'No internet service' dan 'No phone service' menjadi 'No' untuk konsistensi
    cols_to_consolidate_internet = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in cols_to_consolidate_internet:
        if col in df.columns:
            df[col] = df[col].replace('No internet service', 'No')

    if 'MultipleLines' in df.columns:
        df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
    print("Nilai 'No internet service' dan 'No phone service' telah dikonsolidasi menjadi 'No'.")

    # 4. Menangani data duplikat (seharusnya sudah dicek di EDA dan hasilnya tidak ada)
    initial_rows_preprocessed = df.shape[0]
    df.drop_duplicates(inplace=True)
    rows_after_duplicates = df.shape[0]
    if initial_rows_preprocessed > rows_after_duplicates:
        print(f"Duplikat data dihapus. Jumlah baris: {initial_rows_preprocessed} -> {rows_after_duplicates}")
    else:
        print("Tidak ada duplikat data yang ditemukan.")

    # 5. Encoding variabel target 'Churn' (Yes/No) menjadi 1/0
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        print("Variabel target 'Churn' telah di-encode (Yes=1, No=0).")
    else:
        print("Kolom 'Churn' tidak ditemukan, preprocessing target diabaikan.")

    # Pisahkan fitur (X) dan target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    print("Data fitur (X) dan target (y) telah dipisahkan.")

    # Identifikasi kolom kategorikal dan numerik
    categorical_cols = X.select_dtypes(include='object').columns
    numerical_cols = X.select_dtypes(include=np.number).columns

    print(f"Kolom kategorikal yang akan di-encode: {categorical_cols.tolist()}")
    print(f"Kolom numerik yang akan di-scaling: {numerical_cols.tolist()}")

    # 6. Preprocessing menggunakan ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough' # Biarkan kolom lain yang tidak masuk kategori ini
    )

    # Terapkan preprocessor
    X_preprocessed_array = preprocessor.fit_transform(X)

    # Mendapatkan nama-nama fitur baru
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_feature_names = list(numerical_cols) + list(ohe_feature_names)

    X_processed_df = pd.DataFrame(X_preprocessed_array, columns=all_feature_names)

    print("Data setelah preprocessing dengan ColumnTransformer:")
    print(X_processed_df.head())
    print(f"Ukuran data setelah preprocessing: {X_processed_df.shape}")

    # Simpan data yang sudah dipreprocessing
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_file_path_X = os.path.join(output_dir, 'telco_customer_churn_features_preprocessed.csv')
    preprocessed_file_path_y = os.path.join(output_dir, 'telco_customer_churn_target_preprocessed.csv')

    X_processed_df.to_csv(preprocessed_file_path_X, index=False)
    y.to_csv(preprocessed_file_path_y, index=False)

    print(f"\nData fitur (X) yang sudah dipreprocessing disimpan di: {preprocessed_file_path_X}")
    print(f"Data target (y) yang sudah dipreprocessing disimpan di: {preprocessed_file_path_y}")
    print("--- Preprocessing Selesai ---")

    return X_processed_df, y

if __name__ == "__main__":
    # Contoh penggunaan jika dijalankan sebagai skrip mandiri
    # Asumsi file ZIP dataset mentah berada di root repository atau di folder 'namadataset_raw'
    # Untuk GitHub Actions, kita akan menempatkan file ZIP ini di sana.
    raw_data_zip_path = 'namadataset_raw/telco-customer-churn.zip'
    output_processed_dir = 'namadataset_preprocessing'

    # Buat direktori 'namadataset_raw' dan letakkan 'telco-customer-churn.zip' di dalamnya
    # Anda perlu mengunduh telco-customer-churn.zip dari Kaggle dan meletakkannya di sini
    # atau jika ingin lebih otomatis, download_kaggle_dataset bisa dipanggil di sini (tapi lebih kompleks untuk CI)
    # Untuk kemudahan GitHub Actions, kita akan langsung commit file ZIP nya.

    if not os.path.exists(raw_data_zip_path):
        print(f"Peringatan: File {raw_data_zip_path} tidak ditemukan. Silakan letakkan file ZIP dataset mentah di lokasi ini.")
        print("Mengunduh dataset dari Kaggle secara langsung untuk demo. Di GitHub Actions, Anda harus meng-*commit* file ZIP ini.")
        # Jika Anda ingin mendownloadnya secara otomatis untuk pengujian lokal, bisa tambahkan kode ini:
        # Perlu instal 'kaggle' dan konfigurasi API key
        # import kaggle
        # kaggle_dataset_name = "blastchar/telco-customer-churn"
        # !kaggle datasets download -d {kaggle_dataset_name} -p namadataset_raw/
        # print("Dataset telah diunduh ke namadataset_raw/")


    # Panggil fungsi preprocessing
    try:
        X_processed, y_processed = preprocess_telco_churn_data(raw_data_zip_path, output_processed_dir)
        print("\nPreprocessing berhasil dan data disimpan.")
    except Exception as e:
        print(f"\nPreprocessing gagal: {e}")