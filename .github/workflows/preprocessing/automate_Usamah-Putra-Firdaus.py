import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_data(path):
    return pd.read_csv(path)

def explore_data(df):
    print("Jumlah nilai null tiap kolom:\n", df.isnull().sum())
    print("Data terduplikat:", df.duplicated().sum())

    plt.figure(figsize=(6,4))
    sns.countplot(x='gender', data=df)
    plt.title('Gender Distribution')
    plt.savefig("gender_distribution.png")
    plt.close()

    sns.histplot(df['age'], kde=True, color='#FF7601')
    plt.title('Age Distribution')
    plt.savefig("age_distribution.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(x='smoking_history', data=df)
    plt.title('Smoking History Distribution')
    plt.savefig("smoking_history_distribution.png")
    plt.close()

    plt.figure(figsize=(12,6))
    sns.lineplot(x=df['age'], y=df['diabetes'], color='#DC2525')
    plt.title('Age vs Diabetes')
    plt.savefig("age_vs_diabetes.png")
    plt.close()

def clean_data(df):
    df['age'] = df['age'].astype(int)
    df.drop_duplicates(inplace=True)
    df = df[df['gender'].isin(['Female', 'Male'])]

    df.loc[:, 'smoking_history'] = df['smoking_history'].replace({
        'current': 'ever',
        'former': 'ever',
        'not current': 'ever'
    })

    return df

def encode_transform(df):
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])

    df = pd.get_dummies(df, columns=['smoking_history'], drop_first=True)

    # Convert dummy columns to int
    dummy_cols = [col for col in df.columns if 'smoking_history_' in col]
    for col in dummy_cols:
        df[col] = df[col].astype(int)

    # Normalize numerical columns
    numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

def main():

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'diabetes_prediction_raw.csv')

    print("Current working directory:", os.getcwd())
    print("Target file path:", file_path)
    print("File exists?", os.path.exists(file_path))

    if not os.path.exists(file_path):
        print(f"File {file_path} tidak ditemukan.")
        return

    df = load_data(file_path)
    explore_data(df)
    df = clean_data(df)
    df = encode_transform(df)

    print("Data setelah diproses:\n", df.head())

    # Simpan hasil akhir
    output_dir = os.path.dirname(file_path)  # sama dengan direktori project
    output_path = os.path.join(output_dir, 'preprocessing', 'diabetes_prediction_processing.csv')

    # Pastikan folder 'preprocessing' ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Data berhasil disimpan ke {output_path}")



if __name__ == "__main__":
    main()
