import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

def clean_and_encode(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    
    # 1. Calculate Yield
    # Avoid division by zero
    df['Yield'] = df['Production'] / df['Area']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. Handle Missing Values
    print("Missing values before cleaning:")
    print(df.isnull().sum())
    
    # Fill Production missing values with median (or drop)
    df['Production'] = df['Production'].fillna(df['Production'].median())
    df['Yield'] = df['Yield'].fillna(df['Yield'].median())
    
    # 3. Remove Outliers (using IQR for Yield)
    Q1 = df['Yield'].quantile(0.25)
    Q3 = df['Yield'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['Yield'] >= lower_bound) & (df['Yield'] <= upper_bound)]
    print(f"Outliers removed. New shape: {df.shape}")

    # 4. Label Encoding
    le = LabelEncoder()
    categorical_cols = ['State_Name', 'Crop', 'Season', 'District_Name']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    # 5. Visualization (Yield Distribution)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Yield'], kde=True, color='green')
    plt.title('Yield Distribution After Cleaning')
    plt.xlabel('Yield (Production/Area)')
    plt.ylabel('Frequency')
    plt.savefig('ml/yield_distribution.png')
    print("Yield distribution chart saved to ml/yield_distribution.png")

    # Save cleaned data
    df.to_csv('data/cleaned_crop_yield.csv', index=False)
    print("Cleaned data saved to data/cleaned_crop_yield.csv")
    
    return df

if __name__ == "__main__":
    clean_and_encode("data/crop_yield.csv")
