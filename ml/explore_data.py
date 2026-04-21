import pandas as pd
import os

def explore_csv(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    print("--- Dataset Information ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\n--- Columns ---")
    print(df.columns.tolist())
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Summary Statistics ---")
    print(df.describe())

if __name__ == "__main__":
    explore_csv("data/crop_yield.csv")
