import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_model(data_path):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # Features and Target
    # Based on our cleaned data columns: State_Name, District_Name, Crop_Year, Season, Crop, Temperature, Humidity, Soil_Moisture, Area, Yield
    # Note: We predict 'Yield' (Production/Area)
    
    X = df[['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop', 'Temperature', 'Humidity', 'Soil_Moisture', 'Area']]
    y = df['Yield']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(model, 'models/crop_yield_model.pkl')
    print("Model saved to models/crop_yield_model.pkl")

if __name__ == "__main__":
    train_model("data/cleaned_crop_yield.csv")
