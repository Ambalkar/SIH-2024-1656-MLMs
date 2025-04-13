import pandas as pd
import numpy as np
import requests
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json

class WaterQualityPredictor:
    def __init__(self, threshold_file='threshold.csv', previous_data_file='wqns_dataset.csv'):
        # API Configuration
        self.base_url = "https://gemini.incois.gov.in/OceanDataAPI/api/wqns/Kochi/"
        self.headers = {
            "Authorization": "446d183e64e64e8eb4bca1407ab02a89"
        }
        
        # Parameters to fetch
        self.parameters = [
            'ph', 'temperature', 'dissolvedoxygen', 
            'dissolvedmethane', 'cdom', 'salinity'
        ]
        
        # Load thresholds
        self.thresholds = pd.read_csv(threshold_file)
        
        # Load previous data
        self.previous_data = pd.read_csv(previous_data_file)
        
        # TensorFlow model and scaler
        self.model = None
        self.scaler = None

    def fetch_current_data(self):
        """Fetch current data from APIs or allow manual entry if API fails."""
        current_data = {}
        
        for param in self.parameters:
            try:
                response = requests.get(
                    f"{self.base_url}{param}", 
                    headers=self.headers
                )
                response.raise_for_status()
                data = json.loads(response.text)
                
                # Extract the latest value
                if "observationTime" in data and "parameter" in data:
                    latest_index = 0  # First entry in observationTime is the most recent
                    current_data[param] = float(data["parameter"][latest_index])
                else:
                    raise ValueError("Unexpected data format")
            except Exception as e:
                print(f"Error fetching {param}: {e}")
                # Manual entry fallback
                try:
                    current_data[param] = float(input(f"Enter current value for {param}: "))
                except ValueError:
                    print(f"Invalid input for {param}, skipping...")
                    current_data[param] = None

        # Add timestamp
        current_data['observationTime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return current_data

    def prepare_training_data(self):
        """Prepare data for machine learning."""
        # Add condition column to previous data if not exists
        if 'condition' not in self.previous_data.columns:
            self.previous_data['condition'] = self.previous_data.apply(
                lambda row: 'Unsafe' if any(
                    self.check_parameter_condition(row[param], param) == 'Unsafe' 
                    for param in self.parameters
                ) else 'Safe', 
                axis=1
            )
        
        # Prepare features and target
        X = self.previous_data[self.parameters].fillna(0)
        y = (self.previous_data['condition'] == 'Unsafe').astype(int)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build TensorFlow model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        self.model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X_test_scaled, y_test)
        print(f"Model Loss: {loss}, Accuracy: {accuracy}")
        
        return self
    
    def predict_safety(self, current_data):
        """Predict safety of current water quality."""
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained first. Call prepare_training_data() first.")
        
        # Prepare current data for prediction
        current_df = pd.DataFrame([current_data])
        X_current = current_df[self.parameters].fillna(0)
        X_current_scaled = self.scaler.transform(X_current)
        
        # Predict
        prediction = self.model.predict(X_current_scaled)
        condition = 'Unsafe' if prediction[0][0] > 0.5 else 'Safe'
        
        # Manual threshold check as a backup
        manual_conditions = [
            self.check_parameter_condition(current_data[param], param) 
            for param in self.parameters
        ]
        
        # If manual check finds any parameter unsafe, override model prediction
        if 'Unsafe' in manual_conditions:
            condition = 'Unsafe'
        
        return {
            'condition': condition,
            'parameter_conditions': dict(zip(self.parameters, manual_conditions))
        }
    
    def save_model(self, model_filename='water_quality_model.h5', scaler_filename='water_quality_scaler.pkl'):
        """Save trained model and scaler."""
        if self.model is not None:
            self.model.save(model_filename)
            #joblib.dump(self.scaler, scaler_filename)
            print(f"Model saved to {model_filename}")
            print(f"Scaler saved to {scaler_filename}")
        else:
            print("No model to save. Train the model first.")

    def check_parameter_condition(self, value, param_name):
        """Check if a parameter is within its threshold."""
        param_threshold = self.thresholds[self.thresholds['Parameter'] == param_name]
        if param_threshold.empty:
            return 'Unknown'
        min_val = param_threshold['Min'].values[0]
        max_val = param_threshold['Max'].values[0]
        return 'Safe' if min_val <= value <= max_val else 'Unsafe'
    
    def run_prediction_workflow(self):
        """Complete workflow: train model, fetch current data, predict."""
        # Train the model
        self.prepare_training_data()
        
        # Fetch current data
        current_data = self.fetch_current_data()
        
        if not current_data:
            print("Failed to fetch current data.")
            return None
        
        # Predict safety
        prediction_result = self.predict_safety(current_data)
        
        # Save the model
        self.save_model()
        
        return prediction_result

# Run the prediction
predictor = WaterQualityPredictor()
result = predictor.run_prediction_workflow()

if result:
    print("\nPrediction Result:")
    print(f"Overall Condition: {result['condition']}")
    print("\nParameter-wise Conditions:")
    for param, condition in result['parameter_conditions'].items():
        print(f"{param.capitalize()}: {condition}")
