import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from datetime import datetime

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
        
        # Model and scaler will be set during training
        self.model = None
        self.scaler = None
    
    def fetch_current_data(self):
        """Fetch current data from APIs"""
        current_data = {}
        
        for param in self.parameters:
            try:
                response = requests.get(
                    f"{self.base_url}{param}", 
                    headers=self.headers
                )
                response.raise_for_status()
                current_data[param] = float(response.text)
            except Exception as e:
                print(f"Error fetching {param}: {e}")
                return None
        
        # Add timestamp
        current_data['observationTime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return current_data
    
    def check_parameter_condition(self, value, param_name):
        """Check if a parameter is within its threshold"""
        param_threshold = self.thresholds[self.thresholds['Parameter'] == param_name]
        
        if param_threshold.empty:
            return 'Unknown'
        
        min_val = param_threshold['Min'].values[0]
        max_val = param_threshold['Max'].values[0]
        
        return 'Safe' if min_val <= value <= max_val else 'Unsafe'
    
    def prepare_training_data(self):
        """Prepare data for machine learning"""
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
        X = self.previous_data[self.parameters]
        y = (self.previous_data['condition'] == 'Unsafe').astype(int)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest Classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        print("Model Performance:")
        print(classification_report(y_test, y_pred))
        
        return self
    
    def predict_safety(self, current_data):
        """Predict safety of current water quality"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained first. Call prepare_training_data() first.")
        
        # Prepare current data for prediction
        current_df = pd.DataFrame([current_data])
        X_current = current_df[self.parameters]
        X_current_scaled = self.scaler.transform(X_current)
        
        # Predict
        prediction = self.model.predict(X_current_scaled)
        prediction_proba = self.model.predict_proba(X_current_scaled)
        
        # Interpret results
        condition = 'Unsafe' if prediction[0] == 1 else 'Safe'
        confidence = prediction_proba[0][prediction[0]]
        
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
            'confidence': confidence,
            'parameter_conditions': dict(zip(self.parameters, manual_conditions))
        }
    
    def save_model(self, model_filename='water_quality_model.joblib', scaler_filename='water_quality_scaler.joblib'):
        """Save trained model and scaler"""
        if self.model is not None:
            joblib.dump(self.model, model_filename)
            joblib.dump(self.scaler, scaler_filename)
            print(f"Model saved to {model_filename}")
            print(f"Scaler saved to {scaler_filename}")
        else:
            print("No model to save. Train the model first.")
    
    def run_prediction_workflow(self):
        """Complete workflow: train model, fetch current data, predict"""
        # Train the model
        self.prepare_training_data()
        
        # Fetch current data
        current_data = self.fetch_current_data()
        
        if current_data is None:
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
    print(f"Prediction Confidence: {result['confidence']:.2%}")
    print("\nParameter-wise Conditions:")
    for param, condition in result['parameter_conditions'].items():
        print(f"{param.capitalize()}: {condition}")