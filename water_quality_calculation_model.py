import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json

class WaterQualityCalculator:
    def __init__(self, threshold_file='threshold.csv', data_file='updated_wqns.csv'):
        # Define parameter order - this will be used consistently throughout the class
        self.parameters = [
            'ph', 'temperature', 'dissolvedoxygen', 
            'dissolvedmethane', 'cdom', 'salinity'
        ]
        
        # Load thresholds
        self.thresholds = pd.read_csv(threshold_file)
        
        # Load data
        self.data = pd.read_csv(data_file)
        
        # TensorFlow model and scaler
        self.model = None
        self.scaler = None
        
        # Train the model on initialization
        self.prepare_training_data()

    def prepare_training_data(self):
        """Prepare data for machine learning."""
        # Add condition column to data if not exists
        if 'condition' not in self.data.columns:
            self.data['condition'] = self.data.apply(
                lambda row: 'Unsafe' if any(
                    self.check_parameter_condition(row[param], param) == 'Unsafe' 
                    for param in self.parameters
                ) else 'Safe', 
                axis=1
            )
        
        # Prepare features and target - ensure consistent parameter order
        X = self.data[self.parameters].fillna(0)  # Use self.parameters to maintain order
        y = (self.data['condition'] == 'Unsafe').astype(int)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build TensorFlow model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(len(self.parameters),)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        self.model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=0)
        
        return self

    def calculate_water_quality(self, params):
        """Calculate water quality based on input parameters."""
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained first.")
        
        # Ensure parameters are in the correct order
        ordered_params = {param: float(params[param]) for param in self.parameters}
        
        # Create DataFrame with consistent parameter order
        input_data = pd.DataFrame([ordered_params])
        
        # Verify all required parameters are present
        missing_params = set(self.parameters) - set(params.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        # Scale the input data
        input_scaled = self.scaler.transform(input_data)
        
        # Get model prediction
        prediction = self.model.predict(input_scaled, verbose=0)
        condition = 'Unsafe' if prediction[0][0] > 0.5 else 'Safe'
        
        # Check each parameter against thresholds
        parameter_conditions = {
            param: self.check_parameter_condition(params[param], param)
            for param in self.parameters
        }
        
        # Calculate overall score (0-100)
        safe_params = sum(1 for condition in parameter_conditions.values() if condition == 'Safe')
        overall_score = (safe_params / len(self.parameters)) * 100
        
        # Prepare detailed results
        results = {
            'overall_condition': condition,
            'overall_score': overall_score,
            'prediction_confidence': float(prediction[0][0]),
            'parameter_conditions': parameter_conditions,
            'parameter_values': params,
            'thresholds': {
                param: {
                    'min': float(self.thresholds[self.thresholds['Parameter'] == param]['Min'].values[0]),
                    'max': float(self.thresholds[self.thresholds['Parameter'] == param]['Max'].values[0])
                }
                for param in self.parameters
            }
        }
        
        return results

    def check_parameter_condition(self, value, param_name):
        """Check if a parameter is within its threshold."""
        param_threshold = self.thresholds[self.thresholds['Parameter'] == param_name]
        if param_threshold.empty:
            return 'Unknown'
        min_val = param_threshold['Min'].values[0]
        max_val = param_threshold['Max'].values[0]
        return 'Safe' if min_val <= value <= max_val else 'Unsafe'
