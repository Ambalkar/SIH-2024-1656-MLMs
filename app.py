from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the trained model
model_path = 'weather_condition_predictor.h5'
try:
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the label encoder
label_encoder_path = 'label_encoder.pkl'
try:
    with open(label_encoder_path, 'rb') as le_file:
        label_encoder = pickle.load(le_file)
    print(f"Label encoder loaded successfully from {label_encoder_path}")
except Exception as e:
    print(f"Error loading label encoder: {e}")
    label_encoder = None

# Define the expected feature columns in the correct order
feature_columns = ['temperature_2m', 'relative_humidity_2m', 'precipitation']

# Define safety level classification based on weather conditions
def classify_safety(condition):
    safety_mapping = {
        'Clear': 'Safe',
        'Partly Cloudy': 'Safe',
        'Cloudy': 'Moderately Safe',
        'Fog': 'Caution',
        'Drizzle': 'Moderately Safe',
        'Rain': 'Caution',
        'Heavy Rain': 'Unsafe',
        'Thunderstorm': 'Unsafe',
        'Snow': 'Unsafe',
        'Sleet': 'Unsafe'
        # Add more mappings as needed
    }
    return safety_mapping.get(condition, 'Unknown')

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    if model is not None and label_encoder is not None:
        return jsonify({"status": "healthy", "message": "API is running and models are loaded"})
    else:
        return jsonify({"status": "degraded", "message": "API is running but models failed to load"}), 503

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_encoder is None:
        return jsonify({"error": "Model or label encoder not loaded properly"}), 500
    
    try:
        # Get input data from the request
        input_data = request.json
        
        # Check if all required features are present
        for feature in feature_columns:
            if feature not in input_data:
                return jsonify({"error": f"Missing required feature: {feature}"}), 400
        
        # Prepare input for the model in the correct order
        features = [input_data[col] for col in feature_columns]
        
        # Reshape the input for the model
        input_array = np.array([features])
        
        # Make prediction
        prediction_probabilities = model.predict(input_array)
        predicted_class_index = np.argmax(prediction_probabilities[0])
        
        # Get the weather condition label
        predicted_condition = label_encoder.inverse_transform([predicted_class_index])[0]
        
        # Get the safety level based on the condition
        safety_level = classify_safety(predicted_condition)
        
        # Get confidence level (probability of the predicted class)
        confidence = float(prediction_probabilities[0][predicted_class_index])
        
        # Prepare response
        response = {
            "weather_condition": predicted_condition,
            "safety_level": safety_level,
            "confidence": confidence,
            "input_data": input_data
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during prediction: {error_details}")
        return jsonify({"error": str(e), "details": error_details}), 500

@app.route('/fetch_and_predict', methods=['GET'])
def fetch_and_predict():
    """Fetch current weather data and make a prediction"""
    try:
        # Get location parameters (default to Kochi, India if not provided)
        latitude = request.args.get('latitude', '9.9399')
        longitude = request.args.get('longitude', '76.2602')
        
        # Fetch weather data from Open-Meteo API
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": True,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation",
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return jsonify({"error": f"Failed to fetch weather data: {response.text}"}), 500
        
        weather_data = response.json()
        
        # Extract required features
        current_data = weather_data.get('current', {})
        
        # Create input data dictionary
        input_data = {
            'temperature_2m': current_data.get('temperature_2m'),
            'relative_humidity_2m': current_data.get('relative_humidity_2m'),
            'precipitation': current_data.get('precipitation', 0.0)
        }
        
        # Check for missing values and get from hourly if needed
        if None in input_data.values() and 'hourly' in weather_data:
            hourly = weather_data['hourly']
            if input_data['temperature_2m'] is None and 'temperature_2m' in hourly:
                input_data['temperature_2m'] = hourly['temperature_2m'][0]
            if input_data['relative_humidity_2m'] is None and 'relative_humidity_2m' in hourly:
                input_data['relative_humidity_2m'] = hourly['relative_humidity_2m'][0]
            if input_data['precipitation'] is None and 'precipitation' in hourly:
                input_data['precipitation'] = hourly['precipitation'][0]
        
        # Check if we still have missing values
        for key, value in input_data.items():
            if value is None:
                return jsonify({"error": f"Missing weather data for {key}"}), 500
        
        # Forward to the predict endpoint
        return predict()
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error fetching and predicting: {error_details}")
        return jsonify({"error": str(e), "details": error_details}), 500

if __name__ == '__main__':
    # Import requests here to avoid issues if the module is not installed
    # when only running the prediction endpoint
    import requests
    app.run(host='0.0.0.0', port=5000, debug=False)