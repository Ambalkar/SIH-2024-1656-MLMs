# Updated code for `app.py`
from flask import Flask, request, jsonify, render_template
import json
import os
import joblib

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'water_quality_scaler.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        date = data.get('date')
        time = data.get('time')
        if not date or not time:
            return jsonify({'error': 'Date and time are required!'}), 400

        prediction_input = [date, time]  # Adjust this based on model's input structure
        prediction, probabilities = model.predict(prediction_input)

        return jsonify({
            'prediction': prediction,
            'probabilities': probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

