from flask import Flask, render_template, request, jsonify
from water_quality_prediction_model import WaterQualityPredictor
from water_quality_calculation_model import WaterQualityCalculator
import pandas as pd
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# Initialize models
try:
    prediction_model = WaterQualityPredictor('updated_wqns.csv')
    calculation_model = WaterQualityCalculator('threshold.csv', 'updated_wqns.csv')
    model_results = prediction_model.train_model()  # Train the prediction model on startup
except Exception as e:
    print(f"Error initializing models: {e}")
    raise

@app.route('/')
def home():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle water quality prediction requests"""
    try:
        # Get the date from the request
        data = request.get_json()
        if not data or 'date' not in data:
            return jsonify({'success': False, 'error': 'No date provided'})

        # Parse the input date
        try:
            input_date = datetime.fromisoformat(data['date'].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid date format'})

        # Make prediction
        prediction = prediction_model.predict_water_quality(input_date, model_results)

        # Prepare response with standard deviations
        response = {
            'success': True,
            'prediction': {
                'condition': prediction['condition'],
                'probabilities': prediction['probabilities']
            },
            'model_metrics': {
                'accuracy': model_results['accuracy'],
                'accuracy_std': model_results['accuracy_std'],
                'f1_score': model_results['f1_score'],
                'f1_score_std': model_results['f1_score_std'],
                'precision': model_results['precision'],
                'precision_std': model_results['precision_std'],
                'recall': model_results['recall'],
                'recall_std': model_results['recall_std']
            },
            'feature_importance': dict(zip(
                ['salinity', 'ph', 'dissolvedoxygen', 'dissolvedmethane', 'cdom', 'temperature', 
                 'hour', 'month', 'day_of_week', 'day_of_year'],
                model_results['model'].feature_importances_
            ))
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/calculate', methods=['POST'])
def calculate():
    """Handle water quality calculation requests"""
    try:
        # Get parameters from the request
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})

        # Define the expected parameter order
        expected_params = [
            'ph', 'temperature', 'dissolvedoxygen', 
            'dissolvedmethane', 'cdom', 'salinity'
        ]

        # Map frontend parameter names to backend names
        param_mapping = {
            'ph': 'ph',
            'temperature': 'temperature',
            'dissolved_oxygen': 'dissolvedoxygen',
            'dissolved_methane': 'dissolvedmethane',
            'cdom': 'cdom',
            'salinity': 'salinity'
        }

        # Extract and validate parameters
        try:
            # Convert parameters using the mapping
            params = {}
            for frontend_param, backend_param in param_mapping.items():
                if frontend_param in data:
                    params[backend_param] = float(data[frontend_param])
                else:
                    return jsonify({
                        'success': False, 
                        'error': f'Missing required parameter: {frontend_param}'
                    })

            # Validate parameter ranges
            for param, value in params.items():
                if value < 0:  # Basic validation - parameters shouldn't be negative
                    return jsonify({
                        'success': False, 
                        'error': f'Invalid {param} value: {value}. Value cannot be negative.'
                    })

        except (ValueError, TypeError) as e:
            return jsonify({'success': False, 'error': f'Invalid parameter value: {str(e)}'})

        # Calculate water quality
        results = calculation_model.calculate_water_quality(params)

        # Prepare response
        response = {
            'success': True,
            'results': {
                'overall_condition': results['overall_condition'],
                'overall_score': results['overall_score'],
                'prediction_confidence': results['prediction_confidence'],
                'parameter_conditions': results['parameter_conditions'],
                'parameter_values': results['parameter_values'],
                'thresholds': results['thresholds']
            }
        }

        return jsonify(response)

    except ValueError as e:
        return jsonify({'success': False, 'error': f'Invalid parameter value: {str(e)}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 