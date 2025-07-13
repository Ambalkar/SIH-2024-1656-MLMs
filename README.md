# Weather Condition Prediction Project

## Overview
This project provides a machine learning-based weather condition prediction system with safety level classification. It includes a trained TensorFlow model to predict weather conditions based on temperature, humidity, and precipitation data. The system exposes a Flask API for making predictions and a simple frontend UI for users to input a city name and view weather predictions along with safety levels.

## Features
- Train a neural network model on historical weather data to classify weather conditions.
- Flask API to serve the trained model and provide prediction endpoints.
- Fetch live weather data from Open-Meteo API or user input for predictions.
- Safety level classification based on predicted weather conditions.
- Frontend interface to input city names, fetch weather data from OpenWeather API, and display predictions.

## Project Structure
- `ml.py`: Script to train the TensorFlow model, save the model and label encoder, and includes functions to fetch live weather data and predict conditions.
- `app.py`: Flask API server that loads the trained model and label encoder, exposes endpoints for health check, prediction, and fetching live weather data for prediction.
- `index2.html`: Frontend UI for users to input city names and view weather predictions and safety levels.
- `weather_condition_predictor.h5`: Saved TensorFlow model file (generated after training).
- `label_encoder.pkl`: Saved label encoder for weather condition labels (generated after training).
- `WeatherDataset_with_conditions_updated.csv`: Historical weather dataset used for training.
- Other files: `label_encoder_classes.pkl`, `params.txt`, and safety predictor models (not currently used in the main API).

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Dependencies
The project requires the following Python packages:
- Flask
- Flask-CORS
- numpy
- pandas
- scikit-learn
- tensorflow
- requests

You can install these packages using pip:

```bash
pip install flask flask-cors numpy pandas scikit-learn tensorflow requests
```

## Usage

### 1. Train the Model
To train the weather condition prediction model, run:

```bash
python ml.py
```

This will:
- Load and preprocess the historical weather dataset (`WeatherDataset_with_conditions_updated.csv`).
- Train a neural network model.
- Save the trained model as `weather_condition_predictor.h5`.
- Save the label encoder as `label_encoder.pkl`.

### 2. Run the Flask API Server
Start the Flask API server by running:

```bash
python app.py
```

The server will start on `http://0.0.0.0:5000`.

#### API Endpoints
- `GET /health`: Health check endpoint to verify if the model and label encoder are loaded.
- `POST /predict`: Accepts JSON input with features `temperature_2m`, `relative_humidity_2m`, and `precipitation` to return predicted weather condition, safety level, and confidence.
- `GET /fetch_and_predict`: Fetches current weather data from Open-Meteo API (default location Kochi, India) and returns prediction.

### 3. Use the Frontend
Open `index2.html` in a web browser. The frontend allows you to:
- Enter a city name.
- Fetch current weather data from OpenWeather API.
- Send relevant weather features to the Flask API for prediction.
- Display the weather condition, safety level, and prediction confidence.

**Note:** Replace the placeholder OpenWeather API key in `index2.html` with your own API key from [OpenWeather](https://openweathermap.org/api).

## Safety Level Classification
The system classifies safety levels based on predicted weather conditions as follows:

| Weather Condition | Safety Level     |
|-------------------|------------------|
| Clear             | Safe             |
| Partly Cloudy     | Safe             |
| Cloudy            | Moderately Safe  |
| Drizzle           | Moderately Safe  |
| Fog               | Caution          |
| Rain              | Caution          |
| Heavy Rain        | Unsafe           |
| Thunderstorm      | Unsafe           |
| Snow              | Unsafe           |
| Sleet             | Unsafe           |

## Notes
- The model is trained on historical weather data and may require retraining with updated datasets for improved accuracy.
- The frontend depends on an active Flask API server running locally or accessible via network.
- The project currently uses Open-Meteo API for live weather data fetching in the backend and OpenWeather API in the frontend for city-based weather data.

## License
This project is provided as-is without any warranty.

## Contact
For questions or support, please contact the project maintainer.
