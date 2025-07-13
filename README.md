# Water Quality Analysis System

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Usage](#usage)
6. [System Architecture](#system-architecture)
   - [Models](#models)
   - [Web Application](#web-application)
7. [Data Management](#data-management)
8. [Maintenance and Support](#maintenance-and-support)

## Overview

A comprehensive web application for water quality analysis that combines two powerful models:
1. **Prediction Model**: Forecasts water quality conditions using historical data
2. **Calculation Model**: Provides real-time water quality assessment based on current measurements

## Features

- Interactive web interface for both prediction and calculation models
- Real-time predictions and calculations
- Visual representation of results using charts
- Detailed model metrics and feature importance analysis
- Responsive design for both desktop and mobile devices
- Comprehensive error handling and validation
- Real-time parameter validation against thresholds
- Detailed recommendations for unsafe parameters

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository or download the source code
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. Ensure required data files are in the project directory:
   - `updated_wqns.csv` - For the prediction model
   - `historical_wqns.csv` (optional) - For enhanced prediction accuracy
   - `threshold.csv` - For parameter validation

2. Start the Flask application:
   ```bash
   python app.py
   ```

3. Access the application at:
   ```
   http://localhost:5000
   ```

### Using the Models

#### Prediction Model
1. Select the "Prediction Model" tab
2. Choose a date and time for prediction
3. Click "Predict Water Quality" to see results
4. View:
   - Predicted condition
   - Probability breakdown
   - Model metrics
   - Feature importance analysis

#### Calculation Model
1. Select the "Calculation Model" tab
2. Enter water quality parameters:
   - Salinity (0-35 ppt)
   - pH (6.5-8.5)
   - Dissolved Oxygen (4.0-6.0 mg/L)
   - Dissolved Methane (0-0.5 mg/L)
   - CDOM (0-5 m-1)
   - Temperature (15-30°C)
3. Click "Calculate Water Quality" for assessment
4. Review:
   - Overall water quality score
   - Parameter conditions
   - Threshold comparisons
   - Recommendations

## System Architecture

### Models

#### 1. Water Quality Prediction Model
- **Type**: Random Forest Classifier
- **Features**:
  - Water quality parameters (6)
  - Temporal features (4)
- **Output**: Binary classification (Safe/Unsafe)
- **Implementation**: See [Model Details](#model-details)

#### 2. Water Quality Calculation Model
- **Type**: Neural Network (TensorFlow)
- **Structure**: 6-64-32-1 architecture
- **Input**: Current parameters
- **Output**: Quality assessment with confidence
- **Implementation**: See [Model Details](#model-details)

### Web Application

#### 1. Backend (Flask)
- **Routes**:
  - `/`: Main dashboard
  - `/predict`: Prediction endpoint
  - `/calculate`: Calculation endpoint
- **Features**:
  - Request handling
  - Data validation
  - Model management
  - Error handling

#### 2. Frontend (HTML/JavaScript)
- **Components**:
  - Navigation tabs
  - Input forms
  - Results display
  - Visualizations
- **Features**:
  - Responsive design
  - Real-time updates
  - Interactive charts
  - Error feedback

## Data Management

### Data Sources
- `updated_wqns.csv`: Current year's data
- `historical_wqns.csv`: Historical data (optional)
- `threshold.csv`: Parameter thresholds

### Parameters
| Parameter | Safe Range | Unit | Significance |
|-----------|------------|------|--------------|
| pH | 6.5-8.5 | pH | Water acidity/alkalinity |
| Temperature | 15-30 | °C | Water temperature |
| Dissolved Oxygen | 4.0-6.0 | mg/L | Oxygen levels |
| Dissolved Methane | 0-0.5 | mg/L | Organic decomposition |
| CDOM | 0-5 | m-1 | Organic matter |
| Salinity | 0-35 | ppt | Salt content |

### Data Collection
- **Frequency**: Hourly measurements
- **Location**: Kochi, India
- **Source**: INCOIS API
- **Endpoint**: https://gemini.incois.gov.in/OceanDataAPI/api/wqns/Kochi/

### Data Quality
- Missing value imputation
- Outlier detection
- Physical/chemical validation
- Historical consistency checks

## Maintenance and Support

### Model Maintenance
1. **Regular Updates**:
   - Model retraining
   - Threshold reviews
   - Performance monitoring

2. **Data Validation**:
   - Input validation
   - Outlier handling
   - Missing value management

3. **Performance Monitoring**:
   - Accuracy checks
   - Cross-validation
   - Feature importance updates

### Support
For issues or questions, please contact the development team.

## Model Details

### Prediction Model Implementation
```python
# Data Preprocessing
def preprocess_data(self, historical_data=None):
    # Convert timestamps
    # Extract features
    # Handle missing values
    # Scale features

# Model Training
def train_model(self, historical_data=None):
    # 5-fold cross-validation
    # Feature scaling
    # Model parameters:
    #   - n_estimators: 100
    #   - max_depth: 8
    #   - min_samples_split: 10
    #   - min_samples_leaf: 4
    #   - max_features: 'sqrt'

# Prediction
def predict_water_quality(self, input_date):
    # Generate features
    # Scale input
    # Make prediction
    # Return probabilities
```

### Calculation Model Implementation
```python
# Parameter Validation
def check_parameter_condition(self, value, param_name):
    # Compare thresholds
    # Return condition

# Quality Calculation
def calculate_water_quality(self, params):
    # Validate parameters
    # Scale input
    # Get prediction
    # Calculate score
    # Check conditions
```

## File Structure
```
├── app.py                      # Flask application
├── water_quality_prediction_model.py    # Prediction model
├── water_quality_calculation_model.py   # Calculation model
├── templates/
│   └── index.html             # Web interface
├── requirements.txt           # Dependencies
├── updated_wqns.csv          # Current data
├── historical_wqns.csv       # Historical data
└── threshold.csv             # Parameter thresholds
``` 