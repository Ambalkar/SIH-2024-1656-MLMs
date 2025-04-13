import pandas as pd
import requests
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

# Step 1: Load and preprocess the historical weather dataset
historical_data_path = 'WeatherDataset_with_conditions_updated.csv'
data = pd.read_csv(historical_data_path)

# Step 2: Ensure the data contains only relevant columns and handle any preprocessing needed
feature_columns = ['temperature_2m', 'relative_humidity_2m', 'precipitation']
target_column = 'conditions'

# Handle missing values
data = data[feature_columns + [target_column]].dropna()

# Encode the target column
label_encoder = LabelEncoder()
data[target_column] = label_encoder.fit_transform(data[target_column])

# Step 3: Split data for training and testing
X = data[feature_columns].values
y = data[target_column].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define and train the TensorFlow model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model accuracy on test set: {accuracy * 100:.2f}%")

# Save the model for future use
model_filename = 'weather_condition_predictor.h5'
model.save(model_filename)
print(f"Trained model saved as {model_filename}")
# Also save the label encoder for consistent predictions
import pickle
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)
print("Label encoder saved as label_encoder.pkl")

# Step 5: Function to fetch live data from Open-Meteo API
def fetch_live_weather_data():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 9.9399,
        "longitude": 76.2602,
        "current": True,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        print(f"API response received successfully")
        
        # Extract current weather data
        current_data = data.get('current', {})
        
        # Create a dictionary with the required features
        weather_data = {
            'temperature_2m': current_data.get('temperature_2m', None),
            'relative_humidity_2m': current_data.get('relative_humidity_2m', None),
            'precipitation': current_data.get('precipitation', 0.0)  # Often not in current, might need to get from hourly
        }
        
        # If any data is missing from current, try to get it from the first hourly entry
        if None in weather_data.values() and 'hourly' in data:
            hourly = data['hourly']
            if weather_data['temperature_2m'] is None and 'temperature_2m' in hourly:
                weather_data['temperature_2m'] = hourly['temperature_2m'][0]
            if weather_data['relative_humidity_2m'] is None and 'relative_humidity_2m' in hourly:
                weather_data['relative_humidity_2m'] = hourly['relative_humidity_2m'][0]
            if weather_data['precipitation'] is None and 'precipitation' in hourly:
                weather_data['precipitation'] = hourly['precipitation'][0]
        
        print(f"Extracted weather data: {weather_data}")
        return weather_data
    else:
        print(f"Error fetching data from API: {response.status_code}")
        print(f"Response content: {response.text}")
        return None

# Step 6: Predict weather condition based on live data
def predict_live_weather_condition():
    live_data = fetch_live_weather_data()
    if live_data:
        # Check if any values are still None
        for key, value in live_data.items():
            if value is None:
                print(f"Warning: Missing value for {key}, setting to 0")
                live_data[key] = 0
        
        # Convert live data to DataFrame
        live_data_df = pd.DataFrame([live_data])
        
        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in live_data_df.columns:
                print(f"Warning: Missing column {col}, adding with default value 0")
                live_data_df[col] = 0

        try:
            # Load the trained model
            loaded_model = load_model(model_filename)
            
            # Load the label encoder
            with open('label_encoder.pkl', 'rb') as le_file:
                loaded_label_encoder = pickle.load(le_file)
            
            # Make prediction
            prediction = loaded_model.predict(live_data_df[feature_columns])
            predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
            predicted_class = loaded_label_encoder.inverse_transform([predicted_class_index])[0]
            
            print(f"Predicted weather condition: {predicted_class}")
            return predicted_class
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    else:
        print("No live data available for prediction")
        return None

# Execute prediction on live data
if __name__ == "__main__":
    predict_live_weather_condition()