import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score
)
import joblib
from datetime import datetime, timedelta
import warnings

class WaterQualityPredictor:
    def __init__(self, dataset_path):
        """
        Initialize the Water Quality Prediction Model
        
        :param dataset_path: Path to the CSV file containing water quality data
        """
        # Suppress warnings
        warnings.filterwarnings('ignore')
        
        # Load the dataset
        self.df = pd.read_csv(dataset_path)
        
        # Convert observation time to datetime
        self.df['observationTime'] = pd.to_datetime(self.df['observationTime'])
        
        # Extract additional time-based features
        self.df['hour'] = self.df['observationTime'].dt.hour
        self.df['month'] = self.df['observationTime'].dt.month
        self.df['day_of_week'] = self.df['observationTime'].dt.dayofweek
        self.df['day_of_year'] = self.df['observationTime'].dt.dayofyear
        
        # Prepare label encoder for condition
        self.label_encoder = LabelEncoder()
    
    def preprocess_data(self, historical_data=None):
        """
        Preprocess the data for machine learning
        
        :param historical_data: Optional historical dataset for more comprehensive training
        :return: Preprocessed features and encoded target variable
        """
        # Use current dataset or combine with historical data if provided
        if historical_data is not None:
            # Preprocess historical data similarly
            historical_data['observationTime'] = pd.to_datetime(historical_data['observationTime'])
            historical_data['hour'] = historical_data['observationTime'].dt.hour
            historical_data['month'] = historical_data['observationTime'].dt.month
            historical_data['day_of_week'] = historical_data['observationTime'].dt.dayofweek
            historical_data['day_of_year'] = historical_data['observationTime'].dt.dayofyear
            
            # Combine datasets
            combined_df = pd.concat([self.df, historical_data], ignore_index=True)
        else:
            combined_df = self.df
        
        # Select features for prediction
        features = [
            'salinity', 'ph', 'dissolvedoxygen', 
            'dissolvedmethane', 'cdom', 'temperature', 
            'hour', 'month', 'day_of_week', 'day_of_year'
        ]
        
        # Prepare features
        X = combined_df[features]
        
        # Handle missing values by imputing with mean
        X = X.fillna(X.mean())
        
        # Encode target variable
        y = self.label_encoder.fit_transform(combined_df['condition'])
        
        return X, y

    def train_model(self, historical_data=None, test_size=0.2, random_state=42):
        """
        Train a Random Forest Classifier with comprehensive evaluation
        
        :param historical_data: Optional historical dataset for more comprehensive training
        :param test_size: Proportion of dataset to include in test split
        :param random_state: Controls the shuffling applied to the data before applying the split
        :return: Dictionary containing model, scaler, and detailed evaluation metrics
        """
        # Prepare features and target
        X, y = self.preprocess_data(historical_data)
        
        # Perform stratified k-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        
        # Initialize lists to store cross-validation metrics
        cv_accuracy_scores = []
        cv_f1_scores = []
        cv_precision_scores = []
        cv_recall_scores = []
        
        # Perform cross-validation
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,  # Reduced number of trees
                max_depth=8,      # Reduced max depth
                min_samples_split=10,  # Increased min samples split
                min_samples_leaf=4,    # Added min samples leaf
                max_features='sqrt',   # Use sqrt of features
                random_state=random_state,
                class_weight='balanced'
            )
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val_scaled)
            
            # Calculate metrics
            cv_accuracy_scores.append(accuracy_score(y_val, y_pred))
            cv_f1_scores.append(f1_score(y_val, y_pred, average='weighted'))
            cv_precision_scores.append(precision_score(y_val, y_pred, average='weighted'))
            cv_recall_scores.append(recall_score(y_val, y_pred, average='weighted'))
        
        # Calculate mean and std of cross-validation metrics
        mean_accuracy = np.mean(cv_accuracy_scores)
        std_accuracy = np.std(cv_accuracy_scores)
        mean_f1 = np.mean(cv_f1_scores)
        std_f1 = np.std(cv_f1_scores)
        mean_precision = np.mean(cv_precision_scores)
        std_precision = np.std(cv_precision_scores)
        mean_recall = np.mean(cv_recall_scores)
        std_recall = np.std(cv_recall_scores)
        
        # Train final model on full dataset
        final_scaler = StandardScaler()
        X_scaled = final_scaler.fit_transform(X)
        
        final_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=random_state,
            class_weight='balanced'
        )
        final_model.fit(X_scaled, y)
        
        # Prepare detailed evaluation metrics
        return {
            'model': final_model,
            'scaler': final_scaler,
            'label_encoder': self.label_encoder,
            'accuracy': mean_accuracy,
            'accuracy_std': std_accuracy,
            'f1_score': mean_f1,
            'f1_score_std': std_f1,
            'precision': mean_precision,
            'precision_std': std_precision,
            'recall': mean_recall,
            'recall_std': std_recall,
            'cv_scores': {
                'accuracy': cv_accuracy_scores,
                'f1': cv_f1_scores,
                'precision': cv_precision_scores,
                'recall': cv_recall_scores
            }
        }

    def predict_water_quality(self, input_date, model_results, generate_dynamic_features=True):
        """
        Predict water quality for a specific date with more dynamic feature generation
        
        :param input_date: Date to predict water quality
        :param model_results: Dictionary containing trained model and preprocessing objects
        :param generate_dynamic_features: Whether to dynamically generate features
        :return: Predicted water quality condition with probability
        """
        # If dynamic feature generation is enabled, create more comprehensive input
        if generate_dynamic_features:
            # Generate input features with more variation
            features_variation = self._generate_feature_variations()
        else:
            # Use base average features
            features_variation = pd.DataFrame({
                'salinity': [self.df['salinity'].mean()],
                'ph': [self.df['ph'].mean()],
                'dissolvedoxygen': [self.df['dissolvedoxygen'].mean()],
                'dissolvedmethane': [self.df['dissolvedmethane'].mean()],
                'cdom': [self.df['cdom'].mean()],
                'temperature': [self.df['temperature'].mean()]
            })
        
        # Add time-based features
        for index in range(len(features_variation)):
            features_variation.loc[index, 'hour'] = input_date.hour
            features_variation.loc[index, 'month'] = input_date.month
            features_variation.loc[index, 'day_of_week'] = input_date.weekday()
            features_variation.loc[index, 'day_of_year'] = input_date.timetuple().tm_yday
        
        # Scale the input features
        input_scaled = model_results['scaler'].transform(features_variation)
        
        # Predict water quality with probabilities
        prediction_probas = model_results['model'].predict_proba(input_scaled)
        prediction_encoded = model_results['model'].predict(input_scaled)
        
        # Get the most likely prediction and its probability
        most_likely_index = np.argmax(prediction_probas.mean(axis=0))
        most_likely_prediction = model_results['label_encoder'].inverse_transform([most_likely_index])[0]
        
        return {
            'condition': most_likely_prediction,
            'probabilities': {
                condition: prob for condition, prob in 
                zip(model_results['label_encoder'].classes_, prediction_probas.mean(axis=0))
            }
        }

    def _generate_feature_variations(self):
        """
        Generate variations of input features based on historical data ranges
        
        :return: DataFrame with feature variations
        """
        np.random.seed(42)  # For reproducibility
        num_variations = 20  # Number of feature combinations to generate
        
        feature_ranges = {
            'salinity': (self.df['salinity'].min(), self.df['salinity'].max()),
            'ph': (self.df['ph'].min(), self.df['ph'].max()),
            'dissolvedoxygen': (self.df['dissolvedoxygen'].min(), self.df['dissolvedoxygen'].max()),
            'dissolvedmethane': (self.df['dissolvedmethane'].min(), self.df['dissolvedmethane'].max()),
            'cdom': (self.df['cdom'].min(), self.df['cdom'].max()),
            'temperature': (self.df['temperature'].min(), self.df['temperature'].max())
        }
        
        variations = []
        for _ in range(num_variations):
            variation = {}
            for feature, (min_val, max_val) in feature_ranges.items():
                # Generate random values within the feature's range
                variation[feature] = np.random.uniform(min_val, max_val)
            variations.append(variation)
        
        return pd.DataFrame(variations)

def main():
    try:
        # Load current year's data
        current_predictor = WaterQualityPredictor('updated_wqns.csv')
        
        # Optional: Load previous year's data for more comprehensive training
        try:
            historical_data = pd.read_csv('historical_wqns.csv')
        except FileNotFoundError:
            historical_data = None
        
        # Train the model
        model_results = current_predictor.train_model(historical_data)
        
        print("\nDetailed Classification Report:")
        print(model_results['classification_report'])
        
        # Analyze feature importance
        features = ['salinity', 'ph', 'dissolvedoxygen', 'dissolvedmethane', 'cdom', 'temperature', 
                    'hour', 'month', 'day_of_week', 'day_of_year']
        print("\nFeature Importance:")
        for feature, importance in sorted(
            zip(features, model_results['model'].feature_importances_), 
            key=lambda x: x[1], 
            reverse=True
        ):
            print(f"{feature}: {importance:.4f}")
        
        # Interactive prediction for specific dates
        while True:
            try:
                date_input = input("\nEnter a date and time for prediction (YYYY-MM-DD HH:MM, or 'quit' to exit): ")
                
                if date_input.lower() == 'quit':
                    break
                
                sample_date = datetime.strptime(date_input, '%Y-%m-%d %H:%M')
                prediction = current_predictor.predict_water_quality(sample_date, model_results)
                
                # Display prediction with detailed probability breakdown
                print(f"\nPredicted Water Quality for {sample_date}:")
                print(f"Most Likely Condition: {prediction['condition']}")
                print("\nCondition Probabilities:")
                for condition, probability in prediction['probabilities'].items():
                    print(f"{condition}: {probability*100:.2f}%")
                
            except ValueError:
                print("Invalid date format. Please enter the date in the format YYYY-MM-DD HH:MM.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your dataset is correctly formatted and exists.")

if __name__ == "__main__":
    main()