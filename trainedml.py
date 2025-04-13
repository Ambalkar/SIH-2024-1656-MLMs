# Updated code for `trainedml.py`
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

class WaterQualityPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def preprocess_data(self, data_path):
        data = pd.read_csv(data_path)
        X = data.drop('Conditions', axis=1)
        y = data['Conditions']
        return X, y

    def train(self, data_path):
        X, y = self.preprocess_data(data_path)

        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            print(classification_report(y_test, predictions))

    def predict(self, input_data):
        prediction = self.model.predict([input_data])
        probabilities = self.model.predict_proba([input_data])[0]
        labels = self.model.classes_
        return prediction[0], dict(zip(labels, probabilities))

    def save_model(self, path):
        joblib.dump(self, path)

if __name__ == '__main__':
    predictor = WaterQualityPredictor()
    predictor.train('updated_wqns.csv')  # Update path as needed
    predictor.save_model('WaterQualityPredictor.pkl')
