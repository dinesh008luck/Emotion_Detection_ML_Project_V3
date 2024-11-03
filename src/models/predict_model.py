import pandas as pd
import os
import logging
import yaml
import joblib
import json
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, roc_auc_score

# Logger setup
logger = logging.getLogger('predict_model')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('predict_model.log')
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)


# Load parameters from params.yaml
def load_params(params_path):
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info("Parameters loaded successfully.")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise


# Load Model
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


# Load Data
def load_data(test_path):
    try:
        test_df = pd.read_csv(test_path)
        logger.info("Test data loaded successfully.")
        return test_df
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise


# Predict
def predict(model, X_test):
    try:
        predictions = model.predict(X_test)
        logger.info("Predictions made successfully.")
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


# Evaluate
def evaluate_model(y_test, predictions):
    try:
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        logger.info(f"Model evaluation completed with accuracy: {accuracy}")
        logger.info(f"Classification Report:\n{report}")
        return accuracy, report
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


# Save predictions in CSV
def save_predictions(predictions, test_data, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_df = test_data.copy()
        output_df['predicted_label'] = predictions
        output_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved successfully to {output_path}.")
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        raise


# Calculate and log metrics
def calculate_metrics(y_true, y_pred, y_pred_prob):
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        auc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr')

        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"AUC: {auc}")

        metrics_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise


# Save metrics to JSON
def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise


def main():
    params_path = 'params.yaml'
    model_path = './models/trained_model.pkl'
    test_path = './data/features/feature_test_data.csv'
    output_path = './data/predictions/predicted_results.csv'
    metrics_output_path = 'reports/metrics.json'

    try:
        params = load_params(params_path)
        max_features = params['build_features']['max_features']

        model = load_model(model_path)
        test_df = load_data(test_path)

        X_test = test_df.drop(columns=['label'])
        y_test = test_df['label']

        y_pred = predict(model, X_test)
        y_pred_prob = model.predict_proba(X_test)

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_prob)

        # Save metrics to JSON
        save_metrics(metrics, metrics_output_path)

        save_predictions(y_pred, test_df, output_path)

        logger.info(f"Prediction completed with max_features: {max_features}")
    except Exception as e:
        logger.critical(f"Critical error in prediction process: {e}")
        raise


if __name__ == "__main__":
    main()
