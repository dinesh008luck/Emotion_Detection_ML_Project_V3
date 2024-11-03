import pandas as pd
import os
import logging
import yaml
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Logger setup
logger = logging.getLogger('train_model')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('train_model.log')
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Load parameters from params.yaml
def load_params(params_path):
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            n_estimators = params['train_model']['n_estimators']
            learning_rate = params['train_model']['learning_rate']
        logger.info("Parameters loaded successfully.")
        return n_estimators, learning_rate
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

# Load Data
def load_data(train_path):
    try:
        train_df = pd.read_csv(train_path)
        logger.info("Training data loaded successfully.")
        return train_df
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise

# Train Model
def train_model(X_train, y_train, n_estimators, learning_rate):
    try:
        model = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
        model.fit(X_train, y_train)
        logger.info("Model trained successfully.")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

# Save Model
def save_model(model, model_path):
    try:
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}.")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def main():
    params_path = 'params.yaml'
    train_path = './data/features/feature_train_data.csv'
    model_path = './models/trained_model.pkl'

    try:
        n_estimators, learning_rate = load_params(params_path)

        train_df = load_data(train_path)

        X_train = train_df.drop(columns=['label'])
        y_train = train_df['label']

        model = train_model(X_train, y_train, n_estimators, learning_rate)
        save_model(model, model_path)

        logger.info("Training completed successfully.")
    except Exception as e:
        logger.critical(f"Critical error in training process: {e}")
        raise

if __name__ == "__main__":
    main()
