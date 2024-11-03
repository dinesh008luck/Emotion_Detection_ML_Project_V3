import numpy as np
import pandas as pd
import os
import yaml
import logging
from sklearn.feature_extraction.text import CountVectorizer

def setup_logger(log_file='feature_building.log', logger_name='feature_build_logger'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

def load_params(param_file):
    try:
        with open(param_file, 'r') as file:
            params = yaml.safe_load(file)
        max_features = params['build_features']['max_features']
        logger.info("Parameters loaded successfully.")
        return max_features
    except FileNotFoundError:
        logger.error(f"Parameter file {param_file} not found.")
        raise
    except KeyError as e:
        logger.error(f"Missing key in the parameters file: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading parameters: {e}")
        raise

def load_data(train_path, test_path):
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.info(f"Data successfully loaded from {train_path} and {test_path}.")
        return train_data, test_data
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"No data found: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise

def preprocess_data(train_data, test_data):
    try:
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        
        # Check for required columns
        if 'content' not in train_data.columns or 'sentiment' not in train_data.columns:
            raise KeyError("Required column(s) 'content' or 'sentiment' not found in train data.")
        if 'content' not in test_data.columns or 'sentiment' not in test_data.columns:
            raise KeyError("Required column(s) 'content' or 'sentiment' not found in test data.")
        
        X_train = train_data['content'].values
        Y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        Y_test = test_data['sentiment'].values
        logger.info("Data preprocessing completed successfully.")
        return X_train, Y_train, X_test, Y_test
    except KeyError as e:
        logger.error(f"Missing key in DataFrame: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data preprocessing: {e}")
        raise

def apply_bow(X_train, X_test, max_features):
    try:
        if not len(X_train) or not len(X_test):
            raise ValueError("Input data is empty, unable to perform Bag of Words transformation.")
            
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logger.info("Bag of Words transformation completed successfully.")
        return X_train_bow, X_test_bow
    except Exception as e:
        logger.error(f"Error during Bag of Words transformation: {e}")
        raise

# Save Data
def save_data(train_df, test_df, output_path):
    try:
        os.makedirs(output_path, exist_ok=True)
        train_df.to_csv(os.path.join(output_path, "feature_train_data.csv"), index=False)
        test_df.to_csv(os.path.join(output_path, "feature_test_data.csv"), index=False)
        logger.info(f"Data successfully saved at {output_path}.")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

# Main Workflow
def main():
    param_file = "params.yaml"
    train_path = './data/processed/train_data_processed.csv'
    test_path = './data/processed/test_data_processed.csv'
    output_path = './data/features'

    try:
        max_features = load_params(param_file)
        train_data, test_data = load_data(train_path, test_path)
        X_train, Y_train, X_test, Y_test = preprocess_data(train_data, test_data)
        X_train_bow, X_test_bow = apply_bow(X_train, X_test, max_features)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = Y_train
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = Y_test

        save_data(train_df, test_df, output_path)
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
