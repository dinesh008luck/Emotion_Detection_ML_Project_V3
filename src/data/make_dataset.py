import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

import logging

# logging configration

logger = logging.getLogger('Make_dataset_log')
logger.setLevel('DEBUG')

consolor_handler = logging.StreamHandler()
consolor_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('error.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s- %(name)s- %(levelname)s- %(message)s')
consolor_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(consolor_handler)
logger.addHandler(file_handler)

# test size define


def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        test_size = params['make_dataset']['test_size']
        return test_size
    except FileNotFoundError:
        logger.error("The Params file was not found")
    except KeyError as e:
        logger.error('Missing key in the parameters file')
    except yaml.YAMLError as e:
        logger.error('YAML parsing error')
    except Exception as e:
        logger.error('Unexpected error')
        raise


def read_date(url: str) -> pd.DataFrame:
    try:
        # Load the dataset
        df = pd.read_csv(url)
        return df
    except pd.errors.EmptyDataError:
        logger.error('No data found at URL')
    except pd.errors.ParserError:
        logger.error('Error parsing the CSV file at URL')
    except Exception as e:
        logger.error('Unexpected error')
        raise


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Drop the 'tweet_id' column as it is not needed
        df.drop(columns=['tweet_id'], inplace=True)
        # Define sentiment categories
        category_mapping = {
            'empty': 2,
            'sadness': 0,
            'enthusiasm': 1,
            'neutral': 2,
            'worry': 0,
            'surprise': 1,
            'love': 1,
            'fun': 1,
            'hate': 0,
            'happiness': 1,
            'boredom': 0,
            'relief': 1,
            'anger': 0
        }

        # Map sentiments to categories
        df['sentiment'] = df['sentiment'].map(category_mapping)

        # Create a copy of the DataFrame
        df_final = df.copy()

        return df_final
    except KeyError as e:
        logger.error('Missing column in the DataFrame')
    except Exception as e:
        logger.error('Unexpected error')
        raise


def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        
        # Create the directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)
        # Save the training and test sets to CSV files
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except Exception as e:
        logger.error('Unexpected error while saving data')
        raise


def main():
    try:
        test_size = load_params('params.yaml')
        df = read_date('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        df_final = process_data(df)

        if df_final.empty:
            logger.error('Data processing failed')
            return

        # Split the data into training and test sets
        train_data, test_data = train_test_split(df_final, test_size= test_size, random_state=60)

        # Define the path to save the data
        data_path = os.path.join("data", "raw")

        # Create the directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)

        save_data(data_path, train_data, test_data)

    except Exception as e:
        logger.error('Fail to complete the make dataset error in main')
        raise



if __name__ == "__main__":
    main()
