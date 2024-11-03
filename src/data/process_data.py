import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Logging setup configuration
def setup_logger(log_file='Process_Data.log', logger_name='Process Data logger'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

def load_data(train_data_path, test_data_path):
    try:
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        logger.info(f"Data successfully loaded from {train_data_path} and {test_data_path}.")
        return train_df, test_df
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"No data found: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise

def download_nltk_resources():
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        logger.info("NLTK resources downloaded successfully.")
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {e}")
        raise

def lemmatize_text(text):
    try:
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    except Exception as e:
        logger.error(f"Error during lemmatization: {e}")
        raise

def remove_stopwords(text):
    try:
        stop_words = set(stopwords.words("english"))
        return " ".join([word for word in text.split() if word not in stop_words])
    except Exception as e:
        logger.error(f"Error removing stop words: {e}")
        raise

def remove_numbers(text):
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        logger.error(f"Error removing numbers: {e}")
        raise

def to_lower_case(text):
    try:
        return text.lower()
    except Exception as e:
        logger.error(f"Error converting text to lower case: {e}")
        raise

def remove_punctuation(text):
    try:
        text = re.sub(r'[^\w\s]', ' ', text)
        return " ".join(text.split())
    except Exception as e:
        logger.error(f"Error removing punctuation: {e}")
        raise

def remove_urls(text):
    try:
        return re.sub(r'https?://\S+|www\.\S+', '', text)
    except Exception as e:
        logger.error(f"Error removing URLs: {e}")
        raise

# Process Text
def process_text(df):
    try:
        df['content'] = df['content'].apply(to_lower_case)
        df['content'] = df['content'].apply(remove_stopwords)
        df['content'] = df['content'].apply(remove_numbers)
        df['content'] = df['content'].apply(remove_punctuation)
        df['content'] = df['content'].apply(remove_urls)
        df['content'] = df['content'].apply(lemmatize_text)
        logger.info("Text normalization completed successfully.")
        return df
    except KeyError as e:
        logger.error(f"Column not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

# Save Processed Data
def save_data(df, output_path, file_name):
    try:
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, file_name)
        df.to_csv(full_path, index=False)
        logger.info(f"Data successfully saved at {full_path}.")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main(train_data_path, test_data_path, output_data_path):
    download_nltk_resources()
    train_data, test_data = load_data(train_data_path, test_data_path)
    train_data_processed = process_text(train_data)
    test_data_processed = process_text(test_data)
    save_data(train_data_processed, output_data_path, 'train_data_processed.csv')
    save_data(test_data_processed, output_data_path, 'test_data_processed.csv')

if __name__ == "__main__":
    TRAIN_DATA_PATH = './data/raw/train.csv'
    TEST_DATA_PATH = './data/raw/test.csv'
    OUTPUT_DATA_PATH = './data/processed'

    try:
        main(TRAIN_DATA_PATH, TEST_DATA_PATH, OUTPUT_DATA_PATH)
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}")
        raise
