import pandas as pd
from churn_predictor.utils.paths_internal import data_raw_dir
from churn_predictor.utils.paths_internal import data_processed_dir

def get_raw_dataframe():
    dataset_path = data_raw_dir("Churn_Modelling.csv")
    return pd.read_csv(dataset_path)

def get_cleaned_dataframe():
    dataset_path = data_processed_dir("cleaned_dataset.csv")
    return pd.read_csv(dataset_path)
