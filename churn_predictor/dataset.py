import pandas as pd
from churn_predictor.utils.paths_internal import data_raw_dir

def get_dataframe():
    dataset_path = data_raw_dir("Churn_Modelling.csv")
    return pd.read_csv(dataset_path)