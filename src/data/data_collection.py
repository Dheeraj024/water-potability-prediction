import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml


# initial code 

# test_size =yaml.safe_load(open("params.yaml"))["data_collection"]["test_size"]
# data = pd.read_csv(r"D:\DK\Work_Life\Technical_Skills\Projects\Project_1\dataset\water_potability.csv")
# train_data, test_data = train_test_split(data, test_size= test_size, random_state=42)

# data_path = os.path.join("data","raw")
# os.makedirs(data_path, exist_ok=True)

# train_data.to_csv(os.path.join(data_path,"train.csv"), index=False)
# test_data.to_csv(os.path.join(data_path,"test.csv"), index=False)

# modular code

def load_params(file_path: str) -> float:
    try:
        with open(file_path, "r") as file:
            params = yaml.safe_load(file)
        return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {file_path}: {e}")


def load_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")



def split_data(data: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return train_test_split(data, test_size=test_size, random_state=42)
    except Exception as e:
        raise Exception(f"Error splitting data: {e}")



def save_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {file_path}: {e}")


def main():
    try:
        params_filepath = "params.yaml"
        data_filepath = r"D:\DK\Work_Life\Technical_Skills\Projects\Project_1\dataset\water_potability.csv"
        raw_data_path = os.path.join("data","raw")

        data = load_data(data_filepath)
        test_size = load_params(params_filepath)
        train_data, test_data = split_data(data, test_size)
        os.makedirs(raw_data_path, exist_ok=True)

        save_data(train_data, os.path.join(raw_data_path,"train.csv"))
        save_data(test_data, os.path.join(raw_data_path,"test.csv"))
    except Exception as e:
        print(f"An error occurred in the data collection process: {e}")


if __name__ == "__main__":
    main()