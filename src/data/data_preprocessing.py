import pandas as pd
import numpy as np
import os

# Initial code

# train_data = pd.read_csv("./data/raw/train.csv")
# test_data = pd.read_csv("./data/raw/test.csv")


# def fill_missing_with_median(df):
#     for column in df.columns:
#         if df[column].isnull().any():
#             median_value = df[column].median()
#             df[column] = df[column].fillna(median_value)
#     return df

# train_processed_data = fill_missing_with_median(train_data)
# test_processed_data = fill_missing_with_median(test_data)

# data_path = os.path.join("data","processed")

# os.makedirs(data_path, exist_ok=True)

# train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"), index=False)
# test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"), index=False)

# Modular code

def load_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")
    
def fill_missing_with_mean(df):
    try:
        for column in df.columns:
            if df[column].isnull().any():
                mean_value = df[column].mean()
                df[column] = df[column].fillna(mean_value)
        return df
    except Exception as e:
        raise Exception(f"Error filling missing values: {e}")

def save_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {file_path}: {e}")
    
def main():
    try:
        raw_data_path = "./data/raw/"
        processed_data_path = os.path.join("data","processed")

        train_data = load_data(os.path.join(raw_data_path, "train.csv"))
        test_data = load_data(os.path.join(raw_data_path, "test.csv"))

        train_processed_data = fill_missing_with_mean(train_data)
        test_processed_data = fill_missing_with_mean(test_data)

        os.makedirs(processed_data_path, exist_ok=True)

        save_data(train_processed_data, os.path.join(processed_data_path,"train_processed_mean.csv"))
        save_data(test_processed_data, os.path.join(processed_data_path,"test_processed_mean.csv"))
    except Exception as e:
        raise Exception(f"Error in data preprocessing pipeline: {e}")
    
if __name__ == "__main__":
    main()