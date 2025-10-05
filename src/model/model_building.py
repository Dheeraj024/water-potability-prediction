import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

# initial code

# train_data = pd.read_csv("./data/processed/train_processed.csv")

# X_train = train_data.drop("Potability", axis=1).values
# y_train = train_data["Potability"].values

# n_estimators = yaml.safe_load(open("params.yaml"))["model_building"]["n_estimators"]


# model = RandomForestClassifier(n_estimators= n_estimators)
# model.fit(X_train, y_train)

# pickle.dump(model, open("random_forest_model.pkl", "wb"))

# modular code

def load_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")

def load_params(params_path: str) -> int:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        return params["model_building"]["n_estimators"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {params_path}: {e}")
    
def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = df.drop("Potability", axis=1)
        y = df["Potability"]
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")

def train_model(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int) -> RandomForestClassifier:
    try:
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise Exception(f"Error training model: {e}")

def save_model(model: RandomForestClassifier, file_path: str) -> None:
    try:
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Error saving model to {file_path}: {e}")
    
def main():
    try:
        processed_data_path = "./data/processed/train_processed_mean.csv"
        model_output_path = "models/random_forest_model.pkl"
        params_filepath = "params.yaml"

        n_estimators = load_params(params_filepath)
        train_data = load_data(processed_data_path)
        X_train, y_train = prepare_data(train_data)
        model = train_model(X_train, y_train, n_estimators)

        save_model(model, model_output_path)
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()