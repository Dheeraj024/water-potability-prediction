import pandas as pd
import numpy as np
import pickle
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# initial code

# test_data = pd.read_csv("./data/processed/test_processed.csv")

# X_test = test_data.drop("Potability", axis=1).values
# y_test = test_data["Potability"].values

# model = pickle.load(open("random_forest_model.pkl", "rb"))

# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# metrics_dict = {
#     "accuracy": accuracy,
#     "precision": precision,
#     "recall": recall,
#     "f1": f1
# }

# with open("metrics.json", "w") as file:
#     json.dump(metrics_dict, file, indent=4)

# modular code

def load_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")

def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = df.drop("Potability", axis=1)
        y = df["Potability"]
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")   

def load_model(model_path: str):
    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise Exception(f"Error loading model from {model_path}: {e}")

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred) 
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")
    
def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {file_path}: {e}")

def main():
    try:
        test_data_path = "./data/processed/test_processed.csv"
        model_path = "models/random_forest_model.pkl"
        metrics_output_path = "reports/metrics.json"

        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics_dict = evaluate_model(model, X_test, y_test)

        save_metrics(metrics_dict, metrics_output_path)
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()