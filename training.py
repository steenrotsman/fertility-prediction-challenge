"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""

import argparse

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from submission import KWARGS, clean_df

parser = argparse.ArgumentParser(description="Train model.")
parser.add_argument("data_path", help="Path to data CSV file.")
parser.add_argument("background_data_path", help="Path to background data CSV file.")
parser.add_argument("ground_truth_path", help="Path to ground truth data CSV file.")
args = parser.parse_args()


def main():
    df = pd.read_csv(args.data_path, **KWARGS)
    bg = pd.read_csv(args.background_data_path, **KWARGS)
    gt = pd.read_csv(args.ground_truth_path)

    data = clean_df(df, bg)
    train_save_model(data, gt)


def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")

    # Filter cases for whom the outcome is not available
    model_df = model_df[~model_df["new_child"].isna()]

    # Split into X and y
    X = model_df.drop(
        ["nomem_encr", "new_child", "wavecode", "wave", "nohouse_encr"], axis=1
    )
    y = model_df["new_child"]

    # Classifier model
    model = LGBMClassifier(verbose=-1, random_seed=123)
    model.fit(X, y)
    joblib.dump(model, "model.joblib")

    # Get estimate of score
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, stratify=y, random_state=123)

    model.fit(X1, y1)
    y_pred = model.predict(X2)
    print(f1_score(y2, y_pred))

    model.fit(X2, y2)
    y_pred = model.predict(X1)
    print(f1_score(y1, y_pred))
    # 0.7675988428158148
    # 0.7977422389463782


if __name__ == "__main__":
    main()
