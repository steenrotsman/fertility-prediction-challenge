"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from submission import DROP_COLS, KWARGS, clean_df

parser = argparse.ArgumentParser(description="Train model.")
parser.add_argument("data_path", help="Path to data CSV file.")
parser.add_argument("background_data_path", help="Path to background data CSV file.")
parser.add_argument("ground_truth_path", help="Path to ground truth data CSV file.")
args = parser.parse_args()


def main():
    df = pd.read_csv(args.data_path, **KWARGS)
    bg = pd.read_csv(args.background_data_path, **KWARGS)
    gt = pd.read_csv(args.ground_truth_path, **KWARGS)

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
    X = model_df.drop(DROP_COLS + ["new_child"], axis=1)
    y = model_df["new_child"]

    # Classifier model
    model = LGBMClassifier(verbose=-1, random_seed=123)
    model.fit(X, y)
    joblib.dump(model, "model.joblib")

    # Tune classification threshold
    np.set_printoptions(linewidth=400)
    rng = np.random.default_rng(123)
    folds = 100
    f1s = [[] for i in range(folds)]
    for i in range(folds):
        state = rng.integers(0, 1000)
        df1, df2 = stratified_group_split(model_df, "nohouse_encr", y, 0.2, state)
        X1 = df1.drop(DROP_COLS + ["new_child"], axis=1)
        X2 = df2.drop(DROP_COLS + ["new_child"], axis=1)
        y1 = df1["new_child"]
        y2 = df2["new_child"]

        model = LGBMClassifier(verbose=-1, random_seed=123)
        model.fit(X1, y1)

        for thresh in range(0, 51, 1):
            y_pred = (model.predict_proba(X2)[:, 1] > thresh / 100).astype(int)
            f1 = f1_score(y2, y_pred)
            f1s[i].append(round(f1, 3))
    f1s = np.array(f1s)
    print(f1s.mean(axis=0).round(3))
    print(f1s.std(axis=0).round(3))
    print(np.argmax(f1s.mean(axis=0)))


def stratified_group_split(df, group_col, stratify, test_size=0.5, random_state=None):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df[group_col].values
    train_idx, test_idx = next(gss.split(df, groups=groups, y=stratify))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, test_df


if __name__ == "__main__":
    main()
