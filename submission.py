"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

from os.path import join

import joblib
import pandas as pd

CODEBOOK_PATH = join("PreFer", "codebooks", "PreFer_codebook.csv")
KWARGS = {"encoding": "latin-1", "encoding_errors": "replace", "low_memory": False}
SURVEYS = ["cf", "ca", "cd", "ci", "ch", "cp", "gr", "cv", "he", "ma", "cr", "cs", "cw"]
WAVES = [f"{x:02}{chr(x+89)}" for x in range(8, 21)]


def clean_df(df, background_df=None):
    """
    Preprocess the input dataframe to feed the model.
    # If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
    background (pd.DataFrame): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

    Returns:
    pd.DataFrame: The cleaned dataframe with only the necessary columns and processed variables.
    """
    codebook = pd.read_csv(CODEBOOK_PATH, **KWARGS)

    # Load metadata
    meta = codebook[codebook["dataset"] == "PreFer_train_data.csv"]
    meta_bg = codebook[
        (codebook["dataset"] == "PreFer_train_background_data.csv")
        | (codebook["var_name"] == "nomem_encr")
    ]
    categorical_data = meta[meta["type_var"] == "categorical"]["var_name"].tolist()
    categorical_bg = meta_bg[meta_bg["type_var"] == "categorical"]["var_name"].tolist()
    numeric_data = meta[meta["type_var"] == "numeric"]["var_name"].tolist()
    numeric_bg = meta_bg[meta_bg["type_var"] == "numeric"]["var_name"].tolist()

    # Keep onnly categorical and numeric columns
    df = df[categorical_data + numeric_data]
    background_df = background_df[categorical_bg + numeric_bg]

    # Pivot waves wide to long
    df_long = df.melt(
        "nomem_encr", [f"cf{wave}_m" for wave in WAVES], "wavecode", "wave"
    )
    df_long = df_long.dropna()
    df_long["wavecode"] = df_long["wavecode"].str.extract(r"(\d\d\w)")
    df_long["wave"] = df_long["wave"].astype("int")

    # Add background info
    df_long = pd.merge(df_long, background_df, how="left", on=["nomem_encr", "wave"])

    # Pivot wave-specific variables wide to long
    for code in SURVEYS:
        # Question code xxx leaves room for 1000 questions
        for i in range(1000):
            value_vars = [
                c for c in df.columns if c[:2] == code and c[-3:] == f"{i:03}"
            ]
            if value_vars:
                tmp = df.melt("nomem_encr", value_vars, "wavecode", f"{code}{i:03}")
                tmp["wavecode"] = tmp["wavecode"].str.extract(r"(\d\d\w)")
                df_long = pd.merge(
                    df_long, tmp, how="left", on=["nomem_encr", "wavecode"]
                )

    # Impute NAs with -1
    df_long.fillna(-1)

    return df_long


def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    model = joblib.load(model_path)

    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)

    # Exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.drop(
        ["nomem_encr", "wavecode", "wave", "nohouse_encr"], axis=1
    )

    # Generate probability predictions that individual had a child
    predictions = model.predict_proba(vars_without_id)[:, 1]

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )

    # Combine predictions for individual
    df_predict = (
        df_predict.groupby("nomem_encr")["prediction"]
        .prod()
        .round()
        .astype(int)
        .reset_index()
    )

    # Return only dataset with predictions and identifier
    return df_predict
