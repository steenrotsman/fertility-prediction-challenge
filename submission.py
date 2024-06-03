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

import joblib
import pandas as pd
from scipy.stats import linregress

THRESHOLD = 0.28
DROP_COLS = ["nomem_encr", "wave", "nohouse_encr", "outcome_available"]
KWARGS = {"encoding": "latin-1", "encoding_errors": "replace", "low_memory": False}


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
    df = drop_irrelevant_cols(df)
    df = join_background(df, background_df)
    df = add_cols(df)

    return df


def drop_irrelevant_cols(df):
    drop_waves = ["08a", "09b", "10c", "11d", "12e", "13f", "14g", "15h", "16i"]
    old_waves = [c for c in df.columns for wave in drop_waves if wave in c]
    fieldwork_period = ["cf17j_m", "cf18k_m", "cf19l_m", "cf20m_m"]
    datetime_string = df.select_dtypes(include=["object"]).columns.tolist()
    fake_df = pd.read_csv("PreFer_fake_data.csv", **KWARGS)
    datetime_string += fake_df.select_dtypes(include=["object"]).columns.tolist()
    df = df.drop(old_waves + fieldwork_period + datetime_string, axis=1)

    return df


def join_background(df, bg):
    max_waves = bg.groupby("nomem_encr")["wave"].max().reset_index()
    return df.merge(max_waves, on="nomem_encr").merge(bg, on=["nomem_encr", "wave"])


def add_cols(df):
    for num in ("130", "128", "029"):
        df[f"cf{num}"] = df[f"cf20m{num}"]
        for wave in ("19l", "18k", "17j"):
            df.loc[df[f"cf{num}"].isna(), f"cf{num}"] = df.loc[
                df[f"cf{num}"].isna(), f"cf{wave}{num}"
            ]
    waves = ("20m", "19l", "18k", "17j")
    df["cf031"] = df[[f"cf{wave}031" for wave in waves]].max(axis=1)
    df["cf456"] = df[[f"cf{wave}456" for wave in waves]].max(axis=1)
    df["birthyear_last_child"] = df[
        [f"cf{wave}{num}" for wave in waves for num in range(456, 471)]
    ].max(axis=1)

    n_respondents = df.groupby("nohouse_encr").count()["nomem_encr"].reset_index()
    n_respondents["n_respondents"] = n_respondents["nomem_encr"]
    n_respondents = n_respondents[["nohouse_encr", "n_respondents"]]
    df = pd.merge(df, n_respondents, on="nohouse_encr")

    vars = (
        "woning",
        "sted",
        "woonvorm",
        "partner",
        "oplzon",
        "oplmet",
        "burgstat",
        "belbezig",
    )
    for var in vars:
        unique = (
            pd.melt(
                df,
                id_vars="nomem_encr",
                value_vars=[f"{var}_{year}" for year in range(2017, 2021)],
            )
            .groupby("nomem_encr")
            .value.nunique()
            .reset_index()[["nomem_encr", "value"]]
            .rename(columns={"value": f"{var}_unique"})
        )
    df = pd.merge(df, unique, on="nomem_encr")

    var = "nettohh_f"
    nettohh = pd.melt(
        df,
        id_vars="nomem_encr",
        value_vars=[f"{var}_{year}" for year in range(2017, 2021)],
    ).dropna()
    nettohh["variable"] = nettohh.variable.str.replace("nettohh_f_", "").astype("int64")
    slope = (
        nettohh.groupby("nomem_encr")
        .apply(lambda v: linregress(v.variable, v.value)[0])
        .reset_index()
        .rename({0: "nettohh_f_slope"}, axis=1)
    )
    df = pd.merge(df, slope, on="nomem_encr", how="outer")

    return df


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
    vars_without_id = df.drop(DROP_COLS, axis=1)

    # Generate probability predictions that individual had a child
    predictions = model.predict_proba(vars_without_id)[:, 1]

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )

    # Combine predictions for individual
    df_predict["prediction"] = (df_predict["prediction"] > THRESHOLD).astype(int)

    # Return only dataset with predictions and identifier
    return df_predict
