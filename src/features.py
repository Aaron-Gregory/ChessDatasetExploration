"""
Source file for all feature extraction code.
"""

import pandas as pd


def add_game_outcome_column(df):
    """
    Takes a DataFrame with a "winner" column taking values "white", "black", and "draw".

    Adds on two columns and returns the result:
    - "outcome", which maps white/black to +/- 1, and draws to 0
    - "is_draw", which is 1 if the game is a draw and 0 else
    """
    df["outcome"] = df["winner"].map({"white": 1, "draw": 0, "black": -1})
    df["is_draw"] = (df["winner"] == "draw").astype(int)
    return df


def add_time_component_columns(df):
    # Convert timestamps to datetime objects and extract relevant components
    created_at_datetime = pd.to_datetime(df["created_at"], unit="ms")
    df["month_of_year"] = created_at_datetime.dt.month
    df["day_of_week"] = created_at_datetime.dt.dayofweek
    df["hour_of_day"] = created_at_datetime.dt.hour

    return df


def add_approx_time_limit_column(df, max_time_minutes):
    # Split the increment code into base time and increment time
    df[["base_minutes", "increment_seconds"]] = df["increment_code"].str.split(
        "+", expand=True
    )

    # Convert base time and increment time to numeric
    df["base_minutes"] = pd.to_numeric(df["base_minutes"], errors="coerce")
    df["increment_seconds"] = pd.to_numeric(df["increment_seconds"], errors="coerce")

    # Calculate the approximate time limit based on base time and increment time
    df["approx_time_limit_minutes"] = (
        df["base_minutes"] + df["increment_seconds"] * (40 - df["opening_ply"]) / 60
    )

    # Cap the approximate time limit at 60 minutes
    df["approx_time_limit_minutes"] = df["approx_time_limit_minutes"].clip(
        upper=max_time_minutes
    )

    df.drop(["base_minutes", "increment_seconds"], axis=1, inplace=True)

    return df


def add_game_duration_column(df):
    # Convert Unix Epoch timestamps to datetime objects
    df["created_at"] = pd.to_datetime(df["created_at"], unit="ms")
    df["last_move_at"] = pd.to_datetime(df["last_move_at"], unit="ms")

    # Compute game durations in seconds
    df["game_duration"] = (df["last_move_at"] - df["created_at"]).dt.total_seconds()

    return df


def add_truncated_eco_column(df, chars, size_cutoff):
    column_name = f"eco_{chars}_chars"

    # Truncate ECO codes to the specified number of characters
    df[column_name] = df["opening_eco"].str[:chars]

    # Count the number of games by truncated ECO code
    games_by_eco = df[column_name].value_counts()

    # Filter out codes with counts less than the cutoff
    codes_to_keep = games_by_eco[games_by_eco >= size_cutoff]

    # Replace codes with counts less than the cutoff with 'other'
    df.loc[~df[column_name].isin(codes_to_keep.index), column_name] = "other"

    return df, column_name


def add_one_hot_encoding_for_truncated_eco_code(df, chars, cutoff):
    # Add truncated codes column
    df, column_name = add_truncated_eco_column(df, chars, cutoff)

    # Convert new column to one-hot encoding
    one_hot_encoded = pd.get_dummies(df[column_name], prefix=f"eco_{chars}").astype(int)

    # Concatenate the one-hot encoded columns with the original DataFrame
    df = pd.concat([df, one_hot_encoded], axis=1)

    # Remove the categorical column
    df.drop(column_name, axis=1, inplace=True)

    return df
