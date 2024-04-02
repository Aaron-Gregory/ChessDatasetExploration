"""
We extract the following features:

Relating to player ratings:
* white rating
* black rating
* rating difference (white - black)
* whether game is rated

Relating to time limit:
* approximate time limit in hours (assuming 40 move game)

Related to time of play:
* month of year
* day of week
* hour of day

Related to position:
* First 1, 2, and 3 character of ECO codes for which there are more than 100 games available
*   (each of the above is one-hot encoded)

Target variable:
* Game outcome (white, black, and draw, encoded as +1, -1, and 0)
* Whether game ended in a draw

All bools are being stored as ints, because it reduces the resulting file to 37% of the size it would have otherwise.
"""

import pandas as pd

from scripts.duration_analysis import MAX_GAME_TIME_LIMIT_MINUTES
from scripts.position_analysis import CUTOFF
from src.features import (
    add_approx_time_limit_column,
    add_game_outcome_column,
    add_one_hot_encoding_for_truncated_eco_code,
    add_time_component_columns,
)


# Load the dataset
df = pd.read_csv("./data/chess_games.csv")
df.drop_duplicates(subset="id", keep="first", inplace=True)


# Drop irrelevant columns
df.drop(
    [
        "id",
        "white_id",
        "black_id",
        "opening_name",
        "last_move_at",
    ],
    axis=1,
    inplace=True,
)

# normalize ratings around 1500, with 1000 mapped to -1 and 2000 mapped to +1
df["white_rating"] = (df["white_rating"] / 500) - 3
df["black_rating"] = (df["black_rating"] / 500) - 3
df["white_rating_advantage"] = df["white_rating"] - df["black_rating"]
df["rated"] = df["rated"].astype(int)


# Compute approx time limit, after which increment code and number of opening moves are no longer needed
df = add_approx_time_limit_column(df, MAX_GAME_TIME_LIMIT_MINUTES)
df["approx_time_limit_hours"] = df["approx_time_limit_minutes"] / 60
df.drop(
    ["increment_code", "opening_ply", "approx_time_limit_minutes"], axis=1, inplace=True
)


# Add encodings of truncated versions of ECO codes for 1, 2, and 3 character versions
# After which, the original ECO codes are no longer needed
df = add_one_hot_encoding_for_truncated_eco_code(df, 1, CUTOFF)
df = add_one_hot_encoding_for_truncated_eco_code(df, 2, CUTOFF)
df = add_one_hot_encoding_for_truncated_eco_code(df, 3, CUTOFF)
df.drop("opening_eco", axis=1, inplace=True)


# Extract month of year, day of week, and hour of day
# Then remove the unix timestamp
df = add_time_component_columns(df)
df.drop("created_at", axis=1, inplace=True)


# Add numerical encoding of game outcome, and remove string encoding
df = add_game_outcome_column(df)
df.drop("winner", axis=1, inplace=True)


# Save the final design matrix
df.to_csv("./data/processed_chess_games.csv", index=False)
