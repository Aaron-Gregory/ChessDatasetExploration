"""
This script gives a summary of the "created_at" and "last_move_at" fields, as well as game durations.
It shows that something is wrong with these fields in the chess_games.csv dataset.

First off, it is noteworthy that all timestamps seem to be in the format "{1.5f}E+12" (e.g. "1.50421E+12"),
which means they have a granularity of 10**7 ms, or about 3 hours.

The effect of this 3 hour granularity is that in 91.64% (all but 8.36%) of games the values of created_at
and last_move_at coincide, preventing us from extracting useful duration information from these columns.
Instead, we will approximate game duration by using the increment code and assuming a 40 move game.
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.features import add_approx_time_limit_column, add_game_duration_column

MAX_GAME_TIME_LIMIT_MINUTES = 60


def start_and_end_time_analysis(df):
    df = add_game_duration_column(df)

    # Print out summary statistics of game durations
    print("Summary statistics of game durations:")
    print(df[["created_at", "last_move_at", "game_duration"]].describe())

    # Calculate the percentage of games with nonzero duration
    nonzero_duration_percentage = (df["game_duration"] > 0).mean() * 100
    print(
        f"Percentage of games with nonzero duration: {nonzero_duration_percentage:.2f}%"
    )

    # Plot sorted game durations
    sorted_durations = (
        df["game_duration"].sort_values(ascending=False).reset_index(drop=True)
    )
    print(sorted_durations)
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_durations, marker=".", linestyle="None", markersize=3)
    plt.xlabel("Game Index")
    plt.ylabel("Game Duration (seconds)")
    plt.title("Sorted Game Durations")
    plt.grid(True)
    plt.show()


def plot_games_by_approx_time_limit(df):
    # compute the data to show
    df = add_approx_time_limit_column(df, MAX_GAME_TIME_LIMIT_MINUTES)

    # Print out summary statistics of game time limits
    print("Summary statistics of game time limits:")
    print(df[["approx_time_limit_minutes"]].describe())

    # Sort the DataFrame by the approximate time limit in descending order
    df_sorted = df.sort_values(
        by="approx_time_limit_minutes", ascending=True
    ).reset_index(drop=True)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(df_sorted["approx_time_limit_minutes"], marker="o", linestyle="None")
    plt.title(
        f"Games Sorted by Approximate Time Limit, Capped at {MAX_GAME_TIME_LIMIT_MINUTES} minutes"
    )
    plt.xlabel("Game Index (Sorted by Approximate Time Limit)")
    plt.ylabel("Approximate Time Limit (minutes)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("./data/chess_games.csv")
    df.drop_duplicates(subset="id", keep="first", inplace=True)

    # Show that start and end times mostly coincide
    start_and_end_time_analysis(df)

    # Show the approx time we'll use
    plot_games_by_approx_time_limit(df)
