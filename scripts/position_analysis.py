import pandas as pd
import matplotlib.pyplot as plt

from src.features import add_truncated_eco_column

CUTOFF = 100


def show_games_by_code_before_and_after_cutoff(df, chars):
    # Truncate ECO codes to the specified number of characters
    df, column_name = add_truncated_eco_column(df, chars, 1)

    # Count the number of games by truncated ECO code
    games_by_eco = df[column_name].value_counts()

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    games_by_eco.plot(kind="bar", color="skyblue")
    plt.title(f"Number of Games by ECO Code (truncated to {chars} characters)")
    plt.xlabel("ECO Code")
    plt.ylabel("Number of Games")
    plt.xticks(rotation=45)  # Rotate x-axis tick labels for better readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Truncate ECO codes to the specified number of characters
    df, column_name = add_truncated_eco_column(df, chars, CUTOFF)
    games_by_eco_trunc = df[column_name].value_counts()

    # Plot the results
    plt.subplot(2, 1, 2)
    games_by_eco_trunc.plot(kind="bar", color="skyblue")
    plt.title(
        f'Number of Games by ECO Code (with "other" category, cutoff of {CUTOFF} games per codes)'
    )
    plt.xlabel("ECO Code")
    plt.ylabel("Number of Games")
    plt.xticks(rotation=45)  # Rotate x-axis tick labels for better readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("./data/chess_games.csv")
    df.drop_duplicates(subset="id", keep="first", inplace=True)

    show_games_by_code_before_and_after_cutoff(df, 3)
    show_games_by_code_before_and_after_cutoff(df, 2)
    show_games_by_code_before_and_after_cutoff(df, 1)
