import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scripts.position_analysis import CUTOFF
from src.features import add_time_component_columns, add_truncated_eco_column


def show_rating_analysis(df):
    # Rated vs Unrated Analysis
    plt.figure(figsize=(17, 6))

    # Map outcome to integer values
    outcome_color_mapping = {
        "white": "#DADADA",
        "draw": "#C04040",
        "black": "#3F3F3F",
    }
    df["outcome_color"] = df["winner"].map(outcome_color_mapping)
    df["rating_difference"] = df["white_rating"] - df["black_rating"]
    df_rated = df[df["rated"] == True]
    df_unrated = df[df["rated"] == False]

    # Subplot for rated distribution
    plt.subplot(1, 5, 1)
    sns.countplot(x="rated", data=df)
    plt.title("Rated Distribution")

    # Subplots for winner distribution
    order = ["white", "black", "draw"]
    plt.subplot(1, 5, 2)
    sns.countplot(x="winner", data=df_rated, palette=outcome_color_mapping, order=order)
    plt.title("Rated Winner Distribution")

    plt.subplot(1, 5, 3)
    sns.countplot(
        x="winner", data=df_unrated, palette=outcome_color_mapping, order=order
    )
    plt.title("Unrated Winner Distribution")

    # Subplot for scatter plot
    plt.subplot(1, 5, 4)
    plt.scatter(
        df_rated["white_rating"],
        df_rated["black_rating"],
        c=df_rated["outcome_color"],
        s=3,
    )
    plt.xlabel("White Rating")
    plt.ylabel("Black Rating")
    plt.title("Rated Game Outcomes")
    plt.grid(True)
    # Create legend for outcome colors
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=outcome_color_mapping["white"],
            markersize=10,
            label="White Win",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=outcome_color_mapping["draw"],
            markersize=10,
            label="Draw",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=outcome_color_mapping["black"],
            markersize=10,
            label="Black Win",
        ),
    ]
    plt.legend(handles=legend_handles, loc="upper left")
    # Subplot for scatter plot
    plt.subplot(1, 5, 5)
    plt.scatter(
        df_unrated["white_rating"],
        df_unrated["black_rating"],
        c=df_unrated["outcome_color"],
        s=10,
    )
    plt.xlabel("White Rating")
    plt.ylabel("Black Rating")
    plt.title("Unrated Game Outcome")
    plt.grid(True)
    # Create legend for outcome colors
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=outcome_color_mapping["white"],
            markersize=10,
            label="White Win",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=outcome_color_mapping["draw"],
            markersize=10,
            label="Draw",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=outcome_color_mapping["black"],
            markersize=10,
            label="Black Win",
        ),
    ]
    plt.legend(handles=legend_handles, loc="upper left")
    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title("Game Outcomes by Ratedness and Ratings")
    plt.show()

    # Analysis of game outcomes based on player ratings
    plt.figure(figsize=(17, 6))
    plt.subplot(1, 3, 1)
    sns.histplot(
        df["white_rating"], color="blue", kde=True, label="White Player Rating"
    )
    sns.histplot(df["black_rating"], color="red", kde=True, label="Black Player Rating")
    plt.title("Distribution of Player Ratings")
    plt.xlabel("Rating")
    plt.legend()

    plt.subplot(1, 3, 2)
    sns.histplot(
        df_rated["rating_difference"], color="red", kde=True, label="Rated games"
    )
    sns.histplot(
        df_unrated["rating_difference"], color="blue", kde=True, label="Unrated games"
    )
    plt.title("Distribution of Rating Differences")
    plt.xlabel("Rating Difference")
    plt.legend()

    plt.subplot(1, 3, 3)
    sns.histplot(
        df_rated[df_rated["winner"] == "draw"]["rating_difference"],
        color="red",
        kde=True,
        label="Rated games",
    )
    sns.histplot(
        df_unrated[df_unrated["winner"] == "draw"]["rating_difference"],
        color="blue",
        kde=True,
        label="Unrated games",
    )
    plt.title("Distribution of Rating Differences in Drawn Games")
    plt.xlabel("Rating Difference in Drawn Games")
    plt.legend()
    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title("Game Outcomes by Ratedness and Ratings")
    plt.show()


def show_time_analysis(df):
    # Convert timestamps to datetime objects
    df = add_time_component_columns(df)

    outcome_mapping = {
        "white": +1,
        "draw": 0,
        "black": -1,
    }
    df["outcome"] = df["winner"].map(outcome_mapping)

    plt.figure(figsize=(17, 8))
    plt.subplot(2, 3, 1)
    df.groupby("month_of_year").size().plot(kind="bar")
    plt.title("Number of Games by Month of Year")

    plt.subplot(2, 3, 2)
    df.groupby("day_of_week").size().plot(kind="bar")
    plt.title("Number of Games by Day of Week")

    plt.subplot(2, 3, 3)
    df.groupby("hour_of_day").size().plot(kind="bar")
    plt.title("Number of Games by Hour of Day")
    plt.xticks(rotation=0)

    # Define color mapping for winner
    winner_color_mapping = {"white": "skyblue", "black": "salmon", "draw": "lightgreen"}

    # Plot histograms for outcome densities over day of week, month of year, and hour of day
    game_counts = (
        df.groupby(["month_of_year", "winner"]).size().reset_index(name="count")
    )

    # Group by month_of_year and calculate the sum of counts for each month
    month_sums = game_counts.groupby("month_of_year")["count"].sum()

    # Divide the count of each row by the sum of counts for its month
    game_counts["normalized_count"] = game_counts.apply(
        lambda row: row["count"] / month_sums[row["month_of_year"]], axis=1
    )

    # Histogram for month of year
    plt.subplot(2, 3, 4)
    for outcome, color in winner_color_mapping.items():
        plt.scatter(
            game_counts.loc[game_counts["winner"] == outcome, "month_of_year"],
            game_counts.loc[game_counts["winner"] == outcome, "normalized_count"],
            color=color,
            label=outcome,
        )
    plt.title("Outcome Density by Month")
    plt.xlabel("Month of Year")
    plt.ylabel("Density")
    plt.legend()

    game_counts = df.groupby(["day_of_week", "winner"]).size().reset_index(name="count")

    # Group by day_of_week and calculate the sum of counts for each day
    day_sums = game_counts.groupby("day_of_week")["count"].sum()

    # Divide the count of each row by the sum of counts for its day
    game_counts["normalized_count"] = game_counts.apply(
        lambda row: row["count"] / day_sums[row["day_of_week"]], axis=1
    )

    # Scatter plot for day of week
    plt.subplot(2, 3, 5)
    for outcome, color in winner_color_mapping.items():
        plt.scatter(
            game_counts.loc[game_counts["winner"] == outcome, "day_of_week"],
            game_counts.loc[game_counts["winner"] == outcome, "normalized_count"],
            color=color,
            label=outcome,
        )
    plt.title("Outcome Density by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Density")
    plt.legend()

    game_counts = df.groupby(["hour_of_day", "winner"]).size().reset_index(name="count")

    # Group by hour_of_day and calculate the sum of counts for each hour
    hour_sums = game_counts.groupby("hour_of_day")["count"].sum()

    # Divide the count of each row by the sum of counts for its hour
    game_counts["normalized_count"] = game_counts.apply(
        lambda row: row["count"] / hour_sums[row["hour_of_day"]], axis=1
    )

    # Scatter plot for hour of day
    plt.subplot(2, 3, 6)
    for outcome, color in winner_color_mapping.items():
        plt.scatter(
            game_counts.loc[game_counts["winner"] == outcome, "hour_of_day"],
            game_counts.loc[game_counts["winner"] == outcome, "normalized_count"],
            color=color,
            label=outcome,
        )
    plt.title("Outcome Density by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title("Effect of Start Time on Game Outcome")
    plt.show()


def show_position_analysis(df, chars):
    # Extract the first and second characters of ECO code
    df, column_name = add_truncated_eco_column(df, chars, CUTOFF)

    num_games_by_eco = (
        df.groupby([column_name])[["id"]].size().reset_index(name="count")
    )

    # Plot ECO code vs average player rating
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 2, 1)
    sns.barplot(
        x=column_name,
        y="count",
        data=num_games_by_eco,
        hue="count",
        palette="dark:skyblue",
    )
    plt.title(f"Number of Games Played")
    plt.xlabel(f"ECO Code (first {chars} characters)")
    plt.ylabel("Number of Games")
    plt.xticks(rotation=45)
    plt.legend().remove()

    # Calculate average player ratings grouped ECO code
    avg_ratings_by_eco = (
        df.groupby([column_name])[["white_rating", "black_rating"]].mean().reset_index()
    )

    avg_ratings_by_eco["rating_mean"] = (
        avg_ratings_by_eco["white_rating"] + avg_ratings_by_eco["black_rating"]
    ) / 2

    # Plot ECO code vs average player rating
    plt.subplot(2, 2, 2)
    sns.barplot(
        x=column_name,
        y="rating_mean",
        data=avg_ratings_by_eco,
        hue="rating_mean",
        palette="dark:skyblue",
    )
    plt.title(f"Average Player Rating")
    plt.xlabel(f"ECO Code (first {chars} characters)")
    plt.ylabel("Average Rating")
    plt.xticks(rotation=45)
    plt.legend().remove()

    # Calculate win and draw rates grouped by the first and second characters of ECO code
    eco_outcomes = (
        df.groupby([column_name])["winner"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )

    # Plot ECO code vs likelihood of white win
    plt.subplot(2, 2, 3)
    sns.barplot(
        data=eco_outcomes["white"].reset_index(),
        x=column_name,
        y="white",
        hue="white",
        palette="dark:skyblue",
    )
    plt.title("Chance of White Winning")
    plt.xlabel(f"ECO Code (first {chars} characters)")
    plt.ylabel("Likelihood of White Win")
    plt.xticks(rotation=45)
    plt.legend().remove()

    # Plot ECO code vs likelihood of draw
    plt.subplot(2, 2, 4)
    sns.barplot(
        data=eco_outcomes["draw"].reset_index(),
        x=column_name,
        y="draw",
        hue="draw",
        palette="dark:skyblue",
    )
    plt.title("Chance of Draw")
    plt.xlabel(f"ECO Code (first {chars} characters)")
    plt.ylabel("Likelihood of Draw")
    plt.xticks(rotation=45)
    plt.legend().remove()
    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title(
        f"Effect of Position (first {chars} characters) on Outcome"
    )
    plt.show()


if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("./data/chess_games.csv")
    df.drop_duplicates(subset="id", keep="first", inplace=True)

    # show that player ratings affect the game outcome
    show_rating_analysis(df)
    show_time_analysis(df)
    show_position_analysis(df, 2)
