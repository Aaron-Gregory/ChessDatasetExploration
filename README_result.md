## Processed Chess Games Dataset

The processed dataset is derived from the original Chess Games Dataset, with various features engineered to facilitate machine learning tasks. The goal remains the same: predicting the winner of a chess game.

### Features

#### Relating to Player Ratings:
- `white_rating`: ELO rating of the white player, normalized around 1500 such that 2000 maps to 1 and 1000 maps to -1 (Float)
- `black_rating`: ELO rating of the black player, normalized around 1500 such that 2000 maps to 1 and 1000 maps to -1 (Float)
- `white_rating_advantage`: Difference in normalized ratings between white and black players (Float)
- `rated`: Indicator for whether the game is rated (Integer, 0 for False, 1 for True)

#### Related to Time Limit:
- `approx_time_limit_hours`: Approximate time limit for the game in hours, assuming an average of 40 moves (Float)

#### Related to Time of Gameplay:
- `month_of_year`: Month of the year when the game was played (Unsigned Integer, 1-12)
- `day_of_week`: Day of the week when the game was played (Unsigned Integer, 0-6)
- `hour_of_day`: Hour of the day when the game was played (Unsigned Integer, 0-23)

#### Related to Position:
- `eco_1_*`: One-hot encoding for the first character of the truncated ECO code. `*` ranges from `A` to `E`. (Binary)
- `eco_2_*`: One-hot encoding for the first two characters of the truncated ECO code. `*` is either two characters of a ECO code, or `other`. Some ECO codes will not be represented, as they fall into `other`. (Binary)
- `eco_3_*`: One-hot encoding for the first three characters of the truncated ECO code. `*` is either a full 3-character ECO code, or `other`. Some ECO codes will not be represented, as they fall into `other`. (Binary)

#### Target Variables:
- `outcome`: Encoded outcome of the game, where `1` represents white win, `-1` represents black win, and `0` represents a draw (Integer)
- `is_draw`: Indicator for whether the game ended in a draw (Integer, 0 for False, 1 for True)

### Dataset Information

- **Size**: The dataset contains 20058 datapoints. Every ECO code segment represented appears at least 100 times, as do the `other` labels.
- **Anomalies**: The original dataset did not have trustworthy game duration info - over 90% of games had the same timestamp for game creation and last move.
- **Storage**: The dataset is stored in CSV format as `processed_chess_games.csv`. Expanding the ECO codes into one-hot vectors substantially increases the size of the file, but this has been done in the interest of simplifying preprocessing when the dataset is used.

### Usage

This dataset has been evaluated on two models, logistic regression and random forests. Both hit an accuracy of ~63% for winner prediction. Random forest GINI importances suggest that normalized ratings are by far the most significant feature, with the time limit and time at which the game is played being the next most significant predictors of outcome. Logistic regression also keys in most on the players' normalized rating differences and game time limit, but then focusses on the openings being played. This difference is likely due to the GINI importance of openings being diffused over the elements of the one-hot vectors in the case of random forests.