# Dataset Preparation for Chess Games

This repository contains a coding assignment completed for a Machine Learning Engineer (MLE) position. The purpose is to evaluate skills in data preparation, feature engineering, and the design aspects of machine learning projects.

## Project Objective

The goal here is to prepare and analyze a dataset for machine learning. The general focus should be on quality of approach, methodologies, and decision-making processes. 

## Implementation vs. Design

Given the iterative nature of machine learning projects, the emphasis here is on data preparation and understanding. Implementation should include, but not be limited to:

- **Data Exploration**: Conduct exploratory data analysis to identify patterns, anomalies, and potential biases.
- **Data Preparation**: Clean and preprocess the data to construct a design matrix suitable for model training.

While actual model training is not required, keep in mind the following:

- **Model Selection**: Justify the choice of model(s) for this task.
- **Evaluation Strategy**: Describe how you would evaluate the model's performance, including metrics.
- **Interpretability**: Discuss how the model's results could be interpreted, especially regarding feature importance.

## Data

The dataset comprises chess games from lichess.com, with the task being to predict the `Winner` (either `White`, `Black`, or `Draw`).

**Note**: This dataset may contain anomalies or missing values. Approach it as you would real-world data.

### Source

[Chess Games Dataset](./data/chess_games.csv)

### Schema

- `Id`: Unique game identifier (String)
- `Rated`: Whether the game is rated (Bool)
- `Created_at`: Game start time (Unix Epoch, UInt)
- `Last_move_at`: Game end time (Unix Epoch, UInt)
- `Winner`: Game outcome (`White`, `Black`, `Draw`)
- `Increment_code`: Time control format (`{Minutes}+{Seconds}`)
- `White_id`: White player's alias (String)
- `White_rating`: White player's ELO (UInt)
- `Black_id`: Black player's alias (String)
- `Black_rating`: Black player's ELO (UInt)
- `Opening_eco`: Opening code (String)
- `Opening_name`: Name of the opening (String)
- `Opening_ply`: Moves in the opening phase (UInt)

## Project Guidelines

- **Format**: Python scripts.
- **Documentation**: Comment your code to explain your process, decisions, and any assumptions.

## Evaluation Criteria

Focus on the thoroughness of the data preparation, the clarity of documentation, and the depth of the design discussion.

Enjoy!