import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the preprocessed dataset
df = pd.read_csv("./data/processed_chess_games.csv")

# Define features and target variable
features = [
    "rated",
    "white_rating",
    "black_rating",
    "approx_time_limit_hours",
    "white_rating_advantage",
    "eco_1_A",
    "eco_1_B",
    "eco_1_C",
    "eco_1_D",
    "eco_1_E",
    "eco_2_A0",
    "eco_2_A1",
    "eco_2_A2",
    "eco_2_A4",
    "eco_2_A5",
    "eco_2_A8",
    "eco_2_B0",
    "eco_2_B1",
    "eco_2_B2",
    "eco_2_B3",
    "eco_2_B4",
    "eco_2_B5",
    "eco_2_B9",
    "eco_2_C0",
    "eco_2_C1",
    "eco_2_C2",
    "eco_2_C3",
    "eco_2_C4",
    "eco_2_C5",
    "eco_2_C6",
    "eco_2_C7",
    "eco_2_D0",
    "eco_2_D1",
    "eco_2_D2",
    "eco_2_D3",
    "eco_2_other",
    "eco_3_A00",
    "eco_3_A01",
    "eco_3_A04",
    "eco_3_A06",
    "eco_3_A10",
    "eco_3_A40",
    "eco_3_A45",
    "eco_3_B00",
    "eco_3_B01",
    "eco_3_B02",
    "eco_3_B06",
    "eco_3_B07",
    "eco_3_B10",
    "eco_3_B12",
    "eco_3_B20",
    "eco_3_B21",
    "eco_3_B23",
    "eco_3_B30",
    "eco_3_B32",
    "eco_3_B40",
    "eco_3_B50",
    "eco_3_B90",
    "eco_3_C00",
    "eco_3_C01",
    "eco_3_C02",
    "eco_3_C20",
    "eco_3_C21",
    "eco_3_C23",
    "eco_3_C24",
    "eco_3_C40",
    "eco_3_C41",
    "eco_3_C42",
    "eco_3_C44",
    "eco_3_C45",
    "eco_3_C46",
    "eco_3_C50",
    "eco_3_C55",
    "eco_3_C57",
    "eco_3_C60",
    "eco_3_C62",
    "eco_3_C65",
    "eco_3_D00",
    "eco_3_D02",
    "eco_3_D06",
    "eco_3_D10",
    "eco_3_D20",
    "eco_3_D30",
    "eco_3_other",
    "month_of_year",
    "day_of_week",
    "hour_of_day",
]
target = "outcome"

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=13
)

# Train a logistic regression model
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = logistic_regression.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Interpretation: Coefficients of logistic regression
pd.set_option("display.max_rows", None)
coefficients = pd.DataFrame(
    {"Feature": features, "Coefficient": logistic_regression.coef_[0]}
)

# Sort the coefficients by absolute value
coefficients["Absolute_Coefficient"] = coefficients["Coefficient"].abs()
coefficients_sorted = coefficients.sort_values(
    by="Absolute_Coefficient", ascending=True
)

# Print the sorted coefficients
print("\nCoefficients (Ascending in Significance):")
print(coefficients_sorted)
