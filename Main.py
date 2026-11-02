
import pandas as pd

from RWD_KNN import RWDKNN

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)



# Load SCADA dataset


data = pd.read_csv("SCADA.csv")



# Define predictors and target


features = [
    "Bearing temperature",
    "Vibration (g)",
    "Power (kw)"
]

target = "Generator temperature"

X = data[features].values
y = data[target].values



# Chronological train-test split


split_index = int(0.8 * len(data))

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]



# Initialize RWD-KNN


rwd_model = RWDKNN(
    k_values=[1, 3, 5, 7, 9]
)



# Select optimal K using training data


cv_results = rwd_model.select_k(
    X_train,
    y_train,
    n_splits=3
)

print("\nCross-validation results:")
print(cv_results)

print(
    "\nOptimal K:",
    rwd_model.best_k
)



# Fit final RWD-KNN on training data


rwd_model.fit(
    X_train,
    y_train
)

rwd_predictions = rwd_model.predict(
    X_test
)



# Standard KNN comparison


standard_knn = KNeighborsRegressor(
    n_neighbors=rwd_model.best_k
)

standard_knn.fit(
    X_train,
    y_train
)

standard_predictions = standard_knn.predict(
    X_test
)



# Linear regression comparison


linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(
    X_test
)



# Evaluation


models = {
    "Standard KNN": standard_predictions,
    "RWD-KNN": rwd_predictions,
    "Linear Regression": linear_predictions
}

print("\n--- Model Comparison ---")

for name, predictions in models.items():

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print(f"\n{name}")
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))



# Display feature relevance weights


print("\nFeature Relevance Weights:")

for feature, weight in zip(
    features,
    rwd_model.feature_weights
):

    print(
        f"{feature}: {weight:.4f}"
    )



# Real-time prediction


new_observation = [[
    89.2,      # Bearing temperature
    0.82,      # Vibration
    1250.0     # Power
]]

predicted_temperature = (
    rwd_model.predict(new_observation)
)

print(
    "\nPredicted Generator Temperature:",
    round(float(predicted_temperature[0]), 2),
    "°C"
)
