import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

DATASET_PATH = r"C:/Users/punug/OneDrive/Desktop/fourth-sem/machine_learning/placement_prediction/dataset/final_preprocess_M2.csv"

OUTPUT_FOLDER = r"C:/Users/punug/OneDrive/Desktop/fourth-sem/machine_learning/placement_prediction/outputs/Linear_Regression_with_Metrics_M2"

IMAGE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "images"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("LINEAR REGRESSION - PLACEMENT PREDICTION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Columns:")

for column in df.columns:
    print(column)

feature_columns = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterviewScore"
]

target_column = "PlacementStatus"

required_columns = feature_columns + [target_column]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

model_df = df[required_columns].copy()

x = model_df[feature_columns]
y = model_df[target_column]

imputer = SimpleImputer(strategy="mean")

x = pd.DataFrame(
    imputer.fit_transform(x),
    columns=feature_columns
)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(x_train))
print("Testing samples:", len(x_test))

model = LinearRegression()

model.fit(x_train, y_train)

print("\nModel training completed")

print("\nIntercept (b0):")
print(model.intercept_)

coefficient_df = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient": model.coef_
})

print("\nCoefficients:")
print(coefficient_df)

equation = f"y = {model.intercept_:.4f}"

for coefficient, feature in zip(
    model.coef_,
    feature_columns
):
    equation += (
        f" + ({coefficient:.4f}*{feature})"
    )

print("\nLinear Regression Equation:")
print(equation)

y_pred = model.predict(x_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

results = x_test.copy()

results["Actual"] = y_test.values
results["Predicted"] = y_pred

results["Residual"] = (
    results["Actual"]
    - results["Predicted"]
)

results["Absolute_Error"] = abs(
    results["Residual"]
)

prediction_file = os.path.join(
    OUTPUT_FOLDER,
    "linear_regression_predictions.csv"
)

results.to_csv(
    prediction_file,
    index=False
)

coefficient_file = os.path.join(
    OUTPUT_FOLDER,
    "linear_regression_coefficients.csv"
)

coefficient_df.to_csv(
    coefficient_file,
    index=False
)

metrics_df = pd.DataFrame({
    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ],
    "Value": [
        mae,
        mse,
        rmse,
        r2
    ]
})

metrics_file = os.path.join(
    OUTPUT_FOLDER,
    "linear_regression_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Placement")
plt.ylabel("Predicted Placement")

plt.title(
    "Actual vs Predicted Placement"
)

plt.grid(True)

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "actual_vs_predicted.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    results["Residual"],
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Placement")
plt.ylabel("Residual")

plt.title("Residual Plot")

plt.grid(True)

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "residual_plot.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(10, 6))

plt.bar(
    coefficient_df["Feature"],
    coefficient_df["Coefficient"]
)

plt.xlabel("Features")
plt.ylabel("Coefficient")

plt.title(
    "Feature Coefficients"
)

plt.xticks(
    rotation=30
)

plt.grid(axis="y")

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "feature_coefficients.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

equation_file = os.path.join(
    OUTPUT_FOLDER,
    "linear_regression_equation.txt"
)

with open(
    equation_file,
    "w",
    encoding="utf-8"
) as file:
    file.write(
        "Linear Regression Equation\n"
    )

    file.write(
        "=" * 40 + "\n"
    )

    file.write(equation)

print("\nPROCESS COMPLETED SUCCESSFULLY")