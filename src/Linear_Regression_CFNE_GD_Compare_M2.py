import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = r"C:/Users/punug/OneDrive/Desktop/fourth-sem/machine_learning/placement_prediction/dataset/final_preprocess_M2.csv"

data = pd.read_csv(DATASET_PATH)

# ============================================================
# 2. HANDLE MISSING VALUES
# ============================================================

print("Dataset shape:", data.shape)

print("\nTotal missing values before cleaning:")
print(data.isna().sum().sum())

# Separate features and target

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Replace infinity values

X = X.replace([np.inf, -np.inf], np.nan)

# Fill missing feature values with the column mean

imputer = SimpleImputer(strategy="mean")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

# Remove missing target values if any

valid_rows = ~pd.isna(y)

X = X.loc[valid_rows]

y = y.loc[valid_rows]

print("\nTotal missing values after cleaning:")
print(X.isna().sum().sum())

# Convert to NumPy arrays

X = X.values
y = y.values

# ============================================================
# 3. CREATE IMAGE OUTPUT FOLDER
# ============================================================

IMAGE_FOLDER = r"C:/Users/punug/OneDrive/Desktop/fourth-sem/machine_learning/placement_prediction/outputs/Linear_Regression_CFNE_GD_Compare_M2"

os.makedirs(IMAGE_FOLDER, exist_ok=True)

print("\nImage output folder:")
print(IMAGE_FOLDER)

# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================================
# 5. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ============================================================
# 6. CLOSED-FORM NORMAL EQUATION
# ============================================================

X_train_bias = np.c_[
    np.ones((X_train.shape[0], 1)),
    X_train
]

X_test_bias = np.c_[
    np.ones((X_test.shape[0], 1)),
    X_test
]

# Use pseudo-inverse for stability

theta = np.linalg.pinv(
    X_train_bias.T @ X_train_bias
) @ X_train_bias.T @ y_train

pred_normal = X_test_bias @ theta

mse_normal = mean_squared_error(
    y_test,
    pred_normal
)

r2_normal = r2_score(
    y_test,
    pred_normal
)

print("\n------ Closed Form Normal Equation ------")

print("MSE:", mse_normal)

print("R2:", r2_normal)

# ============================================================
# 7. GRADIENT DESCENT
# ============================================================

X_train_gd = np.c_[
    np.ones((X_train_scaled.shape[0], 1)),
    X_train_scaled
]

X_test_gd = np.c_[
    np.ones((X_test_scaled.shape[0], 1)),
    X_test_scaled
]

m = len(y_train)

theta_gd = np.zeros(
    X_train_gd.shape[1]
)

learning_rate = 0.01

epochs = 1000

loss_history = []

for epoch in range(epochs):

    predictions = X_train_gd @ theta_gd

    errors = predictions - y_train

    gradients = (
        2 / m
    ) * X_train_gd.T @ errors

    theta_gd -= learning_rate * gradients

    loss = np.mean(errors ** 2)

    loss_history.append(loss)

pred_gd = X_test_gd @ theta_gd

mse_gd = mean_squared_error(
    y_test,
    pred_gd
)

r2_gd = r2_score(
    y_test,
    pred_gd
)

print("\n------ Gradient Descent ------")

print("MSE:", mse_gd)

print("R2:", r2_gd)

# ============================================================
# 8. COMPARISON
# ============================================================

print("\n=========== COMPARISON ===========")

print("\nNormal Equation")

print("MSE =", mse_normal)

print("R2 =", r2_normal)

print("\nGradient Descent")

print("MSE =", mse_gd)

print("R2 =", r2_gd)

# ============================================================
# IMAGE 1: ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    pred_normal,
    alpha=0.5,
    label="Normal Equation"
)

plt.scatter(
    y_test,
    pred_gd,
    alpha=0.5,
    label="Gradient Descent"
)

minimum = min(
    y_test.min(),
    pred_normal.min(),
    pred_gd.min()
)

maximum = max(
    y_test.max(),
    pred_normal.max(),
    pred_gd.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted Values")

plt.legend()

plt.grid(True)

image1 = os.path.join(
    IMAGE_FOLDER,
    "actual_vs_predicted.png"
)

plt.savefig(
    image1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# IMAGE 2: RESIDUAL COMPARISON
# ============================================================

normal_residuals = y_test - pred_normal

gd_residuals = y_test - pred_gd

plt.figure(figsize=(8, 6))

plt.scatter(
    pred_normal,
    normal_residuals,
    alpha=0.5,
    label="Normal Equation"
)

plt.scatter(
    pred_gd,
    gd_residuals,
    alpha=0.5,
    label="Gradient Descent"
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Values")

plt.ylabel("Residuals")

plt.title("Residual Comparison")

plt.legend()

plt.grid(True)

image2 = os.path.join(
    IMAGE_FOLDER,
    "residual_comparison.png"
)

plt.savefig(
    image2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# IMAGE 3: LOSS CURVE
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    range(1, epochs + 1),
    loss_history
)

plt.xlabel("Epoch")

plt.ylabel("MSE")

plt.title("Gradient Descent Convergence")

plt.grid(True)

image3 = os.path.join(
    IMAGE_FOLDER,
    "gradient_descent_loss.png"
)

plt.savefig(
    image3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nImages saved successfully.")

print(image1)
print(image2)
print(image3)

print("\nPROCESS COMPLETED SUCCESSFULLY")