import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Read dataset
df = pd.read_csv("C:/Users/punug/OneDrive/Desktop/fourth-sem/machine_learning/placement_prediction/dataset/placement_predict_50K_Raw.csv")

# Create a copy of the dataset
data = df.copy()

# Remove leading/trailing spaces from text columns
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].str.strip()

# Display missing values before cleaning
print("Missing Values Before Cleaning:")
print(data.isnull().sum())

# Remove duplicate rows
before = data.shape[0]
data = data.drop_duplicates()
after = data.shape[0]

print("\nDuplicate Records Removed:", before - after)

# Separate numerical and categorical columns
num_cols = data.select_dtypes(include=np.number).columns.tolist()
cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()

# Handle missing values in numerical columns
if len(num_cols) > 0:
    num_imputer = SimpleImputer(strategy="mean")
    data[num_cols] = num_imputer.fit_transform(data[num_cols])

# Handle missing values in categorical columns
if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

    # Label Encoding
    label_encoders = {}

    for col in cat_cols:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        label_encoders[col] = encoder

# Display missing values after cleaning
print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

# Save cleaned dataset
data.to_csv(
    "C:/Users/punug/OneDrive/Desktop/fourth-sem/machine_learning/placement_prediction/dataset/clean_label_encode_M2.csv",
    index=False
)

print("\n==========================")
print("Original dataset is NOT modified.")
print("Label Encoding completed successfully.")
print("Output file:")
print("clean_label_encode_M2.csv")
print("==========================")