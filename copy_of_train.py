import numpy as np
import pandas as pd
import joblib
import json
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
database_path = "database/database.csv"
df = pd.read_csv(database_path)

# Encode categorical column
encoder = LabelEncoder()
df["highway_type"] = encoder.fit_transform(df["highway_type"])
joblib.dump(encoder, "label_encoder.pkl")  # Save encoder for future use

# Define input (X) and target (y)
X = df.drop(columns=["safety", "priority"])  # Drop target variables
y = df[["safety", "priority"]]  # Select target variables

# Rename feature names to numerical format (f0, f1, ...)
feature_map = {col: f"f{i}" for i, col in enumerate(X.columns)}
X = X.rename(columns=feature_map)  # Rename columns in DataFrame

# Save feature mapping
with open("feature_map.json", "w") as f:
    json.dump(feature_map, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, objective="reg:squarederror")
model.fit(X_train, y_train)

# Save the trained model in JSON format (not pickle)
model.save_model("xgb_model.json")
