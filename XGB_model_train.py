import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor  # Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load dataset
database_path = "database.csv"
df = pd.read_csv(database_path)

# Check column names for debugging
print("Columns in dataset:", df.columns.tolist())

# Encode "highway_type" using LabelEncoder
encoder = LabelEncoder()
df["highway_type"] = encoder.fit_transform(df["highway_type"])

# Save LabelEncoder for future use
joblib.dump(encoder, "label_encoder.pkl")

# Define input (X) and target (y)
X = df.drop(columns=["safety", "priority"])  # Drop target variables
y = df[["safety", "priority"]]  # Select target variables

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Regressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the trained model
joblib.dump(model, "xgb_regressor_model.pkl")
