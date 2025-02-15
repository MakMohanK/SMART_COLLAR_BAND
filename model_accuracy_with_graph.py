import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Load dataset
database_path = "database/database.csv"
df = pd.read_csv(database_path)

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

# Train XGBoost Regressor with evaluation metric tracking
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, objective="reg:squarederror")

eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")

# Save the trained model
joblib.dump(model, "xgb_regressor_model.pkl")

# Function to plot learning curve
def plot_learning_curve(model):
    results = model.evals_result()  # Extract evaluation results
    epochs = len(results["validation_0"]["rmse"])  # Number of training iterations
    print(results)
    x_axis = range(0, epochs)

    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, results["validation_0"]["rmse"][::-1], label="Train RMSE", marker='o')
    plt.plot(x_axis, results["validation_1"]["rmse"][::-1], label="Test RMSE", marker='s')
    plt.xlabel("Iterations")
    plt.ylabel("RMSE")
    plt.title("XGBoost Learning Curve")
    plt.legend()
    plt.grid()
    plt.show()

# Call the function to visualize the learning curve
plot_learning_curve(model)
