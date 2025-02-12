import numpy as np
import pandas as pd
import joblib

# Load the trained model and label encoder
model = joblib.load("xgb_regressor_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Define the new entry (Modify values as needed) secondary,0,2,0,0,1
new_data = {
    "highway_type": "secondary",  # Change this as needed
    "bridge": 0,
    "lane_count": 8,
    "layer_count": 2,
    "speed_limit": 80,
    "oneway": 0
}

# Convert to DataFrame
new_df = pd.DataFrame([new_data])

# Encode "highway_type" using the saved LabelEncoder
new_df["highway_type"] = encoder.transform(new_df["highway_type"])

# Ensure feature order matches training
feature_order = ["highway_type", "bridge", "lane_count", "layer_count", "speed_limit", "oneway"]
new_df = new_df[feature_order]

# Make prediction
prediction = model.predict(new_df)

# Display results
print(f"Predicted Safety: {int(prediction[0][0])}")
print(f"Predicted Priority Level: {int(prediction[0][1])}")
