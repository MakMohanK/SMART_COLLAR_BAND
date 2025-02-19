# import treelite
# # import treelite.runtime

# # Load the correctly saved XGBoost model
# model = treelite.frontend.load_xgboost_model_legacy_binary("xgb_model.json")  # OR "xgb_model.model"

# # Export as a shared library
# model.export_lib(toolchain="gcc", libpath="xgb_model.so", params="xgb_model.json", verbose=True)


# import xgboost as xgb
# import joblib

# # Load the joblib model
# model = joblib.load("xgb_regressor_model.pkl")

# # Save the model in XGBoost's native format
# model.save_model("xgb_model.json")



import json
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType
import xgboost as xgb

# Load the trained XGBoost model
model = xgb.Booster()
model.load_model("xgb_model.json")

# Load feature mapping
with open("feature_map.json", "r") as f:
    feature_map = json.load(f)

# Define input type using numerical feature names
num_features = len(feature_map)
initial_type = [("float_input", FloatTensorType([None, num_features]))]

# Convert XGBoost model to ONNX
onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

# Save ONNX model
with open("xgb_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model successfully converted to ONNX format!")

