
import requests
import csv
import numpy as np
import pandas as pd
import joblib

# Load the trained model and label encoder
model = joblib.load("xgb_regressor_model.pkl")
encoder = joblib.load("label_encoder.pkl")

csv_file_path = 'database2.csv'


new_data = {
    "highway_type": "residential",  # Change this as needed
    "bridge": 0,
    "lane_count": 0,
    "layer_count": 0,
    "speed_limit": 0,
    "oneway": 0
}

def append_csv(entry):
    with open(csv_file_path, mode='a', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(entry)

def get_location_info(latitude, longitude):
    url = f"https://overpass-api.de/api/interpreter?data=[out:json];way(around:10,{latitude},{longitude})[highway];out;"
    response = requests.get(url)
    print("highway_type", "\t","bridge", "\t","lane_count", "\t","layer_count", "\t","speed_limit","\t","oneway")
    if response.status_code == 200:
        data = response.json()['elements']
        for x in data:
            if x['type'] == 'way':
                try:
                    highway_type = x['tags']['highway']
                except:
                    highway_type = 0
                try:
                    bridge = x['tags']['bridge']
                except:
                    bridge = 0
                try:    
                    lane_count = x['tags']['lanes']
                except:
                    lane_count = 0
                try:
                    layer_count = x['tags']['layer']
                except:
                    layer_count = 0
                try:
                    speed_limit = x['tags']['maxspeed']
                except:
                    speed_limit = 0
                try:
                    oneway = x['tags']['oneway']
                    if oneway == 'yes':
                        oneway = 1
                    else:
                        oneway = 0
                except:
                    oneway = 0

            print(highway_type, "\t",bridge, "\t",lane_count, "\t",layer_count, "\t",speed_limit, "\t",oneway)
            entry = [highway_type, bridge, lane_count, layer_count, speed_limit, oneway]
            new_data["highway_type"] = highway_type
            new_data["bridge"] = int(bridge)
            new_data["lane_count"] = int(lane_count)
            new_data["layer_count"] = int(layer_count)
            new_data["speed_limit"] = int(speed_limit)
            new_data["oneway"] = int(oneway)
            append_csv(entry=entry)
            print(new_data)
            if new_data["highway_type"] != 0:
                prediction_on_data(new_data=new_data)
            else:
                print("not found highyway type...")
        return entry

    else:
        print("API Failed to generate Response.")


def prediction_on_data(new_data):
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


get_location_info(18.608260, 73.821062)


