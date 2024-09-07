import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime, timedelta
import pytz

# Load the data from CSV file
df = pd.read_csv('data/buss.csv')

# Function to convert time from HH.MM format to float representing hours
def time_to_float(time_str):
    try:
        # Ensure time_str is a float and contains the expected format
        if isinstance(time_str, (str, float)):
            parts = str(time_str).split('.')
            if len(parts) == 2:
                h, m = map(int, parts)
                return h + m / 60
    except ValueError:
        return np.nan  # Return NaN for invalid values

    return np.nan  # Return NaN if format is incorrect

# Apply the conversion to the 'time' column
df['time'] = df['time'].apply(time_to_float)

# Drop rows with NaN values in 'time'
df = df.dropna(subset=['time'])

# Check if the dataframe is not empty after conversion
if df.empty:
    print("DataFrame is empty after dropping NaN values. Check the data and time format.")
else:
    # Encode categorical data
    label_encoder_bus = LabelEncoder()
    df['bus_encoded'] = label_encoder_bus.fit_transform(df['bus'])

    label_encoder_routes = LabelEncoder()
    df['routes_encoded'] = label_encoder_routes.fit_transform(df['routes'])

    # Features and target variables
    X = df[['time']]
    y_bus = df['bus_encoded']
    y_routes = df['routes_encoded']

    # Split the data into training and testing sets
    X_train, X_test, y_bus_train, y_bus_test = train_test_split(X, y_bus, test_size=0.3, random_state=26)
    _, _, y_routes_train, y_routes_test = train_test_split(X, y_routes, test_size=0.3, random_state=26)

    # Initialize and train the Decision Tree Classifier for bus prediction
    bus_classifier = DecisionTreeClassifier()
    bus_classifier.fit(X_train, y_bus_train)

    # Initialize and train the Decision Tree Classifier for routes prediction
    routes_classifier = DecisionTreeClassifier()
    routes_classifier.fit(X_train, y_routes_train)

    # Make predictions
    bus_predictions = bus_classifier.predict(X_test)
    routes_predictions = routes_classifier.predict(X_test)

    bus_accuracy = accuracy_score(y_bus_test, bus_predictions)
    routes_accuracy = accuracy_score(y_routes_test, routes_predictions)

    print("Bus Prediction Accuracy:", bus_accuracy)
    print("Routes Prediction Accuracy:", routes_accuracy)

    # Example prediction
    sample_time = [[15.3]]  # Example time
    predicted_bus_encoded = bus_classifier.predict(sample_time)
    predicted_routes_encoded = routes_classifier.predict(sample_time)

    # Decode the predictions
    predicted_bus = label_encoder_bus.inverse_transform(predicted_bus_encoded)
    predicted_routes = label_encoder_routes.inverse_transform(predicted_routes_encoded)

    print(f"Prediction for time {sample_time[0][0]}: Bus = {predicted_bus[0]}, Routes = {predicted_routes[0]}")

    ############################################################################################################
    # Get the current time in Indian Standard Time (IST)
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    current_time_float = current_time.hour + current_time.minute / 60

    print(f"Current IST Time: {current_time.strftime('%H:%M')}")

    # Find the current bus and route based on the current time
    current_buses_routes = df[df['time'] == current_time_float]
    upcoming_buses_routes = df[df['time'] > current_time_float].sort_values('time')

    # Remove duplicates
    current_buses_routes = current_buses_routes.drop_duplicates(subset=['bus', 'routes', 'time'])
    upcoming_buses_routes = upcoming_buses_routes.drop_duplicates(subset=['bus', 'routes', 'time'])

    if not current_buses_routes.empty:
        print("Current Bus and Route:")
        for _, row in current_buses_routes.iterrows():
            print(f"Bus: {row['bus']}, Route: {row['routes']}, Time: {row['time']}")
    else:
        print("No current bus and route found at this time.")

    if not upcoming_buses_routes.empty:
        print("\nUpcoming Buses and Routes:")
        for _, row in upcoming_buses_routes.iterrows():
            print(f"Bus: {row['bus']}, Route: {row['routes']}, Time: {row['time']:.2f}")
    else:
        print("No upcoming buses and routes.")
