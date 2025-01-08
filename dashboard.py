import streamlit as st
import base64
import os
import streamlit as st
import numpy as np
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime, timedelta
from streamlit_echarts import st_echarts
from keras.models import model_from_json 
import random
import pulp
import matplotlib.pyplot as plt

def SOH_predict():
    # Seed for reproducibility
    np.random.seed(42)

    # Generate random values for each column (using reasonable ranges based on the sample data)
    n_samples = 11  # Increased sample size for a larger dataset

    # Generate random data within reasonable ranges for each column
    data = {
        'terminal_voltage': np.random.uniform(2.5, 4.5, n_samples),
        'terminal_current': np.random.uniform(-1, 1, n_samples),
        'temperature': np.random.uniform(5, 10, n_samples),
        'charge_current': np.random.uniform(0, 1, n_samples),
        'charge_voltage': np.random.uniform(0, 4.5, n_samples),
        'time': np.random.uniform(0, 4500, n_samples),
        'capacity': np.random.uniform(1.2, 1.6, n_samples),
        'cycle': np.random.randint(1, 75, n_samples)
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Select numeric columns to scale
    numeric_columns = ['terminal_voltage', 'terminal_current', 'temperature', 'charge_current', 
                    'charge_voltage', 'time', 'capacity', 'cycle']

    input_data = df[numeric_columns].values  # Shape: (100, 8)
    input_data_reshaped = input_data.reshape((input_data.shape[0], 8, 1))  # Shape: (100, 8, 1)

    def create_dataset(dataset, look_back=1):
        dataX = []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), :]
            dataX.append(a)
        return np.array(dataX)

    # Generate dataset using the look_back parameter
    look_back = 1
    testX = create_dataset(input_data, look_back)

    # Reshape testX to match the input shape expected by the model
    testX = testX.reshape((testX.shape[0], testX.shape[1], testX.shape[2]))  # Shape: (samples, look_back, features)
    # Load trained model

    json_file = open(r"B48_model.json", "r")
    loaded_model_json = json_file.read() 
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # model weight load 
    loaded_model.load_weights(r"B48.weights.h5")

    # Predict on each sample individually
    predicted_SOH_individual = []
    for i in range(testX.shape[0]):
        prediction = loaded_model.predict(testX[i].reshape(look_back, 8))
        predicted_SOH_individual.append(prediction[0][0])

    return(predicted_SOH_individual)

#-----------------------------------------------------------------------------------------------------------------------------------------

# Function to generate data in shape (1, 10, 34)
def make_predictions_power():
    

    # Data generation remains unchanged
    n_rows = 10  # Total rows to generate for prediction
    data = {
        'V12': np.random.uniform(380, 400, n_rows),
        'V23': np.random.uniform(370, 380, n_rows),
        'V31': np.random.uniform(360, 370, n_rows),
        'V1': np.random.uniform(215, 220, n_rows),
        'V2': np.random.uniform(220, 230, n_rows),
        'V3': np.random.uniform(205, 210, n_rows),
        'A1': np.random.uniform(45, 55, n_rows),
        'A2': np.random.uniform(60, 80, n_rows),
        'A3': np.random.uniform(35, 45, n_rows),
        'P1': np.random.uniform(9, 12, n_rows),
        'P2': np.random.uniform(14, 18, n_rows),
        'P3': np.random.uniform(7, 9, n_rows),
        'S1': np.random.uniform(9, 10, n_rows),
        'S2': np.random.uniform(16, 18, n_rows),
        'S3': np.random.uniform(8, 9, n_rows),
        'S(SUM)': np.random.uniform(35, 40, n_rows),
        'Q1': np.random.uniform(2, 3, n_rows),
        'Q2': np.random.uniform(2, 3, n_rows),
        'Q3': np.random.uniform(1, 2, n_rows),
        'Q(SUM)': np.random.uniform(5, 7, n_rows),
        'PF1': np.random.uniform(0.97, 1, n_rows),
        'PF2': np.random.uniform(0.97, 1, n_rows),
        'PF3': np.random.uniform(0.97, 1, n_rows),
        'PF(SUM)': np.random.uniform(0.97, 1, n_rows),
        'PFH': np.random.uniform(0.98, 0.99, n_rows),
        'PHASE1': np.random.uniform(10, 15, n_rows),
        'PHASE2': np.random.uniform(7, 8, n_rows),
        'PHASE3': np.random.uniform(10, 12, n_rows),
        'WH': np.random.uniform(0.003, 0.012, n_rows),
        'SH': np.random.uniform(0.003, 0.01, n_rows),
        'QH': np.random.uniform(0.0005, 0.002, n_rows),
        'FREQ': np.random.uniform(49, 50, n_rows),
        'Season': np.random.choice([1], size=n_rows),
        'P(SUM)': np.random.uniform(34, 35.5, n_rows)
    }

    # Generate 10 evenly spaced timestamps for the current date, including the current time
    current_time = datetime.now()  # Current system time
    time_diff = timedelta(hours=2)  # Time interval between each timestamp

    # Generate 10 timestamps from 9 hours ago to current time
    timestamps = [
        (current_time - timedelta(hours=2 * i)).replace(microsecond=0)  # Remove microseconds
        for i in range(n_rows)
    ][::-1]  # Backward 2-hour intervals

    # Create DataFrame and set the generated timestamps as index
    df = pd.DataFrame(data)
    df['Datetime'] = timestamps
    df = df.set_index('Datetime')
    # Scaling the features (including 'Season')
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = df.copy()
    df_scaled.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    data_scaled = df_scaled.values

    data_scaled_reshaped = data_scaled.reshape(1, 10, 34)
    # Load the trained model
    with open(r"power_consumption_model_new.pkl", 'rb') as file:
        model = pickle.load(file)

    last_sequence = data_scaled_reshaped[-1]  # Last sequence from the test set
    future_predictions = model.predict(last_sequence.reshape(1, 10, data_scaled_reshaped.shape[2]))  # Predict the next 10 values
    future_predictions = future_predictions.flatten()
    # Generate future timestamps (next 10 timestamps, 2-hour intervals)
    future_timestamps = [
        (current_time + timedelta(hours=2 * i)).replace(microsecond=0)  # Forward 2-hour intervals
        for i in range(1, 11)  # Generating 10 future timestamps
    ]
    return future_timestamps, future_predictions


# --------------------------------------------------------------------------------------------------------------------



st.sidebar.title("Configuration")

# Dropdown 1: Choose preferred option
preferred_api = st.sidebar.selectbox(
    "Choose your preferred Option:",
    ["Home", "Power Consumption", "Battery Storage Degradation"]
)
col1, col2 = st.columns([9, 2])  # Adjust column widths as needed

with col2:  # Place the button in the right column
    if st.button("Refresh"):
        # Refresh the page
        st.experimental_rerun()

# Dropdown 2: Conditional options based on preferred_api
if preferred_api == "Home":
    example_chart = st.sidebar.selectbox(
        "Choose a Plot", 
        ["---- Select an Option First ----"], 
        index=0
    )

    future_timestamps,predictions = make_predictions_power()
    predictions_SOH = SOH_predict()
    predictions_SOH = sorted([round(float(value) / 10, 2)*100 for value in predictions_SOH],reverse = True)

    # Parameters
    battery_capacity = 100  # Battery capacity in kWh
    battery_efficiency = 0.9  # Battery discharge efficiency
    charging_efficiency = 0.9  # Battery charging efficiency
    ac_power = 200  # AC power in kW

    # Example prediction functions
    def predict_power_consumption(index,hour):
        if 8 <= hour < 17:  # Educational building hours
            return (predictions[index-1]*5)
        elif 17 <= hour or hour < 8:  # Hostel active hours
            return (predictions * 5 - 30 * 5)
        else:
            return 0

    def predict_soh(index):
        """Returns predicted SOH for the index (1 to 10)."""
        return predictions_SOH[index-1]  # Use the prediction corresponding to the index

    # Optimization model
    optimizer = pulp.LpProblem("Energy_Management_Optimizer", pulp.LpMinimize)

    # Variables
    grid_power = {i: pulp.LpVariable(f"Grid_Power_{i}", lowBound=0) for i in range(1, 11)}
    battery_power = {i: pulp.LpVariable(f"Battery_Power_{i}", lowBound=-50, upBound=50) for i in range(1, 11)}
    battery_discharge = {i: pulp.LpVariable(f"Battery_Discharge_{i}", lowBound=0) for i in range(1, 11)}
    ac_state = {i: pulp.LpVariable(f"AC_State_{i}", cat="Binary") for i in range(1, 11)}

    # Objective: Minimize grid power usage
    optimizer += pulp.lpSum(grid_power[i] for i in range(1, 11)), "Minimize_Grid_Usage"

    # Constraints
    for i, timestamp in enumerate(future_timestamps, start=1):
        # Determine building type based on the hour
        hour = timestamp.hour
        if 8 <= hour < 17:  # Educational building hours
            building_type = "educational"
        elif 17 <= hour or hour < 8:  # Hostel active hours
            building_type = "hostel"
        else:
            building_type = "shutdown"
        # Get predictions
        soh = predict_soh(i)
        total_load = predict_power_consumption(i,hour)

        # Adjust battery capacity based on SOH
        adjusted_battery_capacity = battery_capacity * (soh / 100)

        # Energy balance constraint
        optimizer += grid_power[i] + battery_power[i] == total_load, f"Energy_Balance_{i}"

        # Additional constraints based on building type
        if building_type == "educational":
            # Disable battery during educational hours
            optimizer += battery_power[i] == 0, f"Disable_Battery_{i}"
        elif building_type == "hostel":
            # Battery SOC dynamics during hostel hours
            optimizer += battery_power[i] == battery_discharge[i] * battery_efficiency, f"Battery_Discharge_Relation_{i}"
            # Limit battery capacity based on SOH
            optimizer += battery_power[i] <= adjusted_battery_capacity, f"Battery_Capacity_Limit_{i}"

        # Limit AC operation: 1-hour on, 1-hour off
        if i > 1:  # After the first time slot
            optimizer += ac_state[i] + ac_state[i - 1] == 1, f"AC_Alternating_{i}"

    # Solve the optimization problem
    optimizer.solve()

    time_slots = []
    building_types = []
    grid_powers = []
    battery_powers = []
    battery_soc = []
    ac_states = []

    # Assuming `optimizer.status` and `future_timestamps` are already defined
    print("Optimization Status:", pulp.LpStatus[optimizer.status])

    if pulp.LpStatus[optimizer.status] == "Optimal":
        print("\nResults:")
        for i, timestamp in enumerate(future_timestamps, start=1):
            # Determine building type based on the hour
            hour = timestamp.hour
            if 8 <= hour < 17:  # Educational building hours
                building_type = "educational"
            elif 17 <= hour or hour < 8:  # Hostel active hours
                building_type = "hostel"
            else:
                building_type = "shutdown"

            # Store values in respective lists
            formatted_time = timestamp.strftime("%H:%M")
            time_slots.append(formatted_time)
            building_types.append(building_type.capitalize())
            grid_powers.append(grid_power[i].varValue)
            battery_powers.append(battery_power[i].varValue)
            
            if building_type == 'hostel':
                battery_soc_value = predict_soh(i)
                battery_soc.append(battery_soc_value)
                adjusted_capacity = battery_capacity * (battery_soc_value / 100)
            else:
                battery_soc.append(100)
                adjusted_capacity = battery_capacity  # Assume full capacity for educational buildings

            print("\n")
            print(f"Time Slot: {timestamp}")
            print(f"  Active Building: {building_type.capitalize()}")
            print(f"  Grid Power: {grid_power[i].varValue:.2f} kW")
            print(f"  Battery Power: {battery_power[i].varValue:.2f} kW")
            print(f"  Battery SOC: {battery_soc[-1]:.2f}% (Adjusted Capacity: {adjusted_capacity:.2f} kWh)")
            
            if 8 <= timestamp.hour < 17:
                ac_state_value = ac_state[i].varValue
                ac_states.append(ac_state_value)
                print(f"  AC State: {ac_state_value:.1f}")
            else:
                ac_states.append(None)  # No AC state for shutdown hours

    else:
        print("Optimization failed.")

    #----------------------------------------------------------home-----------------------------------------------------------------

    price_per_kwh = 35
    # Initialize variables to store total energy saved and cost saved
    total_energy_saved = 0
    total_cost_saved = 0

    # Calculate energy and cost saved
    for i, timestamp in enumerate(future_timestamps, start=1):
        total_load = predict_power_consumption(i, timestamp.hour)
        grid_power_usage = grid_power[i].varValue  # Grid power used after optimization
        
        # Energy saved for the current time slot
        energy_saved = abs(total_load - grid_power_usage)
        total_energy_saved += energy_saved
        
        # Cost saved for the current time slot
        cost_saved = energy_saved * price_per_kwh
        total_cost_saved += cost_saved

    # Results
    total_energy = sum(total_energy_saved) / len(total_energy_saved)
    total_cost = sum(total_cost_saved) / len(total_cost_saved)

    # print(f"Total Energy Saved: {total_energy:.2f} kWh")
    # print(f"Total Cost Saved: {total_cost:.2f} PKR")

    # Set the title of the dashboard
    st.title("Energy Management System Dashboard")

    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)

    # Initialize total power demand list
    total_power_demand = []

    # Calculate total power demand for each time slot
    for i, timestamp in enumerate(future_timestamps, start=1):
        # Total power demand is the sum of grid power and absolute value of battery power
        total_demand = grid_power[i].varValue + abs(battery_power[i].varValue)
        total_power_demand.append(total_demand)

    # Calculate the average total power demand
    average_total_demand = round(sum(total_power_demand) / len(total_power_demand), 2)
    # Total Demand
    col1.metric("Total Demand", average_total_demand,"KW")
    
    net_demand = []

    # Calculate net demand for each time slot
    for i, timestamp in enumerate(future_timestamps, start=1):
        # Net demand is grid power minus battery power
        net = grid_power[i].varValue - battery_power[i].varValue
        net_demand.append(round(net, 2))  # Round to 2 decimal places

    # Calculate the average net demand
    average_net_demand = round(sum(net_demand) / len(net_demand), 2)

    # Net Demand
    col2.metric("Net Demand", average_net_demand,"KW")

    grid_power_values = []

    # Calculate net demand for each time slot and store grid power values
    for i, timestamp in enumerate(future_timestamps, start=1):
        # Net demand is grid power minus battery power
        net = grid_power[i].varValue - battery_power[i].varValue
        net_demand.append(round(net, 2))  # Round to 2 decimal places

        # Store the grid power value
        grid_power_values.append(round(grid_power[i].varValue, 2))

    # Calculate the average power production (grid power)
    average_power_production = round(sum(grid_power_values) / len(grid_power_values), 2)

    # GB Production
    col3.metric("Power Production", average_power_production,"KW")

    # Pumped Storage
    if battery_soc[9] < 100:
        col4.metric("Total Battery SOH", battery_soc[9],'-8%')
    else:
        col4.metric("Total Battery SOH", battery_soc[9],'10%')
    
        # Specify the path to your video file
    video_path = "test.mp4"  # Update this to your video file path

    if os.path.exists(video_path):
        # Read the file as bytes
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()

        # Convert bytes to base64
        video_base64 = base64.b64encode(video_bytes).decode('utf-8')

        # HTML to embed the video
        video_html = f"""
        <div style="position: relative; display: inline-block; width: fit-content;">

        <!-- Video -->
        <video autoplay muted loop style="max-width: 100%; height: auto;">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>

        <!-- Circle overlay -->
        <div style="
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-42%, -54%);
            width: 65px;
            height: 65px;
            border-radius: 50%;
            background: rgba(29, 41, 59); /* Optional semi-transparent background */
            pointer-events: none; /* Ensures the circle does not interfere with video interactions */
        "></div>
        <div style="
            position: absolute;
            top: 25%;
            left: 23%;
            transform: translate(-40%, -58%);
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: rgba(29, 41, 59); /* Optional semi-transparent background */
            pointer-events: none; /* Ensures the circle does not interfere with video interactions */
        "></div>
        <div style="
            position: absolute;
            top: 76%;
            left: 23%;
            transform: translate(-41%, -60%);
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: rgba(29, 41, 59); /* Optional semi-transparent background */
            pointer-events: none; /* Ensures the circle does not interfere with video interactions */
        "></div>
        <div style="
            position: absolute;
            top: 76%;
            left: 76%;
            transform: translate(-38%, -60%);
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: rgba(29, 41, 59); /* Optional semi-transparent background */
            pointer-events: none; /* Ensures the circle does not interfere with video interactions */
        "></div>
        <div style="
            position: absolute;
            top: 25%;
            left: 76%;
            transform: translate(-31%, -60%);
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: rgba(29, 41, 59); /* Optional semi-transparent background */
            pointer-events: none; /* Ensures the circle does not interfere with video interactions */
        "></div>
        </div>
        <div style="
            position: absolute;
            top: 79%;
            left: 50%;
            transform: translate(-42%, -54%);
            width: 65px;
            height: 65px;
            border-radius: 50%;
            background: rgba(29, 41, 59); /* Optional semi-transparent background */
            pointer-events: none; /* Ensures the circle does not interfere with video interactions */
        "></div>
        <div style="
            position: absolute;
            top: 53%;
            left: 7%;
            transform: translate(-38%, -60%);
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: rgba(29, 41, 59); /* Optional semi-transparent background */
            pointer-events: none; /* Ensures the circle does not interfere with video interactions */
        "></div>
        <div style="
            position: absolute;
            top: 53%;
            left: 92%;
            transform: translate(-31%, -60%);
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: rgba(29, 41, 59); /* Optional semi-transparent background */
            pointer-events: none; /* Ensures the circle does not interfere with video interactions */
        "></div>
        <div style="
            position: absolute;
            top: 25%;
            left: 23%;
            transform: translate(-38%, -50%);
            font-size: 13px;
            color: white;
            font-weight: normal;
            text-shadow: 0px 0px 0px rgba(0, 0, 0, 0.7);
            pointer-events: none;">
            {average_net_demand} KW
        </div>
        <div style="
            position: absolute;
            top: 78%;
            left: 23%;
            transform: translate(-30%, -45%);
            font-size: 13px;
            color: white;
            font-weight: normal;
            text-shadow: 0px 0px 0px rgba(0, 0, 0, 0.7);
            pointer-events: none;">
            {battery_soc[9]} %
        </div>
        <div style="
            position: absolute;
            top: 78%;
            left: 76%;
            transform: translate(-35%, -45%);
            font-size: 13px;
            color: white;
            font-weight: normal;
            text-shadow: 0px 0px 0px rgba(0, 0, 0, 0.7);
            pointer-events: none;">
            {average_total_demand} KW
        </div>
        <div style="
            position: absolute;
            top: 25%;
            left: 76%;
            transform: translate(-28%, -50%);
            font-size: 13px;
            color: white;
            font-weight: normal;
            text-shadow: 0px 0px 0px rgba(0, 0, 0, 0.7);
            pointer-events: none;">
            {average_power_production} KW
        </div>
        <div style="
            position: absolute;
            top: 50%;
            left: 49%;
            transform: translate(-28%, -35%);
            font-size: 15px;
            color: yellow;
            font-weight: normal;
            text-shadow: 0px 0px 0px rgba(0, 0, 0, 0.7);
            pointer-events: none;">
            {total_cost:.2f}
        </div>
        <div style="
            position: absolute;
            top: 76%;
            left: 49%;
            transform: translate(-22%, -35%);
            font-size: 13px;
            color: white;
            font-weight: normal;
            text-shadow: 0px 0px 0px rgba(0, 0, 0, 0.7);
            pointer-events: none;">
            {total_energy:.2f}
        </div>
        <div style="
            position: absolute;
            top: 51%;
            left: 92%;
            transform: translate(-25%, -40%);
            font-size: 13px;
            color: white;
            font-weight: normal;
            text-shadow: 0px 0px 0px rgba(0, 0, 0, 0.7);
            pointer-events: none;">
            OUTPUT
        </div>
        <div style="
            position: absolute;
            top: 51%;
            left: 6%;
            transform: translate(-15%, -40%);
            font-size: 13px;
            color: white;
            font-weight: normal;
            text-shadow: 0px 0px 0px rgba(0, 0, 0, 0.7);
            pointer-events: none;">
            INPUT
        </div>
        <div style="
            position: absolute;
            top: 54%;
            left: 49%;
            transform: translate(-22%, -20%);
            font-size: 10px;
            color: green;
            font-weight: normal;
            text-shadow: 0px 0px 0px rgba(0, 0, 0, 0.7);
            pointer-events: none;">
            PKR SAVED
        </div>
        <div style="
            position: absolute;
            top: 79%;
            left: 49%;
            transform: translate(-25%, -20%);
            font-size: 10px;
            color: green;
            font-weight: normal;
            text-shadow: 0px 0px 0px rgba(0, 0, 0, 0.7);
            pointer-events: none;">
            KWH ENERGY
        </div>
        <div style="
            position: absolute;
            top: 81%;
            left: 49%;
            transform: translate(-10%, -20%);
            font-size: 10px;
            color: green;
            font-weight: normal;
            text-shadow: 0px 0px 0px rgba(0, 0, 0, 0.7);
            pointer-events: none;">
            SAVED
        </div>
        <!-- text overlay -->
        <div style="
            position: absolute;
            top: 65%;
            left: 92%;
            transform: translate(-35%, -60%);
            width: 80px;
            height: 20px;
            border-radius: 0px;
            background: rgba(29, 41, 59); /* Optional semi-transparent background */
            pointer-events: none; /* Ensures the circle does not interfere with video interactions */
        "></div>
        <div style="
            position: absolute;
            top: 65%;
            left: 7%;
            transform: translate(-35%, -60%);
            width: 80px;
            height: 20px;
            border-radius: 0px;
            background: rgba(29, 41, 59); /* Optional semi-transparent background */
            pointer-events: none; /* Ensures the circle does not interfere with video interactions */
        "></div>
        <div style="
            position: absolute;
            top: 92%;
            left: 22%;
            transform: translate(-35%, -60%);
            width: 80px;
            height: 20px;
            border-radius: 0px;
            background: rgba(29, 41, 59); /* Optional semi-transparent background */
            pointer-events: none; /* Ensures the circle does not interfere with video interactions */
        "></div>
        <div style="
            position: absolute;
            top: 92%;
            left: 49%;
            transform: translate(-35%, -60%);
            width: 80px;
            height: 20px;
            border-radius: 0px;
            background: rgba(29, 41, 59); /* Optional semi-transparent background */
            pointer-events: none; /* Ensures the circle does not interfere with video interactions */
        "></div>
        <div style="
            position: absolute;
            top: 92%;
            left: 76%;
            transform: translate(-35%, -60%);
            width: 80px;
            height: 20px;
            border-radius: 0px;
            background: rgba(29, 41, 59); /* Optional semi-transparent background */
            pointer-events: none; /* Ensures the circle does not interfere with video interactions */
        "></div>
        <div style="
            position: absolute;
            top: 12%;
            left: 76%;
            transform: translate(-35%, -60%);
            width: 100px;
            height: 20px;
            border-radius: 0px;
            background: rgba(29, 41, 59); /* Optional semi-transparent background */
            pointer-events: none; /* Ensures the circle does not interfere with video interactions */
        "></div>
        <div style="
            position: absolute;
            top: 12%;
            left: 22%;
            transform: translate(-35%, -60%);
            width: 100px;
            height: 20px;
            border-radius: 0px;
            background: rgba(29, 41, 59); /* Optional semi-transparent background */
            pointer-events: none; /* Ensures the circle does not interfere with video interactions */
        "></div>
        <div style="
            position: absolute;
            top: 21%;
            left: 49%;
            transform: translate(-35%, -60%);
            width: 82px;
            height: 125px;
            border-radius: 0px;
            background: rgba(29, 41, 59); /* Optional semi-transparent background */
            pointer-events: none; /* Ensures the circle does not interfere with video interactions */
        "></div>
        """

        # Display the video using HTML
        st.markdown(video_html, unsafe_allow_html=True)
    else:
        st.error("Video file not found. Please check the path.")

    #-----------------------------------------------------------------------------------------------------------------------------
    st.subheader('')
    st.subheader("Grid Power")
    options = {
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": time_slots,
        },
        "yAxis": {"type": "value"},
        "series": [
            {
                "data": grid_powers,
                "type": "line",
                "areaStyle": {},
            }
        ],
        "backgroundColor": "rgba(14, 17, 23)",
    }
    st_echarts(options=options)

    st.subheader('')
    st.subheader("AC States with respect to Time")

    # Pre-process the ac_states to map 1 to 'ON' and 0 to 'OFF'
    ac_labels = ["ON" if state == 1 else "OFF" for state in ac_states]
    
    options = {
        "xAxis": {
            "type": "category",
            "data": time_slots,  # X-axis data
        },
        "yAxis": {"type": "value"},
        "series": [
            {
                "data": ac_states,  # Bar data
                "type": "bar",
                "itemStyle": {
                    "color": "#a90000",  # Bar color set to red
                },
                "label": {
                    "show": True,  # Enable labels
                    "position": "top",  # Display labels above the bars
                    "formatter": ac_labels,  # Use pre-processed labels here
                    "color": "white",  # Label color
                },
            }
        ],
        "backgroundColor": "rgba(14, 17, 23)",  # Chart background color
    }

    st_echarts(
        options=options,
        height="400px",
    )





    #-----------------------------------------------------------------------------------------------------------------------------

elif preferred_api == "Power Consumption":

    @st.cache_data
    def generate_predictions():
        # Run make_predictions_power() once for all buildings
        timestamps, building1 = make_predictions_power()
        _, building2 = make_predictions_power()
        _, building3 = make_predictions_power()
        _, building4 = make_predictions_power()
        _, building5 = make_predictions_power()
        
        # Convert to lists to reuse the data
        building1 = building1.tolist()
        building2 = building2.tolist()
        building3 = building3.tolist()
        building4 = building4.tolist()
        building5 = building5.tolist()

        # Format timestamps as strings
        timestamps = [timestamp.strftime('%Y-%m-%d %H:%M:%S') for timestamp in timestamps]

        return timestamps, building1, building2, building3, building4, building5

    # Generate and cache the predictions
    timestamps, building1, building2, building3, building4, building5 = generate_predictions()
    example_chart = st.sidebar.selectbox(
        "Choose a Plot",
        ["Line Chart", "Bar Chart", "Area Chart", "Pie Chart"]
    )

    if example_chart == "Line Chart":
        st.title("Power Consumption Predictions")
        st.title("")
        # Set the title of the dashboard
        # Generate data for the line chart (this could be predictions from your model)
        options = {
            "title": {"text": ""},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["Building 1", "Building 2", "Building 3", "Building 4", "Building 5"]},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "toolbox": {"feature": {"saveAsImage": {}}},
            "xAxis": {
                "type": "category",
                "boundaryGap": False,
                "data": timestamps,
            },
            "yAxis": {
                "type": "value",
                "min": 34,
                "max": 37,
            },
            "series": [
                {"name": "Building 1", "type": "line", "data": building1},
                {"name": "Building 2", "type": "line", "data": building2},
                {"name": "Building 3", "type": "line", "data": building3},
                {"name": "Building 4", "type": "line", "data": building4},
                {"name": "Building 5", "type": "line", "data": building5},
            ],
            "backgroundColor": "rgba(14, 17, 23)",
        }

        # Display the chart
        st_echarts(options=options, height="400px")

    if example_chart == "Bar Chart":
        st.title("Power Consumption Predictions")
        st.title("")
        data = {
            "Building1": building1,
            "Building2": building2,
            "Building3": building3,
            "Building4": building4,
            "Building5": building5,
        }

        # Dropdown for selecting the building
        selected_building = st.selectbox("Select Building", list(data.keys()))

        # Get data for the selected building
        selected_data = data[selected_building]

        # Chart options
        options = {
            "xAxis": {
                "type": "category",
                "data": timestamps,
            },
            "yAxis": {
                "type": "value",
                "min": 30,
                "max": 36,
            },
            "series": [
                {
                    "data": [
                        {"value": value, "itemStyle": {"color": "#a90000"}} if value > 150 else value
                        for value in selected_data
                    ],
                    "type": "bar",
                }
            ],
        }

        # Render the chart
        st_echarts(
            options=options,
            height="400px",
        )

    if example_chart == "Pie Chart":
        st.title("")
        options = {
            "title": {
                "text": "Power Consumption of Buildings",
                "left": "center"
            },
            "tooltip": {"trigger": "item"},
            "legend": {
                "orient": "vertical",
                "left": "left",
            },
            "series": [
                {
                    "name": "Access Sources",
                    "type": "pie",
                    "radius": "50%",
                    "data": [
                        {"value": building1, "name": "Building 1"},
                        {"value": building2, "name": "Building 2"},
                        {"value": building3, "name": "Building 3"},
                        {"value": building4, "name": "Building 4"},
                        {"value": building4, "name": "Building 5"},
                    ],
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 0,
                            "shadowColor": "rgba(0, 0, 0, 0.5)",
                        }
                    },
                    "label": {
                    "fontSize": 20,  # Increase the label size (you can adjust this value)
                    "fontWeight": "bold",  # Optional: Add boldness to the label text
                    "color": "#FFFFFF",  # Optional: Change label text color
                    },
                }
            ],
            "backgroundColor": "rgba(14, 17, 23)",
        }

        # Render the chart
        st_echarts(options=options, height="600px",)
    
    if example_chart == "Area Chart":
        st.title("Power Consumption Predictions")
        st.title("")
        options = {
            "title": {"text": ""},
            "tooltip": {"trigger": "axis","axisPointer": {"type": "cross", "label": {"backgroundColor": "#6a7985"}},},
            "legend": {"data": ["Building 1", "Building 2", "Building 3", "Building 4", "Building 5"]},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "toolbox": {"feature": {"saveAsImage": {}}},
            "xAxis": {"type": "category","boundaryGap": False,"data": timestamps},
            "yAxis": {"type": "value"},
            "series": [
                {"name": "Building 1", "type": "line", "stack": "Total", "areaStyle": {}, "data": building1},
                {"name": "Building 2", "type": "line", "stack": "Total", "areaStyle": {}, "data": building2},
                {"name": "Building 3", "type": "line", "stack": "Total", "areaStyle": {}, "data": building3},
                {"name": "Building 4", "type": "line", "stack": "Total", "areaStyle": {}, "data": building4},
                {"name": "Building 5", "type": "line", "stack": "Total", "areaStyle": {}, "data": building5},
            ],
            "backgroundColor": "rgba(14, 17, 23)",
        }
        st_echarts(options=options, height="400px")


elif preferred_api == "Battery Storage Degradation":
    
    example_chart = st.sidebar.selectbox(
        "Choose a Plot",
        ["Liquid Fill", "Guage Ring", "Line Chart"]
    )

    if example_chart == "Liquid Fill":
        results = SOH_predict()

        # Convert results to float, divide by 10, round to 2 decimal places, and sort the values
        results = sorted([round(float(value) / 10, 2) for value in results], reverse=True)

        st.title("Battery State of Charge (SOC)")
        liquidfill_option = {
            "series": [{"type": "liquidFill", "data": results}],
            "backgroundColor": "rgba(14, 17, 23)",
        }

        st_echarts(liquidfill_option)

    if example_chart == "Guage Ring":
        st.title("Battery State of Charge (SOC)")
        option = {
        "series": [
                {
                    "type": "gauge",
                    "startAngle": 90,
                    "endAngle": -270,
                    "pointer": {"show": False},
                    "progress": {
                        "show": True,
                        "overlap": False,
                        "roundCap": True,
                        "clip": False,
                        "itemStyle": {"borderWidth": 1, "borderColor": "#464646"},
                    },
                    "axisLine": {"lineStyle": {"width": 40}},
                    "splitLine": {"show": False, "distance": 0, "length": 10},
                    "axisTick": {"show": False},
                    "axisLabel": {"show": False, "distance": 50},
                    "data": [
                        {
                            "value": random.randint(1, 99),
                            "name": "Perfect",
                            "title": {"offsetCenter": ["0%", "-30%"]},
                            "detail": {"offsetCenter": ["0%", "-20%"]},
                        },
                        {
                            "value": random.randint(1, 99),
                            "name": "Good",
                            "title": {"offsetCenter": ["0%", "0%"]},
                            "detail": {"offsetCenter": ["0%", "10%"]},
                        },
                        {
                            "value": random.randint(1, 99),
                            "name": "Commonly",
                            "title": {"offsetCenter": ["0%", "30%"]},
                            "detail": {"offsetCenter": ["0%", "40%"]},
                        },
                    ],
                    "title": {"fontSize": 14},
                    "detail": {
                        "width": 50,
                        "height": 14,
                        "fontSize": 14,
                        "color": "auto",
                        "borderColor": "auto",
                        "borderRadius": 20,
                        "borderWidth": 1,
                        "formatter": "{value}%",
                    },
                }
            ],
            "backgroundColor": "rgba(14, 17, 23)",
        }

        st_echarts(option, height="500px", key="echarts")

    if example_chart == "Line Chart":
        st.title("Battery State of Charge (SOC) Prediction")
        results = SOH_predict()
        results = sorted([round(float(value) / 10, 2) for value in results], reverse=True)
        current_time = datetime.now()
        future_timestamps = [
        (current_time + timedelta(hours=2 * i)).replace(microsecond=0)  # Forward 2-hour intervals
        for i in range(1, 11)  # Generating 10 future timestamps
        ]
        timestamps = [timestamp.strftime('%H:%M') for timestamp in future_timestamps]
        option = {
            "xAxis": {
                "type": "category",
                "data": timestamps,
            },
            "yAxis": {"type": "value"},
            "series": [{"data": results, "type": "line"}],
            "backgroundColor": "rgba(14, 17, 23)",
        }
        st_echarts(
            options=option, height="400px",
        )

# Informative Text with Link
st.sidebar.markdown(
    """
    *This* **Energy Management System** *project aims to help businesses or homes use energy more efficiently. 
    It monitors how energy is used, predicts future energy needs, and suggests ways to save energy. 
    By tracking energy consumption and making smart decisions, the system helps reduce waste, lower costs, 
    and make sure energy is used in the best way possible. The goal is to improve sustainability and reduce overall energy expenses.*
    """
)



