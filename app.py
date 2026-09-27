import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("predictive_maintenance_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("Predictive Maintenance and Machine Failure Type Classification")

st.write(
    "Enter the machine operating conditions below to predict the failure type."
)

st.divider()

# Machine inputs
st.subheader("Machine Details")

machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temperature = st.number_input(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=300.0
)

process_temperature = st.number_input(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=310.0
)

rotational_speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=500,
    max_value=5000,
    value=1500
)

torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0,
    value=45.0
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=100
)

st.divider()

# Prediction button
if st.button("Predict Failure Type"):

    # Create input dataframe
    new_machine = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    # Encode Type using the same encoding used during training
    type_mapping = {
        "H": 0,
        "L": 1,
        "M": 2
    }

    new_machine["Type"] = new_machine["Type"].map(type_mapping)

    # Make prediction
    prediction = model.predict(new_machine)

    # Failure type full names
    failure_names = {
        "TWF": "Tool Wear Failure",
        "HDF": "Heat Dissipation Failure",
        "PWF": "Power Failure",
        "OSF": "Overstrain Failure",
        "RNF": "Random Failure",
        "No Failure": "No Machine Failure"
    }

    # Convert prediction to full name
    predicted_failure = failure_names[prediction[0]]

    # Display result
    st.subheader("Prediction Result")

    st.success(
        f"Predicted Failure Type: {predicted_failure}"
    )

    # Display prediction probabilities
    probabilities = model.predict_proba(new_machine)[0]

    probability_df = pd.DataFrame({
        "Failure Type": [
            failure_names[x] for x in model.classes_
        ],
        "Probability": probabilities * 100
    })

    probability_df["Probability"] = (
        probability_df["Probability"].round(2)
    )

    st.subheader("Prediction Probabilities")

    st.dataframe(
        probability_df,
        use_container_width=True
    )
