import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prediction import predict

# Title and description of the app
# Streamlit app setup
st.title('Mud Properties Prediction App')
st.write("""
This app predicts various mud properties using the Random Forest Regressor. 
Enter the inputs, and the model will predict outputs like FANN 600 rpm and FANN 300 rpm.
Additional calculations for Plastic Viscosity (PV), Yield Point (YP), and more will also be provided.
""")

# Function to collect user input
def user_input_features():
    Mud_Wt_In_ppg = st.sidebar.number_input('Mud Wt. In (ppg)', min_value=8.0, max_value=20.0, value=10.0, step=0.1)
    Funnel_Viscosity = st.sidebar.number_input('Funnel Viscosity ', min_value=25.0, max_value=80.0, value=35.0, step=0.1)
    
    data = {
        'Mud Wt. In (ppg)': Mud_Wt_In_ppg,
        'Funnel Viscosity ': Funnel_Viscosity
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Store user input in a dataframe
input_df = user_input_features()

# Load the scaler for normalization
scaler_feature = joblib.load('normalization_feature.pkl')
scaler_target = joblib.load('normalization_target.pkl')

# Normalize the input
input_normalized = pd.DataFrame(scaler_feature.transform(input_df), columns=input_df.columns)

# Display user input features in the app
st.subheader('User Input Features')
st.write(input_df)

# Prediction using the Random Forest model
if st.button("Predict"):
    # Predict FANN 600 rpm and FANN 300 rpm
    prediction_output = predict(input_normalized)
    
    # Denormalize the output (target only, not the input)
    prediction_output_denormalized = scaler_target.inverse_transform(prediction_output)
    
    # Extract FANN 600 and FANN 300 predictions
    Fann_600_rpm = prediction_output_denormalized[0, 0]
    Fann_300_rpm = prediction_output_denormalized[0, 1]
    
    # Display the predictions
    st.subheader('Predicted Outputs')
    st.write(f"FANN 600 rpm: {Fann_600_rpm:.2f}")
    st.write(f"FANN 300 rpm: {Fann_300_rpm:.2f}")
    
    # Calculate additional outputs
    PV = Fann_600_rpm - Fann_300_rpm
    YP = Fann_300_rpm - PV
    AV = Fann_600_rpm / 2
    N = 3.32 * np.log10(Fann_600_rpm / Fann_300_rpm)
    K = Fann_600_rpm / (1000 ** N)
    
    # Display the calculated values
    st.subheader('Additional Calculations')
    st.write(f"Plastic Viscosity (PV): {PV:.2f}")
    st.write(f"Yield Point (YP): {YP:.2f}")
    st.write(f"Absolute Viscosity (AV): {AV:.2f}")
    st.write(f"Flow Behavior Index (N): {N:.4f}")
    st.write(f"Consistency Index (K): {K:.4f}")
