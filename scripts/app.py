import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from PIL import Image

# Load the trained model, encoders, and feature names
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('station_encoder.pkl', 'rb') as f:
    station_encoder = pickle.load(f)

with open('day_encoder.pkl', 'rb') as f:
    day_encoder = pickle.load(f)

with open('stacking_model.pkl', 'rb') as file:
    stacking_model = pickle.load(file)

# Define the available stations
stations = ['CHLMSFD', 'CLCHSTR', 'DISS', 'IPSWICH', 'MANNGTR', 'MRKSTEY', 'SHENFLD', 'STFD', 'STWMRKT', 'WITHAME']

# Title and icons
st.title('🚉 Train Delay Prediction')

# Select station with train icon
station = st.selectbox('Select Station', stations, format_func=lambda x: '🚂 ' + x)

# Select date with calendar icon
date = st.date_input('Select Date 📅', datetime.today())

# Select time with clock icon
departure_time = st.time_input('Select Departure Time 🕒', time(12, 00))

# Summary of input details
st.markdown("### Summary of Selected Details")
st.write(f"**Station:** {station} 🚉")
st.write(f"**Date:** {date.strftime('%A, %d %B %Y')} 📅")
st.write(f"**Departure Time:** {departure_time.strftime('%H:%M')} 🕒")

# Process inputs
def process_inputs(station, date, departure_time):
    # Calculate stops
    sorted_stations = [
        'NRCH', 'NRCHTPJ', 'TROWSEJ', 'TRWSSBJ', 'DISS', 'STWMRKT', 'NEEDHAM',
        'IPSWICH', 'IPSWESJ', 'IPSWEPJ', 'IPSWHJN', 'MANNGTR', 'CLCHSTR', 'MRKSTEY',
        'WITHAME', 'KELVEDN', 'CHLMSFD', 'SHENFLD', 'BRTWOOD', 'HRLDWOD', 'GIDEAPK',
        'ROMFORD', 'CHDWLHT', 'GODMAYS', 'SVNKNGS', 'ILFORD', 'MANRPK', 'FRSTGT',
        'FRSTGTJ', 'STFD', 'MRYLAND', 'LIVST', 'BTHNLGR', 'BOWJ', 'GIDEPKJ',
        'HAGHLYJ', 'HFLPEVL', 'ILFELEJ', 'INGTSTN', 'INGTSTL', 'SHENFUL', 'SHENLEJ',
        'STWMDGL', 'TROWFLR', 'TROWLKJ', 'WHELSTJ'
    ]

    def calculate_stops(station, base_index=sorted_stations.index('NRCH')):
        station_index = sorted_stations.index(station)
        return abs(base_index - station_index)

    stops = calculate_stops(station)

    # Calculate is_weekend
    is_weekend = date.weekday() >= 5

    # Calculate is_offpeak
    def check_offpeak(date, departure_time):
        if is_weekend:
            return False
        time_obj = datetime.strptime(departure_time.strftime('%H:%M:%S'), '%H:%M:%S').time()
        offpeak_start = time(9, 0, 0)
        offpeak_end = time(21, 0, 0)
        return not (offpeak_start <= time_obj <= offpeak_end)

    is_offpeak = check_offpeak(date, departure_time)

    # Prepare input data in the same order as used during model training
    input_data = pd.DataFrame({
        'STATION': [station],
        'HOUR': [departure_time.hour],
        'MIN': [departure_time.minute],
        'DAY': [date.strftime('%A')],
        'is_weekend': [1 if is_weekend else 0],
        'is_offpeak': [1 if is_offpeak else 0],
        'year': [date.year],
        'month': [date.month],
        'day': [date.day],
        'stops': [stops],
        'weekday': [date.weekday()],
        'avg_train_delay': [5]  # Replace with actual average delay if available
    })

    # Encode categorical features
    input_data['STATION'] = station_encoder.transform(input_data['STATION'])
    input_data['DAY'] = day_encoder.transform(input_data['DAY'])

    # Reorder columns to match the training order
    feature_names = ['STATION', 'stops', 'is_weekend', 'is_offpeak', 'DAY', 'HOUR', 'year', 'month', 'day', 'weekday', 'MIN', 'avg_train_delay']
    input_data = input_data[feature_names]

    # Scale features
    input_data_scaled = scaler.transform(input_data)

    return input_data_scaled

# Make prediction
if st.button('Predict'):
    input_data_scaled = process_inputs(station, date, departure_time)
    prediction = stacking_model.predict(input_data_scaled)
    
    # Display the prediction with interactive color coding
    if prediction[0] < 0:
        st.markdown(f"<h3 style='color:green;'>🎉 No Delay Expected: {abs(prediction[0]):.2f} minutes early</h3>", unsafe_allow_html=True)
    elif prediction[0] < 3:
        st.markdown(f"<h3 style='color:orange;'>⚠️ Slight Delay: {prediction[0]:.2f} minutes</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:red;'>🚨 Significant Delay: {prediction[0]:.2f} minutes</h3>", unsafe_allow_html=True)
