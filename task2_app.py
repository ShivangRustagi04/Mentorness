import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load the saved model, encoders, and scaler
model = joblib.load('salary_predictor_model.pkl')
le_sex = joblib.load('label_encoder_sex.pkl')
le_designation = joblib.load('label_encoder_designation.pkl')
le_unit = joblib.load('label_encoder_unit.pkl')
scaler = joblib.load('scaler.pkl')

# Function to preprocess input data
def preprocess_input(data):
    # Convert input data to DataFrame
    data_df = pd.DataFrame([data])

    # Feature Engineering
    data_df['DOJ'] = pd.to_datetime(data_df['DOJ'])
    data_df['CURRENT DATE'] = pd.to_datetime(data_df['CURRENT DATE'])
    data_df['YEARS_IN_COMPANY'] = (data_df['CURRENT DATE'] - data_df['DOJ']).dt.days / 365
    data_df.drop(columns=['DOJ', 'CURRENT DATE'], inplace=True)

    # Data Preprocessing
    data_df['SEX'] = encode_label(le_sex, data_df['SEX'])
    data_df['DESIGNATION'] = encode_label(le_designation, data_df['DESIGNATION'])
    data_df['UNIT'] = encode_label(le_unit, data_df['UNIT'])

    # Scale features
    data_df = scaler.transform(data_df)

    return data_df

# Function to handle label encoding with unseen categories
def encode_label(encoder, data):
    classes = encoder.classes_.tolist()
    for idx, val in enumerate(data):
        if val not in classes:
            data[idx] = classes[0]  # Assign the first class as default
    return encoder.transform(data)

# Streamlit App
def main():
    st.title('Salary Prediction App')

    # Sidebar - Input Form
    st.sidebar.header('Input Parameters')
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female', 'Unknown'], index=0)
    doj = st.sidebar.date_input('Date of Joining')
    current_date = st.sidebar.date_input('Current Date')
    designation = st.sidebar.selectbox('Designation', ['Data Scientist', 'Data Analyst', 'Data Engineer', 'Unknown'], index=0)
    age = st.sidebar.slider('Age', 20, 60, 30)
    unit = st.sidebar.selectbox('Unit', ['Analytics', 'IT', 'Finance', 'Unknown'], index=0)
    leaves_used = st.sidebar.number_input('Leaves Used', min_value=0, max_value=50, value=5)
    leaves_remaining = st.sidebar.number_input('Leaves Remaining', min_value=0, max_value=50, value=20)
    ratings = st.sidebar.slider('Ratings', 1.0, 5.0, 4.0)
    past_exp = st.sidebar.number_input('Past Experience (years)', min_value=0, max_value=30, value=5)

    # Predict button
    if st.sidebar.button('Predict'):
        input_data = {
            'SEX': sex,
            'DOJ': doj,
            'CURRENT DATE': current_date,
            'DESIGNATION': designation,
            'AGE': age,
            'UNIT': unit,
            'LEAVES USED': leaves_used,
            'LEAVES REMAINING': leaves_remaining,
            'RATINGS': ratings,
            'PAST EXP': past_exp
        }

        # Preprocess input data
        input_data_processed = preprocess_input(input_data)

        # Predict
        prediction = model.predict(input_data_processed)

        # Display prediction
        st.subheader('Prediction')
        st.write(f'The predicted salary is: {prediction[0]:,.2f} INR')

if __name__ == '__main__':
    main()
