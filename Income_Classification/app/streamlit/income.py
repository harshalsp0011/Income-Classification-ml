import streamlit as st
import requests
import json

# Streamlit app
st.title("Income Classification App")

# User input form
st.header("Enter your details:")
age = st.number_input("Age", min_value=1, max_value=100, value=30)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)
education_num = st.slider("Education Number", min_value=1, max_value=16, value=9)
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
fnlwgt = st.number_input("Fnlwgt (Final Weight)", min_value=0, value=100000)
sex = st.selectbox("Sex", ['Male', 'Female'])
relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
occupation = st.selectbox("Occupation", [
    'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
    'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
    'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
    'Armed-Forces'
])

# Define the FastAPI endpoint URL
# api_url = "https://income-classifier-backend-630343362277.us-central1.run.app/predict"
api_url = "http://18.222.116.60/predict"
#api_url = "http://127.0.0.1:8000/predict"


# Predict button
if st.button("Predict Income"):
    # Prepare the input data for the API request
    input_data = {
        "Age": age,
        "Hours_per_Week": hours_per_week,
        "Education_Num": education_num,
        "Capital_Gain": capital_gain,
        "Capital_Loss": capital_loss,
        "Fnlwgt": fnlwgt,
        "Sex": sex,
        "Relationship": relationship,
        "Workclass": workclass,
        "Occupation": occupation,
    }

    # Make the API request
    response = requests.post(api_url, json=input_data)
    print(response)

    if response.status_code == 200:
        # Parse the prediction result from the response
        prediction = response.json()
        st.success(f"Predicted Income: {prediction['Predicted Income']}")
    else:
        st.error("Error in making prediction. Please check the FastAPI server.")
