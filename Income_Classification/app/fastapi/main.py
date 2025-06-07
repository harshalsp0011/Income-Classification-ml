import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional



# Use the absolute path to the model file
model = joblib.load('/Users/patil/Workspace/VS code/Income_Classification/app/fastapi/random_forest_model.joblib')


# Initialize FastAPI app
app = FastAPI()

# Define the input data structure using Pydantic
class UserInput(BaseModel):
    Age: int
    Hours_per_Week: int
    Education_Num: int
    Capital_Gain: int
    Capital_Loss: int
    Fnlwgt: int
    Sex: str
    Relationship: str
    Workclass: str
    Occupation: str

# Function to preprocess user input
def preprocess_input(data):
    # Derived Features
    data['Age_to_hours_ratio'] = data['Age'] / data['Hours_per_Week']
    data['Age_squared'] = data['Age'] ** 2
    data['Hours_per_week_squared'] = data['Hours_per_Week'] ** 2
    data['Age_hours_interaction'] = data['Age'] * data['Hours_per_Week']
    
    # One-Hot Encoding for categorical columns
    categorical_cols = ["Sex", "Relationship", "Workclass", "Occupation"]
    data_encoded = pd.get_dummies(pd.DataFrame([data]), columns=categorical_cols, drop_first=True)

    # Ensure the processed input matches the model's expected feature set
    required_columns = [
        'Age', 'Hours_per_Week', 'Education_Num', 'Capital_Gain', 'Capital_Loss', 'Fnlwgt',
        'Sex_Male', 'Relationship_Not-in-family', 'Relationship_Own-child', 'Relationship_Unmarried', 
        'Relationship_Wife', 'Workclass_Private', 'Occupation_Exec-managerial', 
        'Occupation_Prof-specialty', 'Education_9', 
        'Age_to_hours_ratio', 'Age_squared', 'Hours_per_week_squared', 'Age_hours_interaction'
    ]
    for col in required_columns:
        if col not in data_encoded.columns:
            data_encoded[col] = 0
    data_encoded = data_encoded[required_columns]

    return data_encoded

# Define the prediction endpoint
@app.post("/predict")
def predict_income(input_data: UserInput):
    # Convert input data to a dictionary
    input_dict = input_data.dict()

    # Preprocess and predict
    processed_data = preprocess_input(input_dict)
    prediction = model.predict(processed_data)[0]

    # Return the prediction result
    if prediction == 0:
        return {"Predicted Income": "<=50K"}
    else:
        return {"Predicted Income": ">50K"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


"""

[Unit]
Description=gunicorn daemon
Requires=gunicorn.socket
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ec2-user/income
ExecStart=/home/ec2-user/income/env/bin/gunicorn \
          --access-logfile - \
          --workers 5 \
          --bind unix:/run/gunicorn.sock \
          --worker-class uvicorn.workers.UvicornWorker \
          main:app

[Install]
WantedBy=multi-user.target

"""