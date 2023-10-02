from pycaret.regression import load_model, predict_model
import streamlit as st 
import pandas as pd
import numpy as np


model = load_model("Final blender model 27Sep2023")

def predict(model, input_df):
    predictions_df = predict_model(model, input_df)
    predictions = predictions_df["prediction_label"][0]
    return predictions

# Frontend App

st.title("Insurance Charges Predition App")

# Side bar
add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch")
)

st.sidebar.info("This app predict patient hospital charges.")
st.sidebar.success("This is a regression modeling example.")

# Main menu - Online
if add_selectbox == "Online":
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10, max_value=50, value=15)
    children = st.selectbox("Children", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    if st.checkbox("Smoker"):
        smoker = "yes"
    else:
        smoker = "no"
    region = st.selectbox("Region", ["southwest", "northwest", "northeast", "southeast"])
    
    input_dict = {"age": age, "sex": sex, "bmi": bmi, "children": children, 
                "smoker": smoker, "region": region}
    input_df = pd.DataFrame([input_dict])
    
    output = ""
    if st.button("Predict"):
        output = predict(model=model, input_df=input_df)
        output = round(output, 2)
        st.success(f"The output is US${str(output)}")

# Main menu - Batch
if add_selectbox == "Batch":
    file_upload = st.file_uploader("Upload csv file for predictions:", type=["csv"])
    
    if file_upload is not None:
        data = pd.read_csv(file_upload).reset_index()
        
        predictions = predict_model(estimator=model, data=data).reset_index()
        predictions = predictions[["index", "prediction_label"]]
        
        final = pd.merge(
            how="left",
            left=data,
            right=predictions,
            left_on="index",
            right_on="index"
        ).drop("index", axis=1)
        
        st.write(final)
        
        
