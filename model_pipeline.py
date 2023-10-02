from pycaret.regression import *
import streamlit as st 
import pandas as pd
import pandas_profiling as pf


# Load data ----
dataset = pd.read_csv("data/insurance.csv")
dataset.info()


# EDA ----
eda_report = pf.ProfileReport(dataset)
dataset.profile_report().to_file("data/eda_report")


# Data Split ----
data = dataset.sample(frac=0.9, random_state=786)
data_unseen = dataset.drop(data.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)


# Modeling pipeline ----

# Setup
exp1= setup(data=data, target="charges", session_id=123,
      numeric_imputation="mean", categorical_imputation="mode",
      categorical_features=["sex", "smoker", "region"], 
      ordinal_features={"children":["0", "1", "2", "3", "4", "5"]},
      rare_to_value=0.05, rare_value="others",
      transform_target=True, normalize=True, polynomial_features=True, 
      bin_numeric_features=["age", "bmi"]
      )

# Comparison
#models()
best = compare_models()

# Creation
ridge = create_model("ridge")
lr = create_model("lr")
huber = create_model("huber")
gbr = create_model("gbr")

# Tunning
tune_ridge = tune_model(ridge)
tune_lr = tune_model(lr)
tune_huber = tune_model(huber)
tune_gbr = tune_model(gbr)

plot_model(tune_ridge, plot="feature")

# Creation - Blending 
blender = blend_models([tune_ridge, tune_lr, tune_huber, tune_gbr],
             choose_better=True)
print(blender)

# Evaluation - Interpretation
evaluate_model(blender)

# Test Evaluation
predict_model(blender)

# Finalize
final_blender = finalize_model(blender)
print(final_blender)

# Validation
unseen_pred = predict_model(final_blender, data=data_unseen)
unseen_pred["prediction_label"]

# Save 
save_model(final_blender, "Final blender model 27Sep2023")

# # Load
# loaded = load_model("Final blender model 27Sep2023")
# r = predict_model(loaded, data=data_unseen)