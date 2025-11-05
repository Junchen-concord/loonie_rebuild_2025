# ## Train with training pipeline

import os
import pandas as pd

import importlib
import preprocessing_pipeline
importlib.reload(preprocessing_pipeline)
import training_pipeline
importlib.reload(training_pipeline)

# Define file paths
# zip_path = "yangyi\\gen4_training_data\\Originated_Gen4Input_JSON_V4_zipped.zip"
# target_path = os.path.join("gen4_isGood", "gen4_isGood_target.csv")
# output_dir = "gen4_isGood"
# date_col = "i01_ApplicationDate"  # Adjust if your date column is named differently

# pd.set_option('future.no_silent_downcasting', True)

# model_path = os.path.join(output_dir, "gen4_isGood.json")
# target_col = 'isGood'
# param_space = {
#     'learning_rate': (0.01, 0.2, 'log-uniform'),
#     'depth': (4, 10),                 # integer range
#     'l2_leaf_reg': (4, 10)            # integer range
# }

# Define file paths
zip_path = "gen4_isGood/Originated_Gen4Input_JSON_V4_zipped.zip"
target_path = os.path.join("gen4_isGood", "gen4_isGood_target.csv")
output_dir = "outputs/gen4_isGood/outputs_and_model"
date_col = "i01_ApplicationDate"  # Adjust if your date column is named differently
split_date = "2025-06-01"

try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception:
    # older pandas does not support this option
    pass

model_path = os.path.join(output_dir, "gen4_Conv.json")
target_col = 'isGood'
param_space = {
    'learning_rate': (0.01, 0.2, 'log-uniform'),
    'depth': (4, 10),                 # integer range
    'l2_leaf_reg': (4, 10)            # integer range
}

# import inspect
# print(inspect.signature(preprocessing_pipeline.convert_object_to_numeric))

best_model, test_report = training_pipeline.train_pipeline(
    zip_path, target_path, date_col, output_dir, 
    model_path, target_col, split_date, param_space
)
