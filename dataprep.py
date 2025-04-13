import pandas as pd
import numpy as np

# Read the threshold and dataset files
thresholds = pd.read_csv('threshold.csv')
dataset = pd.read_csv('wqns_dataset.csv')

# Create a function to check parameter conditions
def check_parameter_condition(value, param_name, thresholds):
    # Find the min and max for the specific parameter
    param_threshold = thresholds[thresholds['Parameter'] == param_name]
    
    if param_threshold.empty:
        return 'Unknown'
    
    min_val = param_threshold['Min'].values[0]
    max_val = param_threshold['Max'].values[0]
    
    # Check if value is within the threshold range
    if min_val <= value <= max_val:
        return 'Safe'
    else:
        return 'Unsafe'

# List of parameters to check (excluding observationTime)
parameters = [col for col in dataset.columns if col != 'observationTime']

# Create a condition column
def determine_overall_condition(row):
    # Check each parameter's condition
    parameter_conditions = [
        check_parameter_condition(row[param], param, thresholds) 
        for param in parameters
    ]
    
    # If any parameter is Unsafe, the overall condition is Unsafe
    if 'Unsafe' in parameter_conditions:
        return 'Unsafe'
    return 'Safe'

# Add the condition column
dataset['condition'] = dataset.apply(determine_overall_condition, axis=1)

# Save the updated dataset
dataset.to_csv('updated_wqns.csv', index=False)

# Print summary of conditions
condition_summary = dataset['condition'].value_counts()
print("Condition Summary:")
print(condition_summary)
print("\nTotal Rows:", len(dataset))