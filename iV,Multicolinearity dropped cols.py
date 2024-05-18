#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:52:47 2024

@author: aviralh
"""

import pandas as pd
import numpy as np

# Load the dataset
file_path = './iv_df.csv'  # Replace with the correct path to your CSV file
df = pd.read_csv(file_path)

# Function to calculate WOE and IV for each feature
def calc_woe_iv(df, feature, target, epsilon=1e-10):
    # Calculate the number of good and bad for each bin
    total_good = df[target].sum()
    total_bad = len(df[target]) - total_good
    iv_df = df.groupby(feature).agg({target: ['sum', 'count']})
    iv_df.columns = ['good', 'total']
    iv_df['bad'] = iv_df['total'] - iv_df['good']
    
    # Add epsilon to avoid division by zero
    iv_df['good'] = iv_df['good'] + epsilon
    iv_df['bad'] = iv_df['bad'] + epsilon
    
    # Recalculate total good and bad with epsilon added
    total_good += epsilon * len(iv_df)
    total_bad += epsilon * len(iv_df)
    
    # Calculate WOE and IV
    iv_df['dist_good'] = iv_df['good'] / total_good
    iv_df['dist_bad'] = iv_df['bad'] / total_bad
    iv_df['woe'] = np.log(iv_df['dist_good'] / iv_df['dist_bad'])
    iv_df['iv'] = (iv_df['dist_good'] - iv_df['dist_bad']) * iv_df['woe']
    
    # Return the IV value for the feature
    iv = iv_df['iv'].sum()
    return iv

# Calculate IV for all features
iv_values = {}
target = 'Fraud'
for feature in df.columns:
    if feature != target:
        iv = calc_woe_iv(df, feature, target)
        iv_values[feature] = iv

# Calculate the correlation matrix
corr_matrix = df.corr().abs()

# Create a mask to identify highly correlated features
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find pairs of columns with correlation greater than 0.7
to_drop = set()
for column in upper.columns:
    for row in upper.index:
        if upper.loc[row, column] > 0.7:
            if iv_values[row] < iv_values[column]:
                to_drop.add(row)
            else:
                to_drop.add(column)

# Drop the identified columns
df_reduced = df.drop(columns=to_drop)

# Save the reduced dataset to a new CSV file
output_file_path = './iv_df_reduced.csv'
df_reduced.to_csv(output_file_path, index=False)

print(f"Columns to drop due to high correlation: {to_drop}")
print(f"Reduced dataset saved to {output_file_path}")
