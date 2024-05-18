#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:13:42 2024

@author: aviralh
"""

import pandas as pd
import numpy as np

# Load the dataset
file_path = './merged_df copy.csv'
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

# Display IV values
for feature, iv in iv_values.items():
    print(f"Feature: {feature}, IV: {iv}")
