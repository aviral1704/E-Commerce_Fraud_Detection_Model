#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:37:04 2024

@author: aviralh
"""

import pandas as pd
import numpy as np

# Load the dataset
file_path = './iv_df.csv'  # Replace with the correct path to your CSV file
df = pd.read_csv(file_path)

# Calculate the correlation matrix
corr_matrix = df.corr().abs()

# Create a mask to identify highly correlated features
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find the index of feature columns with correlation greater than 0.7
to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]

# Drop highly correlated features
df_reduced = df.drop(columns=to_drop)

# Save the reduced dataset to a new CSV file
output_file_path = './iv_df_reduced.csv'
df_reduced.to_csv(output_file_path, index=False)

print(f"Columns to drop due to high correlation: {to_drop}")
print(f"Reduced dataset saved to {output_file_path}")
