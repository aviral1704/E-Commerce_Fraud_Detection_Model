#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:51:07 2024

@author: aviralh
"""

import pandas as pd

# Load the dataset
file_path = './merged_df copy.csv'  # Replace with the correct path to your CSV file
df = pd.read_csv(file_path)

# Columns to retain
columns_to_retain = [
    'count_failed',
    'unique_mpro_pc',
    'count_American Express',
    'transactionFailed',
    'count_Maestro',
    'count_paypal',
    'paymentMethodRegistrationFailure',
    'unique_mtype_pc',
    'Fraud'
]

# Filter the dataset to retain only the specified columns
df_filtered = df[columns_to_retain]

# Save the filtered dataset to a new CSV file
output_file_path = 'filtered_eda_columns.csv'  # Path where you want to save the new CSV file
df_filtered.to_csv(output_file_path, index=False)

print(f"Filtered dataset saved to {output_file_path}")
