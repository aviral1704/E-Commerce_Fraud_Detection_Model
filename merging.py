#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:15:51 2024

@author: aviralh
"""

import pandas as pd

# Load the datasets
customer_df_final = pd.read_csv('./customer_df_final.csv')
transactions_final = pd.read_csv('./transactions_final copy.csv')

merged_df = pd.merge(transactions_final, customer_df_final, on='customerEmail', how='outer')

print(merged_df.isnull().sum())

merged_df.to_csv('./merged_df.csv', index=False)
