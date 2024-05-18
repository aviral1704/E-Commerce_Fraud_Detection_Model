#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:45:58 2024

@author: aviralh
"""

import pandas as pd

# Load the dataset
transactions_final = pd.read_csv('./processed_customer_transactions.csv')

customer_level_data = transactions_final.groupby('customerEmail').max().reset_index()

customer_level_data.to_csv('./transactions_final.csv', index=False)

print(customer_level_data.isnull().sum())
