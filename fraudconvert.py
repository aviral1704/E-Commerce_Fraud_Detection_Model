#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:06:41 2024

@author: aviralh
"""

import pandas as pd

# Load the dataset
customer_df_final = pd.read_csv('./ecommerce_fraud_detection_dataset/customers_df.csv')

customer_df_final['Fraud'] = customer_df_final['Fraud'].apply(lambda x: 1 if x else 0)



customer_df_final.to_csv('./customer_df_final.csv', index=False)
