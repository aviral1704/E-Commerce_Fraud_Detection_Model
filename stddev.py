#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:05:53 2024

@author: aviralh
"""

import pandas as pd

# Load the dataset
file_path = './merged_df copy.csv'
df = pd.read_csv(file_path)
df=df.drop('customerEmail',axis=1)

std_devs = df.std()

# Display the standard deviations
print(std_devs)