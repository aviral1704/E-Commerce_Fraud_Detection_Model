#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:30:11 2024

@author: aviralh
"""

import pandas as pd

# Load the dataset
transactions = pd.read_csv('./customer_transaction_details.csv')

# 1. Total transaction amount per customer
transactions['total_amount'] = transactions.groupby('customerEmail')['transactionAmount'].transform('sum')

# 2. Columns for counts of each order state per customer
transactions['count_pending'] = transactions[transactions['orderState'] == 'pending'].groupby('customerEmail')['orderState'].transform('count')
transactions['count_failed'] = transactions[transactions['orderState'] == 'failed'].groupby('customerEmail')['orderState'].transform('count')
transactions['count_fulfilled'] = transactions[transactions['orderState'] == 'fulfilled'].groupby('customerEmail')['orderState'].transform('count')

# 3. Count unique payment method providers per customer
transactions['unique_mpro_pc'] = transactions.groupby('customerEmail')['paymentMethodProvider'].transform('nunique')

# 4. Count unique payment method types per customer
transactions['unique_mtype_pc'] = transactions.groupby('customerEmail')['paymentMethodType'].transform('nunique')

# 5. Columns for each unique payment method type
payment_type_counts = pd.get_dummies(transactions['paymentMethodType'], prefix='count')
payment_type_counts = payment_type_counts.groupby(transactions['customerEmail']).transform('sum')
transactions = pd.concat([transactions, payment_type_counts], axis=1)


# 6. Columns for each unique payment method provider
provider_counts = pd.get_dummies(transactions['paymentMethodProvider'], prefix='count')
provider_counts = provider_counts.groupby(transactions['customerEmail']).transform('sum')
transactions = pd.concat([transactions, provider_counts], axis=1)

# 7. Failed total amount
transactions['failed_total_amount'] = transactions[transactions['transactionFailed'] == 1].groupby('customerEmail')['transactionAmount'].transform('sum')

# 8. Success total amount
transactions['success_total_amount'] = transactions[transactions['transactionFailed'] == 0].groupby('customerEmail')['transactionAmount'].transform('sum')

# 9. Pending sum amount
transactions['pending_sum_amount'] = transactions[transactions['orderState'] == 'pending'].groupby('customerEmail')['transactionAmount'].transform('sum')

# 10. Failed sum amount
transactions['failed_sum_amount'] = transactions[transactions['orderState'] == 'failed'].groupby('customerEmail')['transactionAmount'].transform('sum')

# 11. Fulfilled sum amount
transactions['fulfilled_sum_amount'] = transactions[transactions['orderState'] == 'fulfilled'].groupby('customerEmail')['transactionAmount'].transform('sum')

# Fill NaNs with zeros for all new columns where necessary
transactions.fillna(0, inplace=True)

# Optionally, drop duplicates to leave one row per customer
transactions.drop_duplicates('customerEmail', keep='last', inplace=True)

# Save to CSV
transactions.to_csv('./processed_customer_transactions.csv', index=False)
