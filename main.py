import pandas as pd
import numpy as np


def load_transactions(path):
    transactions = pd.read_csv(path)
    return transactions


def compute_balance(transactions):
    value_part = (transactions['value'] / np.max(np.transpose(transactions)[1:]))
    table_part = np.transpose(np.array([value_part] * (len(list(transactions)) - 1)))
    value_paid_part = -np.sum((np.transpose(transactions)[1:] < 0) * np.transpose(transactions)[1:]) * value_part
    table_paid_part = np.transpose(np.array([value_paid_part] * (len(list(transactions)) - 1)))
    balance_impact = (transactions[list(transactions)[1:]] > 0) * table_paid_part \
                     + ((transactions[list(transactions)[1:]] < 0) * transactions[list(transactions)[1:]]) * table_part
    balance_impact.to_csv('balance_impact.csv', index=False)
    return balance_impact


def compute_refund(balance_impact):
    balance = np.sum(balance_impact)
    while np.max(balance) * np.max(balance) > 0.1:
        giver = np.argmin(balance)
        taker = np.argmax(balance)
        value = np.min([np.max(balance), -np.min(balance)])
        balance[giver] += value
        balance[taker] -= value
        print(giver + ' gives ' + str(value) + ' euros to ' + taker)


compute_refund(compute_balance(load_transactions('transactions.csv')))