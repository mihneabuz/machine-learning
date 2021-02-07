import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

print("Loading data...")
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
           'DIS', 'RAD', 'TAX', 'PTRAT', 'B', 'LSTAT', 'MEDV']

data_set = pd.read_csv("housing.csv", delim_whitespace=True)
data_set.columns = columns
print(data_set.head())
print("...\nEntries:", len(data_set), "\n")
print(data_set.describe())

data_set = data_set[data_set["MEDV"] < 50]

print("Processed data")
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(16, 8))
index = 0
axs = axs.flatten()
for k, v in data_set.items():
    sns.histplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()

X_train = data_set.drop(columns=["MEDV", "CHAS"])
results = data_set["MEDV"]

plt.figure(figsize=(16, 8))
sns.heatmap(data_set.corr().abs(), annot=True)
plt.show()
