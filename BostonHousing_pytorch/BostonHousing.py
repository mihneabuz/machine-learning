import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn import preprocessing

test_set_size = 10

print("Loading data...")
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
           'DIS', 'RAD', 'TAX', 'PTRAT', 'B', 'LSTAT', 'MEDV']

data_set = pd.read_csv("housing.csv", delim_whitespace=True)
data_set.columns = columns
print(data_set.head())
print("...\nEntries:", len(data_set), "\n")


# clean up entries with MEDV > 50
data_set = data_set[data_set["MEDV"] < 50].dropna()
#data_set = data_set.sample(frac=1).reset_index(drop=True)
m = len(data_set)


x = data_set.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data_set = pd.DataFrame(x_scaled, columns=columns)
data_set.columns = columns
print(data_set.head())
print("...\nEntries:", len(data_set), "\n")

# some plots to visualize data
print(data_set.describe())
print("Parameter distributions")
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(16, 8))
index = 0
axs = axs.flatten()
for k, v in data_set.items():
    sns.histplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()

plt.figure(figsize=(16, 8))
sns.heatmap(data_set.corr().abs(), annot=True)
print("Correlation heat-map")
plt.show()

# removing CHAS from inputs because of low correlation
inputs = torch.tensor(data_set.drop(columns=["MEDV", "CHAS"]), dtype=torch.float32)
results = torch.tensor(data_set["MEDV"].values, dtype=torch.float32)
assert len(inputs) == len(results)

# load data in torch data loader
tensor_dataset = TensorDataset(inputs, results)
split = (m - test_set_size, test_set_size)
batch_size = 64
train_dataset, test_dataset = random_split(tensor_dataset, split)
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_dataset, 1, shuffle=True)

model = torch.nn.Sequential(
    torch.nn.Linear(12, 1),
    torch.nn.Flatten(0, 1)
)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.000001)

epochs = 300
for epoch in range(epochs):
    total_loss = 0
    for inputs, results in train_dl:
        pred = model(inputs)
        loss = loss_fn(pred, results)
        total_loss += loss

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    if not (epoch + 1) % 10:
        print("Epoch {}, Loss: {:.4f}".format(epoch + 1, loss.item()))

wtf = 0
for inputs, target in test_dl:
        pred = model(inputs)
        print("got: {:.2f}, expected: {:.2f}, diff: {:.2f}".format(
            pred.item(), target.item(), abs(pred.item() - target.item())))
        wtf += 1
        if wtf == 10:
            break
