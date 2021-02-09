import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
TEST_SET_RATIO = 0.3
BATCH_SIZE = 32

print("Loading data...")
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
           'DIS', 'RAD', 'TAX', 'PTRAT', 'B', 'LSTAT', 'MEDV']

data_set = pd.read_csv("housing.csv", delim_whitespace=True)
data_set.columns = columns
m = len(data_set)

# clean up outlier entries with MEDV > 50
data_set = data_set[data_set["MEDV"] < 50].dropna()

# dataset visualization
print("Entries:", len(data_set), "\n")
print(data_set.describe())
print("Parameter distributions")
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(16, 8))
index = 0
axs = axs.flatten()
for k, v in data_set.items():
    sns.histplot(v, ax=axs[index], kde=True)
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()

plt.figure(figsize=(16, 8))
sns.heatmap(data_set.corr().abs(), annot=True)
print("Correlation heat-map")
plt.show()

# mean normalize data
data_set = data_set.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

# we're done with pandas -> going back to numpy
X = data_set.drop(columns=["CHAS", "MEDV"]).to_numpy(dtype=np.float32)
y = data_set["MEDV"].to_numpy(dtype=np.float32)
n = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_RATIO, shuffle=True)

print("Train set:", X_train.shape[0])
print("Test set:", X_test.shape[0])

# now going to pytorch tensors
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
assert X_train.dtype == torch.float32

# making the model
model = nn.Sequential(
    nn.Linear(n, 1),
    nn.Flatten(0, 1)
)
nn.init.normal_(model[0].weight, mean=0, std=0.1)
nn.init.constant_(model[0].bias, val=0)

# making a batch data loader
train_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)

# loss functionm
criterion = nn.MSELoss()

# optimizer
opt = torch.optim.SGD(model.parameters(), lr=0.01)

# initial loss
print("\nInitial train loss: {:.4f}".format(criterion(model(X_train), y_train)))
print("Initial test loss: {:.4f}\n".format(criterion(model(X_test), y_test)))

# training
num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0
    for x, y in train_dl:
        preds = model(x)
        loss = criterion(preds, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss
    print("Epoch {}, Loss: {:.4f}".format(epoch + 1, total_loss.item()))

# final loss
print("\nFinal train loss: {:.4f}".format(criterion(model(X_train), y_train)))
print("Final test loss: {:.4f}".format(criterion(model(X_test), y_test)))
