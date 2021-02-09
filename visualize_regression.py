import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

m = 100

X_train, y_train = datasets.make_regression(n_samples=m, n_features=1,
                                            noise=30, random_state=1)

X = torch.from_numpy(X_train.astype(np.float32))
y = torch.from_numpy(y_train.astype(np.float32))
n = X.shape[1]

model = nn.Sequential(
    nn.Linear(n, 1),
    nn.Flatten(0, 1)
)

criterion = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.012)

fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(16, 8))
axs = axs.flatten()
axs[0].plot(X_train, y_train, 'ro')
predicted = model(X).detach().numpy()
axs[1].plot(X_train, y_train, 'ro')
axs[1].plot(X_train, predicted, 'b')
index = 2

for epoch in range(100):
    preds = model(X)
    loss = criterion(preds, y)

    loss.backward()

    opt.step()
    opt.zero_grad()

    if (epoch + 1) % 10 == 0:
        print("Epoch {}, Loss: {:.4f}".format(epoch + 1, loss.item()))
        predicted = model(X).detach().numpy()
        axs[index].plot(X_train, y_train, 'ro')
        axs[index].plot(X_train, predicted, 'b')
        index += 1

plt.show()
