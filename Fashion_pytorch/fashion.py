from time import time
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_train, load_test

print("Loading data...")

X, Y, dims = load_train()
X_test, y_test, dimst = load_test()

dims = (dims[0], dims[1] * dims[2])
dimst = (dimst[0], dimst[1] * dimst[2])

X = X.reshape(dims)
X_test = X_test.reshape(dimst)
y_test = y_test.reshape(dimst[0])

print("Train set:", X.shape)
print("Train labels:", Y.shape)

# classification dictionary
clothes_dict = {0:"T-shirt", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat",
                5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"}

# visualize some pictures
choice = input("See some examples? Y\\N\n")
while choice.lower() == "y":
    for i in range(5):
        ex = np.random.randint(0, dims[0])
        print("Example:", clothes_dict[Y[ex]])
        plt.figure(clothes_dict[Y[ex]])
        plt.imshow(X[ex].reshape([28, 28]), cmap="gray")
        plt.show()
    choice = input("See some more examples? Y\\N\n")

# make tensors
X_train = torch.from_numpy(X.astype(np.float32))
y_train = torch.from_numpy(Y.astype(np.int64))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.int64))

# model
model = nn.Sequential(
    nn.Linear(dims[1], 500),
    nn.Linear(500, 300),
    nn.Linear(300, 300),
    nn.Linear(300, 10)
)

if torch.cuda.is_available():
    model.cuda()
    X_train = X_train.cuda()
    y_train = y_train.cuda()
    X_test = X_test.cuda()
    y_test = y_test.cuda()

# loss functions
criterion = nn.CrossEntropyLoss()

# optimizer
optim = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.2)

# helper function to calculate accuracy
def calculate_accuracy(X_set, y_set):
    with torch.no_grad():
        _, preds_set = torch.max(model(X_set), dim=1)
        return torch.sum(preds_set == y_set).item() / len(y_set) * 100

print("Initial test accuracy: {:.2f}%\n".format(calculate_accuracy(X_test, y_test)))

# training
epochs = 200
losses = []
print("Training for {} epochs...".format(epochs))
start_time = time()

for epoch in range(epochs):
    preds = model(X_train)
    loss = criterion(preds, y_train)

    optim.zero_grad()
    loss.backward()
    optim.step()

    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print("Epoch {} Loss: {:.4f} Acc: {:.2f}%".format(
            epoch + 1, loss, calculate_accuracy(X_train, y_train)))

print("Done! Training time: {:.2f} m\n".format((time() - start_time) / 60))

plt.figure("Learning Curve")
plt.plot(losses)
plt.show()

print("Test accuracy: {:.2f}%".format(calculate_accuracy(X_test, y_test)))

choice = input("Try some predictions? Y/N\n")
while choice.lower() == 'y':
    for i in range(10):
        ex = np.random.randint(0, dimst[0])
        with torch.no_grad():
            _, pred = torch.max(model(torch.unsqueeze(X_test[ex], 0)), dim=1)
        print("Prediction: ", clothes_dict[pred.item()])
        plt.imshow(X_test[ex].detach().cpu().numpy().reshape(28, 28), cmap='gray')
        plt.show()
    choice = input("Try some more predictions? Y/N\n")
