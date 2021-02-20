from time import time
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_train, load_test

print("Loading data...")

X, Y, dims = load_train()
X_test, y_test, dimst = load_test()
dims = [dims[0], 1, dims[1], dims[2]]
dimst = [dimst[0], 1, dimst[1], dimst[2]]

X = X.reshape(dims) / 256
X_test = X_test.reshape(dimst) / 256
y_test = y_test.reshape(dimst[0])

print("Train set:", X.shape)
print("Train labels:", Y.shape)

# visualize some pictures
choice = input("See some examples? Y\\N\n")
while choice.lower() == "y":
    for i in range(5):
        ex = np.random.randint(0, dims[0])
        print("Example:", Y[ex])
        plt.figure(Y[ex])
        plt.imshow(X[ex].reshape([28, 28]), cmap="gray")
        plt.show()
    choice = input("See some more examples? Y\\N\n")

# check for cuda
cuda = torch.cuda.is_available()

# make tensors
X_train = torch.from_numpy(X.astype(np.float32))
y_train = torch.from_numpy(Y.astype(np.int64))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.int64))

# make train data set
train_ds = torch.utils.data.TensorDataset(X_train, y_train)

# make train data loader
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=8192, shuffle=True)

# model
model = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(4, 10, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(7 * 7 * 10, 200),
    nn.Linear(200, 100),
    nn.Linear(100, 10)
)
if cuda:
    model.cuda()

# loss functions
criterion = nn.CrossEntropyLoss()

# optimizer
optim = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.2)

# helper function to calculate accuracy
def calculate_accuracy(X_set, y_set):
    with torch.no_grad():
        _, preds_set = torch.max(model(X_set), dim=1)
        return torch.sum(preds_set == y_set).item() / len(y_set) * 100

if cuda:
    X_test = X_test.cuda()
    y_test = y_test.cuda()
print("Initial test accuracy: {:.2f}%\n".format(calculate_accuracy(X_test, y_test)))
X_test = X_test.cpu()
y_test = y_test.cpu()

# training
epochs = 20
losses = []
print("Training for {} epochs...".format(epochs))
start_time = time()

for epoch in range(epochs):
    for X, y in train_dl:
        if cuda:
            X = X.cuda()
            y = y.cuda()

        preds = model(X)
        loss = criterion(preds, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print("Epoch {} Loss: {:.4f}".format(epoch + 1, loss))

print("Done! Training time: {:.2f} m\n".format((time() - start_time) / 60))

plt.figure("Learning Curve")
plt.plot(losses)
plt.show()

if cuda:
    X_test = X_test.cuda()
    y_test = y_test.cuda()
print("Test accuracy: {:.2f}%".format(calculate_accuracy(X_test, y_test)))

choice = input("Try some predictions? Y/N\n")
while choice.lower() == 'y':
    for i in range(10):
        ex = np.random.randint(0, dimst[0])
        with torch.no_grad():
            _, pred = torch.max(model(torch.unsqueeze(X_test[ex], 0)), dim=1)
        print("Prediction: ", pred.item())
        plt.imshow(X_test[ex].detach().cpu().numpy().reshape(28, 28), cmap='gray')
        plt.show()
    choice = input("Try some more predictions? Y/N\n")

choice = input("Try to draw some examples? Y/N\n")
if choice.lower() == 'y':
    print("Draw a digit and press enter!")
    model.cpu()
    while choice.lower() == 'y':
        from drawimage import drawImage
        X_draw = np.array(drawImage(), dtype=np.float32) / 256.
        X_draw = torch.from_numpy(X_draw.reshape(1, 1, dims[2], dims[3]))
        prediction = np.argmax(model(X_draw).detach().numpy())
        print("That's a {}!".format(prediction))
        choice = input("Another one? Y/N\n")
