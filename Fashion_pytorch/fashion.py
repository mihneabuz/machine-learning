import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from time import time

traindatafile = "train-images-idx3-ubyte"
trainlabelfile = "train-labels-idx1-ubyte"
testdatafile = "t10k-images-idx3-ubyte"
testlabelfile = "t10k-labels-idx1-ubyte"
DEV_SET_RATION = 0.01

# Loading all data
print("Loading data...")
magic = np.fromfile(traindatafile, dtype=np.int32, count=1).byteswap().squeeze()
assert magic == 2051
dims = np.fromfile(traindatafile, dtype=np.int32, count=3, offset=4).byteswap()
dims = (dims[0], dims[1] * dims[2])
X = np.fromfile(traindatafile, dtype=np.dtype('>u1'), count=dims[0]*dims[1],
                      offset=16).reshape(dims) / 256
print("Train set:", X.shape)

magic = np.fromfile(trainlabelfile, dtype=np.int32, count=1).byteswap().squeeze()
assert magic == 2049
Y_aux = np.fromfile(trainlabelfile, dtype=np.dtype('>u1'), count=dims[0], offset=8)
Y = np.zeros((dims[0], 10))
for i in range(dims[0]):
    Y[i, Y_aux[i] % 10] = 1
print("Train labels:", Y.shape)

magic = np.fromfile(testdatafile, dtype=np.int32, count=1).byteswap().squeeze()
assert(magic == 2051)
dimst = np.fromfile(testdatafile, dtype=np.int32, count=3, offset=4).byteswap()
dimst = (dimst[0], dimst[1] * dimst[2])
X_test = np.fromfile(testdatafile, dtype=np.dtype('>u1'), count=dimst[0]*dimst[1],
                    offset=16).reshape(dimst) / 256

magic = np.fromfile(testlabelfile, dtype=np.int32, count=1).byteswap().squeeze()
assert(magic == 2049)
y_test = np.fromfile(testlabelfile, dtype=np.dtype('>u1'), count=dimst[0], offset=8)


# classification dictionary
clothes_dict = {0:"T-shirt", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat",
                5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"}

# visualize some pictures
choice = input("See some examples? Y\\N\n")
while (choice.lower() == "y"):
    for i in range(5):
        ex = np.random.randint(0, dims[0])
        print("Example:", clothes_dict[np.argmax(Y[ex])])
        plt.figure(clothes_dict[np.argmax(Y[ex])])
        plt.imshow(X[ex].reshape([28, 28]), cmap="gray")
        plt.show()
    choice = input("See some more examples? Y\\N\n")

# make tensors
X_train, X_dev, y_train, y_dev = train_test_split(X, Y_aux, test_size=DEV_SET_RATION, shuffle=True)
X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.int64))
X_dev = torch.from_numpy(X_dev.astype(np.float32))
y_dev = torch.from_numpy(y_dev.astype(np.int64))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.int64))

# model
model = nn.Sequential(
    nn.Linear(dims[1], 500),
    nn.Linear(500, 200),
    nn.Linear(200, 10)
)

if torch.cuda.is_available():
    model.cuda()
    X_train = X_train.cuda()
    y_train = y_train.cuda()
    X_dev = X_dev.cuda()
    y_dev = y_dev.cuda()
    X_test = X_test.cuda()
    y_test = y_test.cuda()

# loss functions
criterion = nn.CrossEntropyLoss()

# optimizer
optim = torch.optim.AdamW(model.parameters(), lr=0.01)

# helper function to calculate accuracy
def calculate_accuracy(X, y):
    with torch.no_grad():
        _, preds = torch.max(model(X), dim=1)
        return torch.sum(preds == y).item() / len(y) * 100

print("Train {:.2f}%".format(calculate_accuracy(X_train, y_train)))
print("Dev {:.2f}%".format(calculate_accuracy(X_dev, y_dev)))
print("Test {:.2f}%\n".format(calculate_accuracy(X_test, y_test)))

# training
epochs = 100
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
        print("Epoch {} Loss: {:.4f}".format(epoch + 1, loss))

print("Done! Training time: {:.2f} m\n".format((time() - start_time) / 60))

plt.figure("Learning Curve")
plt.plot(losses)
plt.show()

print("Train {:.2f}%".format(calculate_accuracy(X_train, y_train)))
print("Dev {:.2f}%".format(calculate_accuracy(X_dev, y_dev)))
print("Test {:.2f}%".format(calculate_accuracy(X_test, y_test)))

choice = input("Try some predictions? Y/N\n")
while (choice.lower() == 'y'):
    for i in range(10):
        ex = np.random.randint(0, dimst[0])
        with torch.no_grad():
            _, pred = torch.max(model(torch.unsqueeze(X_test[ex], 0)), dim=1)
        print("Prediction: ", clothes_dict[pred.item()])
        plt.imshow(X_test[ex].detach().cpu().numpy().reshape(28, 28), cmap='gray')
        plt.show()
    choice = input("Try some more predictions? Y/N\n")
