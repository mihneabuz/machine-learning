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
DEV_SET_RATION = 0.2

# Loading all data
print("Loading data...")
magic = np.fromfile(traindatafile, dtype=np.int32, count=1).byteswap().squeeze()
assert magic == 2051
dims = np.fromfile(traindatafile, dtype=np.int32, count=3, offset=4).byteswap()
dims = (dims[0], dims[1] * dims[2])
X = np.fromfile(traindatafile, dtype=np.dtype('>u1'), count=dims[0]*dims[1],
                      offset=16).reshape(dims) / 255
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
                    offset=16).reshape(dimst) / 255

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

# models
model1 = nn.Sequential(
    nn.Linear(dims[1], 200),
    nn.Linear(200, 10)
)

model2 = nn.Sequential(
    nn.Linear(dims[1], 300),
    nn.Linear(300, 100),
    nn.Linear(100, 10)
)

model3 = nn.Sequential(
    nn.Linear(dims[1], 500),
    nn.Linear(500, 200),
    nn.Linear(200, 10)
)

if torch.cuda.is_available():
    model1.cuda()
    model2.cuda()
    model3.cuda()
    X_train = X_train.cuda()
    y_train = y_train.cuda()
    X_dev = X_dev.cuda()
    y_dev = y_dev.cuda()
    X_test = X_test.cuda()
    y_test = y_test.cuda()

# loss function
criterion = nn.CrossEntropyLoss()

# optimizers
optim1 = torch.optim.SGD(model1.parameters(), lr=0.03, momentum=0.1)
optim2 = torch.optim.Adam(model2.parameters(), lr=0.0003, weight_decay=0.01)
optim3 = torch.optim.AdamW(model3.parameters(), lr=0.0003, weight_decay=0.1)

print("Training 3 models:")
print("Model 1: 784->200->10, Gradient Descent with momentum 0.1")
print("Model 1: 784->300->100->10, Adam with reg lambda 0.01")
print("Model 1: 784->500->200->10, AdamW with reg lambda 0.1\n")

# helper function to calculate accuracy
def calculate_accuracy(model, X, y):
    with torch.no_grad():
        _, preds = torch.max(model(X), dim=1)
        return torch.sum(preds == y).item() / len(y) * 100

print("Initial test accuracy:")
print("Model 1: {:.2f}%".format(calculate_accuracy(model1, X_test, y_test)))
print("Model 2: {:.2f}%".format(calculate_accuracy(model2, X_test, y_test)))
print("Model 3: {:.2f}%\n".format(calculate_accuracy(model3, X_test, y_test)))

# training
epochs = 200
losses = [[], [], []]
print("Training for {} epochs...".format(epochs))
start_time = time()

for epoch in range(epochs):
    preds1 = model1(X_train)
    preds2 = model2(X_train)
    preds3 = model3(X_train)

    loss1 = criterion(preds1, y_train)
    loss2 = criterion(preds2, y_train)
    loss3 = criterion(preds3, y_train)

    optim1.zero_grad()
    optim2.zero_grad()
    optim3.zero_grad()

    loss1.backward()
    loss2.backward()
    loss3.backward()

    optim1.step()
    optim2.step()
    optim3.step()

    losses[0].append(loss1.item())
    losses[1].append(loss2.item())
    losses[2].append(loss3.item())

    if (epoch + 1) % 10 == 0:
        print("Epoch {} Loss: {:.4f}".format(epoch + 1, loss1 + loss2 + loss3))

print("Done! Training time: {:.2f} m\n".format((time() - start_time) / 60))

plt.figure("Learning Curves")
plt.plot(losses[0], "r")
plt.plot(losses[1], "b")
plt.plot(losses[2], "g")
plt.show()

print("Test accuracy:")
acc1 = calculate_accuracy(model1, X_dev, y_dev)
acc2 = calculate_accuracy(model2, X_dev, y_dev)
acc3 = calculate_accuracy(model3, X_dev, y_dev)

print("Model 1: {:.2f}%".format(acc1))
print("Model 2: {:.2f}%".format(acc2))
print("Model 3: {:.2f}%\n".format(acc3))

model = model3
print("Test accuracy: {:.2f}%".format(calculate_accuracy(model, X_test, y_test)))
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
