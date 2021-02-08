import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from time import time

traindatafile = "train-images-idx3-ubyte"
trainlabelfile = "train-labels-idx1-ubyte"
testdatafile = "t10k-images-idx3-ubyte"
testlabelfile = "t10k-labels-idx1-ubyte"

print("Loading data...")
magic1 = np.fromfile(traindatafile, dtype=np.int32, count=1).byteswap().squeeze()
assert(magic1 == 2051)
dims = np.fromfile(traindatafile, dtype=np.int32, count=3, offset=4).byteswap()
dims = (dims[0], dims[1] * dims[2])
X_train = np.fromfile(traindatafile, dtype=np.dtype('>u1'), count=dims[0]*dims[1],
                      offset=16).reshape(dims)
print("Train set:", X_train.shape)

magic2 = np.fromfile(trainlabelfile, dtype=np.int32, count=1).byteswap().squeeze()
assert(magic2 == 2049)
Y_train_aux = np.fromfile(trainlabelfile, dtype=np.dtype('>u1'), count=dims[0], offset=8)
Y_train = np.zeros((dims[0], 10))
for i in range(dims[0]):
    Y_train[i, Y_train_aux[i] % 10] = 1
print("Train labels:", Y_train.shape)

clothes_dict = {0:"T-shirt", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat",
                5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"}

choice = input("See some examples? Y\\N\n")
while (choice.lower() == "y"):
    for i in range(5):
        ex = np.random.randint(0, dims[0])
        print("Example:", clothes_dict[np.argmax(Y_train[ex])])
        plt.imshow(X_train[ex].reshape([28, 28]))
        plt.show()
    choice = input("See some more examples? Y\\N\n")

inputs = torch.tensor(X_train, dtype=torch.float32)
targets = torch.tensor(Y_train_aux, dtype=torch.long)
train_ds = TensorDataset(inputs, targets)
batch_size = 128
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

model = nn.Sequential(
    torch.nn.Linear(784, 500),
    torch.nn.Linear(500, 100),
    torch.nn.Linear(100, 10)
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(10):
    total_loss = 0
    for inputs, targets in train_dl:
        pred = model(inputs)
#        print(pred.shape, targets.shape)
        loss = criterion(pred, targets)
        total_loss += loss

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    print("Epoch {}, Loss: {:.4f}".format(epoch + 1, total_loss.item()))



