import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from time import time

# loading data with torchvision
print('Loading train set...')
train_dataset = datasets.CIFAR10(root='./data/train', train=True, download=True,
                                 transform=transforms.ToTensor())
train_dl = DataLoader(train_dataset, batch_size=5000, shuffle=True)

print('Loading test set...')
test_dataset = datasets.CIFAR10(root='./data/test', train=False, download=True,
                                transform=transforms.ToTensor())
test_dl = DataLoader(test_dataset, batch_size=5000)

print(len(train_dataset), len(test_dataset))

# classification dictionary
class_dict = {0:'plane', 1:'car', 2:'bird', 3:'cat', 4:'deer', 5:'dog',
              6:'frog', 7:'horse', 8:'ship', 9:'truck'}

# visualize some pictures
choice = input("See some examples? Y\\N\n")
while choice.lower() == "y":
    for x, y in train_dl:
        plt.imshow(x[0].permute(1, 2, 0))
        print(class_dict[int(y[0].detach().numpy())])
        plt.show()
    choice = input("See some more examples? Y\\N\n")

# check for cuda
cuda = torch.cuda.is_available()

# create model
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(16),
    nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(64),
    nn.Flatten(),
    nn.Linear(8 * 8 * 64, 512),
    nn.Linear(512, 256),
    nn.Linear(256, 10)
)

if cuda:
    model.cuda()

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optim = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.1)

# accuracy helper function
def calculate_accuracy(dataloader):
    count = 0
    with torch.no_grad():
        for X, y in dataloader:
            if cuda:
                X = X.cuda()
                y = y.cuda()
            _, preds = torch.max(model(X), dim=1)
            count += torch.sum(preds == y).item()
            print(count)
    print("len", len(dataloader) * dataloader.batch_size)
    return count / (len(dataloader) * dataloader.batch_size) * 100

begin = time()
print("Initial accuracy: {:.2f}%".format(calculate_accuracy(test_dl)))
print(time() - begin)



