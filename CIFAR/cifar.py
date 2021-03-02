import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from time import time

# defining transformations
rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=rgb_mean, std=rgb_std)
])

# data augmentation
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(contrast=(0.8, 1.2), saturation=(0.9, 1.1)),
    to_tensor
])

# loading data with torchvision
print('Loading train set...')
train_dataset = datasets.CIFAR10(root='./data/train', train=True, download=True, transform=augment)
train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=3)

print('Loading test set...')
test_dataset = datasets.CIFAR10(root='./data/test', train=False, download=True, transform=to_tensor)
test_dl = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)

print(len(train_dataset), len(test_dataset))

# classification dictionary
class_dict = {0:'plane', 1:'car', 2:'bird', 3:'cat', 4:'deer',
              5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

# visualize some pictures
choice = input("See some examples? Y\\N\n")
im_iter = iter(train_dl)
while choice.lower() == "y":
    images, labels = im_iter.next()
    plt.figure(figsize=(12, 12))
    plt.imshow(utils.make_grid(images).permute(1, 2, 0) * rgb_std + rgb_mean)
    plt.show()
    choice = input("See some more examples? Y\\N\n")

# check for cuda
cuda = torch.cuda.is_available()

# create model
model = nn.Sequential(
    ####################  First Convs  ####################
    nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(32),

    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(32),

    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout(p=0.2, inplace=True),
    #######################################################

    #################### Second Convs #####################
    nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(64),

    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(64),

    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout(p=0.2, inplace=True),
    #######################################################

    #################### Final Convs ######################
    nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(128),

    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(128),

    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout(p=0.2, inplace=True),
    #######################################################

    #################### Dense Layers #####################
    nn.Flatten(),

    nn.Linear(4 * 4 * 128, 256),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(256),

    nn.Linear(256, 10),
    #######################################################
)

if cuda:
    model.cuda()

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optim = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)

# accuracy helper function
def calculate_accuracy(dataloader):
    total = 0
    correct = 0
    with torch.no_grad():
        for x_set, y_set in dataloader:
            if cuda:
                x_set = x_set.cuda()
                y_set = y_set.cuda()
            preds_set = torch.argmax(model(x_set), dim=1)
            total += y_set.size(0)
            correct += (preds_set == y_set).sum().item()
    return correct / total * 100

epochs = 10
print("Trainning for {} epochs...".format(epochs))

# memes
if cuda:
    print("HahA GPU goes brrrrr")

# training loop
model.train()
losses = []
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

        losses.append(loss)

    if not (epoch + 1) % 5:
        print("Epoch: {}  Loss: {:.5f}".format(epoch + 1, loss))

# stats
print('Done! Training time {:.2f}m'.format((time() - start_time) / 60))
plt.figure("Learning Curve")
plt.plot(losses)
plt.show()

model.eval()
print("\nTrain Accuracy: {:.2f}%".format(calculate_accuracy(train_dl)))
print("Test Accuracy: {:.2f}%".format(calculate_accuracy(test_dl)))

# testing model
model.cpu()
choice = input("Make some predictions? Y\\N\n")
im_iter = iter(test_dl)
while choice.lower() == "y":
    images, labels = im_iter.next()
    preds = torch.argmax(model(images), dim=1)
    print("Predictions:", ", ".join([class_dict[x.item()] for x in preds]))
    print("Labels:     ", ", ".join([class_dict[x.item()] for x in labels]))
    plt.figure(figsize=(12, 12))
    plt.imshow(utils.make_grid(images).permute(1, 2, 0) * rgb_std + rgb_mean)
    plt.show()
    choice = input("See some more predictions? Y\\N\n")
