import torch
from torchvision import datasets
import numpy as np

dataset = datasets.CIFAR10(root='./data/train', train=True, download=True)

print(len(dataset))
