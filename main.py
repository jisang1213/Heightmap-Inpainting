# Import the required packages.
import os
import random

import numpy as  np

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
from torchvision import transforms
from torch_pconv import PConv2d

import matplotlib.pyplot as plt

# Fix the random seed.
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)



# Training & optimization hyper-parameters.
num_epochs = 10
learning_rate = 0.1
batch_size = 512
device = 'cuda'


#Create model and loss function
model = UNET(in_channels=1, out_channels=1)
vgg = VGG16FeatureExtractor()
criterion = InpaintingLoss(vgg)

#Create masks and masked input
img = torch.randn(1, 3, 500, 500)
mask = torch.ones((1, 1, 500, 500))
mask[:, :, 250:, :][:, :, :, 250:] = 0
input = img * mask
out = torch.randn(1, 3, 500, 500)
loss = criterion(input, mask, out, img)

#Change train_loader to have input(masked image), mask, and gt image
#Size of image should be 300x300 or 400x400
#Start with ImageNet Dataset

##Load Data and create training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(input, mask, output, img)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")