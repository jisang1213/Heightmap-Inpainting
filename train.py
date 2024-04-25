# Import the required packages.
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Import modules
from dataset import TerrainDataset
from model import UNET
from loss import VGG16FeatureExtractor, InpaintingLoss
from maskgenerator import MaskGenerator


# Fix the random seed.
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Training & optimization hyper-parameters.
num_epochs = 10
learning_rate = 0.1
batch_size = 10
device = "cuda"

# Load data
transform = transforms.Compose(
    [
        transforms.Resize((200, 200)),  # Resize the image to 200x200
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalize the tensor
    ]
)

dataset = TerrainDataset(root_dir="./data/images", transform=transform)  # CHANGE THIS

train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Create model and loss function
model = UNET(in_channels=1, out_channels=1)
vgg = VGG16FeatureExtractor()
criterion = InpaintingLoss(vgg)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Change train_loader to have input(masked image), mask, and gt image. Size of image should be 300x300 or 400x400 but start with 200x200

train_losses = []
test_losses = []

##Load Data and create training loop
for epoch in range(num_epochs):
    # Train phase
    model.train()

    epoch_loss = 0
    epoch_acc = 0
    n_train = 0

    for input, mask, img in train_loader:
        optimizer.zero_grad()
        output = model(input, mask)  # GET MASK
        losses = criterion(
            input, mask, output, img
        )  # all arguments should be 4D tensors e.g. (3, 1, 200, 200)
        loss = (
            losses["valid"]
            + 6 * losses["hole"]
            + 0.05 * losses["perc"]
            + 120 * losses["style"]
            + 0.1 * losses["tv"]
        )
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * input.size(0)
        n_train += input.size(0)

    # Save losses and accuracies in a list so that we can visualize them later.
    epoch_loss /= n_train
    train_losses.append(epoch_loss)

    # Test phase
    model.eval()
    n_test = 0.0
    test_loss = 0.0

    for input, mask, img in test_loader:
        optimizer.zero_grad()
        output = model(input, mask)  # GET MASK
        losses = criterion(
            input, mask, output, img
        )  # all arguments should be 4D tensors e.g. (3, 1, 200, 200)
        loss = (
            losses["valid"]
            + 6 * losses["hole"]
            + 0.05 * losses["perc"]
            + 120 * losses["style"]
            + 0.1 * losses["tv"]
        )
        loss.backward()
        optimizer.step()
        test_loss += loss.item() * input.size(0)
        n_test += input.size(0)

    # Save losses and accuracies in a list so that we can visualize them later.
    test_loss /= n_test
    test_losses.append(test_loss)

    # print(f"Epoch {epoch+1}, Train Loss: {epoch_loss / len(train_loader)}")
    # print(f"    Test Loss: {test_loss / len(test_loader)}\n")
    print(
        "[epoch:{}] train loss : {:.4f}  test_loss : {:.4f}".format(
            epoch + 1, epoch_loss, test_loss
        )
    )


# Visualize the losses for the train and test set.
plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.legend()
plt.show()
