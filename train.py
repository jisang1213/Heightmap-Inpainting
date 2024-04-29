# Import the required packages.
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt


# Import modules
from dataset import TerrainDataset
from model import UNET
from loss import VGG16FeatureExtractor, InpaintingLoss


# Fix the random seed.
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Training & optimization hyper-parameters
num_epochs = 10
learning_rate = 0.1
batch_size = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Load data
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),  # Resize the image to 200x200
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize((0.0,), (1.0,)),  # Normalize the tensor
    ]
)

dataset = TerrainDataset(root_dir="./data/images/", transform=transform)  # CHANGE THIS

train_set, test_set = torch.utils.data.random_split(dataset, [5000, 1000])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Create model and loss function
model = UNET(in_channels=1, out_channels=1).to(device)
vgg = VGG16FeatureExtractor().to(device)
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

    # for input, mask, img in train_loader:
    input, mask, img = next(iter(train_loader))
    
    input = input.to(device)
    mask = mask.to(device)
    img = img.to(device)
    
    mask_squeezed = mask.squeeze()
    
    output = model(input, mask_squeezed)
    print(output.shape)
    
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
    optimizer.zero_grad()
    print("Checkpoint 0")
    loss.backward()
    print("Checkpoint 1")
    optimizer.step()
    print("Checkpoint 2")
    epoch_loss += loss.item() * input.size(0)
    n_train += input.size(0)
        
        
        
    # Save losses and accuracies in a list so that we can visualize them later.
    epoch_loss /= n_train
    train_losses.append(epoch_loss)

    # Test phase
    model.eval()
    n_test = 0.0
    test_loss = 0.0
    # Create the folder to store test result images
    results_path = "./data/results"
    os.makedirs(results_path, exist_ok=True)

    for i, (input, mask, img) in enumerate(test_loader):
        input = input.to(device)
        mask = mask.to(device)
        img = img.to(device)
        
        mask_squeezed = mask.squeeze()
        
        output = model(input, mask_squeezed)
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
        test_loss += loss.item() * input.size(0)
        n_test += input.size(0)

        if i % 100 == 0:
            # Save the first image tensor from the batch
            filename = os.path.join(results_path, f"input_{i}.png")
            vutils.save_image(input[0], filename)
            filename = os.path.join(results_path, f"result_{i}.png")
            vutils.save_image(output[0], filename)

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


# Create a dictionary containing the model's state, optimizer state, and any additional information
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    # Add any other information you want to save
}

# Save the model checkpoint to the file
save_path = "./trained_model.pth"
torch.save(checkpoint, save_path)

# Visualize the losses for the train and test set.
plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.legend()
plt.show()
