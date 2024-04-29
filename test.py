import torch
import torch.nn as nn
from torch_pconv import PConv2d
from model import UNET
from basicUnet import basicUnet


print(torch.backends.cudnn.version())

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = PConv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = PConv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x, mask = self.conv1(x, x.squeeze(0))
#         x = self.relu(x)
#         x, mask = self.conv2(x, mask)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.relu(x)
#         return x
    
# Create an instance of the CNN

# model = CNN()


model = basicUnet()

input_data = torch.randn(1, 1, 64,  64, requires_grad=True)  # Batch size 1, 1 channel, 28x28 image
mask = torch.randn(1, 64, 64, requires_grad=True)  # Batch size 1, 1 channel, 28x28 image
target = torch.ones(1, 1, 64, 64, requires_grad=True)

# Define the loss function
criterion = nn.MSELoss()

outputs = model(input_data)

print(outputs.shape)

# Compute the loss
loss = criterion(outputs, target)

print(f"Starting backprop")

# Backward pass
loss.backward()

print(f"Backprop Successful")

# Print loss
print(f"Epoch 1, Loss: {loss.item()}")








# if torch.cuda.is_available():
#     print("CUDA is available!")
# else:
#     print("CUDA is not available. Using CPU instead.")
    
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# tensor = torch.tensor([1,2]).to('cuda')
# print(tensor.device)
# print(tensor.shape)