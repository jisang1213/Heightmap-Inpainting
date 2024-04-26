import torch

# Define input data and target
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # Input data
y_true = torch.tensor([[0.0], [1.0]])  # Target

# Define a simple linear model
model = torch.nn.Linear(2, 1)

# Define a loss function
criterion = torch.nn.MSELoss()

# Define an optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Forward pass
y_pred = model(x)

# Compute the loss
loss = criterion(y_pred, y_true)

# Perform backpropagation
loss.backward()

# Update model parameters
optimizer.step()

# Print updated parameters
print("Updated parameters:")
for param in model.parameters():
    print(param.data)

# Print updated gradients
print("\nUpdated gradients:")
for param in model.parameters():
    print(param.grad)

# if torch.cuda.is_available():
#     print("CUDA is available!")
# else:
#     print("CUDA is not available. Using CPU instead.")
    
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# tensor = torch.tensor([1,2]).to('cuda')
# print(tensor.device)
# print(tensor.shape)