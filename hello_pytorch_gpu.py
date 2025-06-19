import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create synthetic data and move to GPU
X = torch.linspace(-10, 10, 100).reshape(-1, 1).to(device)
y = (2 * X + 1 + torch.randn(X.shape, device=device) * 0.5).to(device)

# Define the neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input feature, one output feature
    
    def forward(self, x):
        return self.linear(x)

# Initialize the model, loss function, and optimizer
model = SimpleNet().to(device)  # Move model to GPU
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
print("Training the model on GPU...")
epochs = 100
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Plot the results
print("\nPlotting results...")
model.eval()
with torch.no_grad():
    y_pred = model(X)
    # Move data back to CPU for plotting
    X_cpu = X.cpu().numpy()
    y_cpu = y.cpu().numpy()
    y_pred_cpu = y_pred.cpu().numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X_cpu, y_cpu, label='Original Data')
plt.plot(X_cpu, y_pred_cpu, 'r-', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('PyTorch Hello World: Linear Regression (GPU Accelerated)')
plt.legend()
plt.grid(True)
plt.savefig('pytorch_hello_world_gpu.png')
plt.close()

print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.item():.4f}")

print(f"\nTraining completed on {device}! Check 'pytorch_hello_world_gpu.png' for the visualization.") 