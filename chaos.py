import torch
from rbf_layer.rbf_layer import RBFLayer
from rbf_layer.rbf_utils import rbf_inverse_multiquadric
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from matplotlib.animation import FuncAnimation

# Define device
torch.set_default_device('cuda')

# Load the dataset
data = pd.read_csv("aarush/RBF/data/data.csv")

# Prepare data
y_object1 = data[['angle1','pos1x', 'pos1y']].values
y_object2 = data[['angle2','pos2x', 'pos2y']].values

# Convert to PyTorch tensors
y_object1_tensor = torch.tensor(y_object1, dtype=torch.float32)
y_object2_tensor = torch.tensor(y_object2, dtype=torch.float32)

# Define RBF
def euclidean_norm(x):
    return torch.norm(x, p=2, dim=-1)

# Define the neural network model
class RBF(nn.Module):
    def __init__(self):
        super(RBF, self).__init__()
        self.rbf = RBFLayer(in_features_dim=3,
                            num_kernels=12,
                            out_features_dim=3,
                            radial_function=rbf_inverse_multiquadric,
                            norm_function=euclidean_norm,
                            normalization=True)
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(3, 3)

    def forward(self, x):
        x = self.rbf(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

# Instantiate the model
model_object1 = RBF()
model_object2 = RBF()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer_object1 = torch.optim.Adam(model_object1.parameters(), lr=0.001)
optimizer_object2 = torch.optim.Adam(model_object2.parameters(), lr=0.001)

# Training loop
num_epochs = 400
batch_size = 128
trn_losses_object1 = []
trn_losses_object2 = []

for epoch in range(num_epochs):
    model_object1.train()
    model_object2.train()

    # Shuffle the indices for each epoch
    indices = torch.randperm(y_object1_tensor.size(0))
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_object1 = y_object1_tensor[batch_indices]
        batch_object2 = y_object2_tensor[batch_indices]

        optimizer_object1.zero_grad()
        optimizer_object2.zero_grad()

        # Forward pass for object 1
        output_object1 = model_object1(batch_object1)
        loss_object1 = criterion(output_object1, batch_object1)
        loss_object1.backward()
        optimizer_object1.step()
        trn_losses_object1.append(loss_object1.item())

        # Forward pass for object 2
        output_object2 = model_object2(batch_object2)
        loss_object2 = criterion(output_object2, batch_object2)
        loss_object2.backward()
        optimizer_object2.step()
        trn_losses_object2.append(loss_object2.item())

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss_Object1: {loss_object1.item():.4f}, Loss_Object2: {loss_object2.item():.4f}')

# Prepare input data
input_object1 = y_object1_tensor[-1].unsqueeze(0)  # Last known position of object1
input_object2 = y_object2_tensor[-1].unsqueeze(0)  # Last known position of object2

# Predict next 100 steps
predicted_positions_object1 = []
predicted_positions_object2 = []

model_object1.eval() 
model_object2.eval() 

with torch.no_grad():
    for _ in tqdm(range(15000), desc='Predicting', unit='step'):
        # Predict next position for object1
        next_position_object1 = model_object1(input_object1)
        predicted_positions_object1.append(next_position_object1)
        
        # Predict next position for object2
        next_position_object2 = model_object2(input_object2)
        predicted_positions_object2.append(next_position_object2)
        
        # Update input data for next iteration
        input_object1 = next_position_object1
        input_object2 = next_position_object2

# Convert predicted positions to numpy arrays
predicted_positions_object1 = torch.cat(predicted_positions_object1, dim=0).cpu().numpy()
predicted_positions_object2 = torch.cat(predicted_positions_object2, dim=0).cpu().numpy()

# Create animation
fig, ax = plt.subplots(figsize=(40, 40))

def update(frame):
    ax.clear()
    ax.plot(y_object1[:frame, 1], y_object1[:frame, 2], label='Object 1 (Data)', color='blue', marker='o')
    ax.plot(y_object2[:frame, 1], y_object2[:frame, 2], label='Object 2 (Data)', color='red', marker='o')
    ax.plot(predicted_positions_object1[:frame, 1], predicted_positions_object1[:frame, 2], label='Object 1 (Predicted)', color='black', linestyle='dotted')
    ax.plot(predicted_positions_object2[:frame, 1], predicted_positions_object2[:frame, 2], label='Object 2 (Predicted)', color='green', linestyle='dotted')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Object Movement')
    ax.legend()
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)

ani = FuncAnimation(fig, update, frames=len(predicted_positions_object1), interval=50)

# Save animation as file
ani.save('object_movement.gif', writer='pillow')
