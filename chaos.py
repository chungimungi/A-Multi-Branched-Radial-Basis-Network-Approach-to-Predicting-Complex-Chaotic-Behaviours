import torch
from rbf_layer.rbf_layer import RBFLayer
from rbf_layer.rbf_utils import rbf_inverse_multiquadric
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        self.output_layer = nn.Linear(3, 3)

    def forward(self, x):
        x = self.rbf(x)
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
num_epochs = 500
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

model_object1.eval()  # Set model to evaluation mode
model_object2.eval()  # Set model to evaluation mode

with torch.no_grad():
    for _ in tqdm(range(100), desc='Predicting', unit='step'):
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

import matplotlib.pyplot as plt
import numpy as np
import imageio

# Function to plot positions and save as image
import matplotlib.pyplot as plt
import numpy as np
import imageio

# Function to plot positions and save as image
def plot_positions(object1_positions, object2_positions, filename, predicted_object1=None, predicted_object2=None, predicted_color='green'):
    plt.figure(figsize=(40, 40))
    plt.plot(object1_positions[:, 0], object1_positions[:, 1], label='Object 1 (Data)', color='blue')
    plt.plot(object2_positions[:, 0], object2_positions[:, 1], label='Object 2 (Data)', color='red')
    if predicted_object1 is not None:
        plt.plot(predicted_object1[:, 0], predicted_object1[:, 1], label='Object 1 (Predicted)', color=predicted_color)
    if predicted_object2 is not None:
        plt.plot(predicted_object2[:, 0], predicted_object2[:, 1], label='Object 2 (Predicted)', color=predicted_color)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Object Movement')
    plt.legend()
    plt.savefig(filename)
    plt.close()

# Generate predicted positions (assuming you have them)
# predicted_positions_object1 and predicted_positions_object2 should be numpy arrays of shape (num_steps, 3) with columns [angle, posx, posy]
# You should fill these with your actual predicted positions
predicted_positions_object1 = np.random.rand(100, 3)
predicted_positions_object2 = np.random.rand(100, 3)

# Plot and save each frame
image_filenames = []
for i in range(len(predicted_positions_object1)):
    plot_positions(y_object1[:, 1:], y_object2[:, 1:], f'frame_{i:03d}.png',
                    predicted_object1=predicted_positions_object1[:i+1, 1:], predicted_object2=predicted_positions_object2[:i+1, 1:],
                    predicted_color='green')
    image_filenames.append(f'frame_{i:03d}.png')

# Compile frames into GIF
with imageio.get_writer('object_movement.gif', mode='I') as writer:
    for filename in image_filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Clean up image files
import os
for filename in image_filenames:
    os.remove(filename)



