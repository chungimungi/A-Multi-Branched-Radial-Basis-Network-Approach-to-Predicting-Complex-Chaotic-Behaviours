from rbf_layer.rbf_utils import rbf_inverse_multiquadric, rbf_gaussian
from rbf_layer.rbf_layer import RBFLayer
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import os
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000
# Define device
torch.set_default_device('cuda')

# Load the dataset
data = pd.read_csv("data/data.csv")

# Prepare data
y_object1 = data[['angle1', 'pos1x', 'pos1y']].values
y_object2 = data[['angle2', 'pos2x', 'pos2y']].values

# Define RBF
def euclidean_norm(x):
    return torch.norm(x, p=2, dim=-1)

# Define the neural network model
class RBFModel(nn.Module):
    def __init__(self):
        super(RBFModel, self).__init__()
        self.rbf = RBFLayer(in_features_dim=3,
                            num_kernels=12,
                            out_features_dim=3,
                            radial_function = rbf_gaussian,
                            norm_function=euclidean_norm,
                            normalization=True)
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(3, 3)

    def forward(self, x):
        x = self.rbf(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

# Initialize models
model_object1 = RBFModel()
model_object2 = RBFModel()

# Define optimizer
criterion = nn.MSELoss()
optimizer_object1 = torch.optim.Adam(model_object1.parameters(), lr=0.001)
optimizer_object2 = torch.optim.Adam(model_object2.parameters(), lr=0.001)

# Training loop
num_epochs = 1500
batch_size = 128
trn_losses_object1 = []
trn_losses_object2 = []

for epoch in tqdm(range(num_epochs), desc='Epoch', unit='epoch'):
    for batch in range(0, len(y_object1), batch_size):
        input_object1 = torch.tensor(y_object1[batch:batch+batch_size], dtype=torch.float32)
        input_object2 = torch.tensor(y_object2[batch:batch+batch_size], dtype=torch.float32)

        optimizer_object1.zero_grad()
        optimizer_object2.zero_grad()

        output_object1 = model_object1(input_object1)
        output_object2 = model_object2(input_object2)

        loss_object1 = criterion(output_object1, input_object1)
        loss_object2 = criterion(output_object2, input_object2)

        loss_object1.backward()
        loss_object2.backward()

        optimizer_object1.step()
        optimizer_object2.step()

        trn_losses_object1.append(loss_object1.item())
        trn_losses_object2.append(loss_object2.item())

# Evaluation
predicted_positions_object1 = []
predicted_positions_object2 = []

model_object1.eval()
model_object2.eval()

with torch.no_grad():
    for _ in tqdm(range(1500), desc='Predicting', unit='step'):
        next_position_object1 = model_object1(input_object1)
        predicted_positions_object1.append(next_position_object1)

        next_position_object2 = model_object2(input_object2)
        predicted_positions_object2.append(next_position_object2)

predicted_positions_object1 = torch.cat(predicted_positions_object1, dim=0).cpu().numpy()
predicted_positions_object2 = torch.cat(predicted_positions_object2, dim=0).cpu().numpy()

# Plot and save the loss graph
plt.figure(figsize=(40, 40))
plt.plot(trn_losses_object1, label='Object 1 Training Loss')
plt.plot(trn_losses_object2, label='Object 2 Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('training_loss_graph.png')

# Function to plot positions and save as image
def plot_positions(object1_positions, object2_positions, filename, predicted_object1=None, predicted_object2=None, predicted_color='black'):
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

# Trim predicted positions to match the number of evaluation steps
predicted_positions_object1 = predicted_positions_object1[:100]
predicted_positions_object2 = predicted_positions_object2[:100]

# Plot and save each frame
image_filenames = []
for i in range(len(predicted_positions_object1)):
    plot_positions(y_object1[:, 1:], y_object2[:, 1:], f'frame_{i:03d}.png',
                    predicted_object1=predicted_positions_object1[:i+1, 1:], predicted_object2=predicted_positions_object2[:i+1, 1:],
                    predicted_color='black')
    image_filenames.append(f'frame_{i:03d}.png')

# Compile frames into GIF
with imageio.get_writer('object_movement.gif', mode='I') as writer:
    for filename in image_filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Clean up image files
for filename in image_filenames:
    os.remove(filename)
