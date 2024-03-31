from rbf_layer.rbf_utils import rbf_inverse_multiquadric
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
import torchinfo

mpl.rcParams['agg.path.chunksize'] = 10000
# Define device
torch.set_default_device('cuda')

# Load the dataset
data = pd.read_csv("RBF/data/data.csv")

# Prepare data
y_object1 = data[['angle1', 'pos1x', 'pos1y']].values
y_object2 = data[['angle2', 'pos2x', 'pos2y']].values

# Define RBF
def euclidean_norm(x):
    return torch.norm(x, p=2, dim=-1)

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / (x.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)

        x = torch.matmul(attention_weights, value)
        return x

class RBFModel(nn.Module):
    def __init__(self):
        super(RBFModel, self).__init__()
        
        # Branch for learning relationship between column 1 and column 2
        self.branch1 = nn.Sequential(
            RBFLayer(in_features_dim=2,  # Pair of columns (column 1, column 2)
                    num_kernels=32,
                    out_features_dim=16,
                    radial_function=rbf_inverse_multiquadric,
                    norm_function=euclidean_norm,
                    normalization=True),
            nn.Dropout(0.3),
            AttentionLayer(input_dim=16, output_dim=16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Tanh()
        )
        
        # Branch for learning relationship between column 1 and column 3
        self.branch2 = nn.Sequential(
            RBFLayer(in_features_dim=2,  # Pair of columns (column 1, column 3)
                    num_kernels=32,
                    out_features_dim=16,
                    radial_function=rbf_inverse_multiquadric,
                    norm_function=euclidean_norm,
                    normalization=True),
            nn.Dropout(0.3),
            AttentionLayer(input_dim=16, output_dim=16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Tanh()
        )
        
        # Branch for learning relationship between column 2 and column 3
        self.branch3 = nn.Sequential(
            RBFLayer(in_features_dim=2,  # Pair of columns (column 2, column 3)
                    num_kernels=32,
                    out_features_dim=16,
                    radial_function=rbf_inverse_multiquadric,
                    norm_function=euclidean_norm,
                    normalization=True),
            nn.Dropout(0.3),
            AttentionLayer(input_dim=16, output_dim=16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Tanh()
        )
        
        # Merge layer
        self.merge = nn.Sequential(
            nn.Linear(64 * 3, 32),  # Combine outputs of all branches
            nn.ReLU()
        )
        
        # Output layer
        self.output_layer = nn.Linear(32, 3)

    def forward(self, x):
        # Split input into individual columns
        col1, col2, col3 = torch.split(x, 1, dim=1)
        
        # Process each pair of columns separately
        out1 = self.branch1(torch.cat((col1, col2), dim=1))
        out2 = self.branch2(torch.cat((col1, col3), dim=1))
        out3 = self.branch3(torch.cat((col2, col3), dim=1))
        
        # Concatenate the outputs of all branches
        merged = torch.cat((out1, out2, out3), dim=1)
        
        # Merge the outputs
        merged = self.merge(merged)
        
        # Output layer
        output = self.output_layer(merged)
        
        return output

# Initialize models
model_object1 = RBFModel()
torchinfo.summary(model_object1)
model_object2 = RBFModel()

# Define optimizer
criterion = nn.MSELoss()
optimizer_object1 = torch.optim.Adam(model_object1.parameters(), lr=0.001)
optimizer_object2 = torch.optim.Adam(model_object2.parameters(), lr=0.001)

# Training loop
num_epochs = 2000
batch_size = 512
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
plt.figure(figsize=(20, 20))
plt.plot(trn_losses_object1, label='Object 1 Training Loss')
plt.plot(trn_losses_object2, label='Object 2 Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('training_loss_graph.png')

# Function to plot positions and save as image
def plot_positions(object1_positions, object2_positions, filename, predicted_object1=None, predicted_object2=None, predicted_color='black'):
    plt.figure(figsize=(20, 20))
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
predicted_positions_object1 = predicted_positions_object1[:500]
predicted_positions_object2 = predicted_positions_object2[:500]

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