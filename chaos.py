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
num_epochs = 100
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

import turtle

# Initialize Turtle screen
screen = turtle.Screen()
screen.setup(width=800, height=600)
screen.title('Object Movement')

# Draw the dataset movements
object1_turtle = turtle.Turtle()
object1_turtle.color('blue')
object1_turtle.penup()
object1_turtle.speed(0)

object2_turtle = turtle.Turtle()
object2_turtle.color('red')
object2_turtle.penup()
object2_turtle.speed(0)

# Draw the dataset movements
for point1, point2 in zip(y_object1, y_object2):
    object1_turtle.goto(point1[1], point1[2])
    object2_turtle.goto(point2[1], point2[2])

# Draw the predicted path combined with the dataset movements
combined_object1_turtle = turtle.Turtle()
combined_object1_turtle.color('green')
combined_object1_turtle.penup()
combined_object1_turtle.speed(0)

combined_object2_turtle = turtle.Turtle()
combined_object2_turtle.color('purple')
combined_object2_turtle.penup()
combined_object2_turtle.speed(0)

# Start from the last known point of the dataset
last_dataset_point1 = y_object1[-1][1:]
last_dataset_point2 = y_object2[-1][1:]

# Draw the predicted path starting from the last known dataset point
for predicted_point1, predicted_point2 in zip(predicted_positions_object1, predicted_positions_object2):
    combined_object1_turtle.goto(last_dataset_point1[0], last_dataset_point1[1])
    combined_object1_turtle.pendown()
    combined_object1_turtle.goto(predicted_point1[1], predicted_point1[2])
    combined_object1_turtle.penup()
    last_dataset_point1 = (predicted_point1[1], predicted_point1[2])
    
    combined_object2_turtle.goto(last_dataset_point2[0], last_dataset_point2[1])
    combined_object2_turtle.pendown()
    combined_object2_turtle.goto(predicted_point2[1], predicted_point2[2])
    combined_object2_turtle.penup()
    last_dataset_point2 = (predicted_point2[1], predicted_point2[2])

# Keep the window open
screen.mainloop()

