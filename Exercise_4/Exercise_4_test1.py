# %%
################### Imports ###################
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt

import Exercise_4_src as src

# %%
### Plot the distribution of the dataset ###
label_counts = [0 for i in range(10)]
for label in src.train_dataset.targets:
    label_counts[label] += 1

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(range(10), label_counts)
plt.xlabel("Labels")
plt.ylabel("Frequency")
plt.title("Label Distribution in MNIST Dataset")
plt.xticks(range(10), [str(i) for i in range(10)])
plt.show()

# %%
################### 2. Train a CNN ###################
### Inputs ###
input_depth = 1  # Dataset contains 28x28x1 images
output_size = 10
num_epochs = 10
HW_choice = "cpu"
lr = 0.003

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3) # Output size:([10, 10, 26, 26])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) # Output size:([10, 10, 13, 13])
        self.fc1 = nn.Linear(10 * 13 * 13, 10) # input_channels = n_channels_conv * height * width
        self.dropout = nn.Dropout(0.2)

        self.softmax = nn.Softmax(dim=1)

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.softmax(x)

        return x

model = MyNetwork()
loss_values, val_loss_values, accuracy_values = src.train_model(
    model, num_epochs, HW_choice
)

# %%
################### Plot ###################
src.plot(loss_values, val_loss_values, accuracy_values, num_epochs, "Test_result")

# %%

