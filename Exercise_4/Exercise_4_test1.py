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
################### 2. Define CNN ###################
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
        self.fc1 = nn.Linear(10 * 13 * 13, 128) # input_channels = n_channels_conv * height * width
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.2)

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.softmax(self.fc2(x))

        return x

model = MyNetwork()

# %%
################### 2. Train a CNN ###################
loss_values, val_loss_values, accuracy_values = src.train_model(
    model, num_epochs, HW_choice
)

# %%
################### Plotting ###################
src.plot(loss_values, val_loss_values, accuracy_values, num_epochs, "Test_result")


# %%

# 3.a Print first convolutional layer filters
img_batch = next(iter(src.test_loader))[0]
conv1_output = model.conv1(img_batch[1])
layer_visualization = conv1_output.data
for i, feature_map in enumerate(layer_visualization):
    plt.subplot(2, 5, i + 1)
    plt.imshow(feature_map.numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig('feature_maps.png')


# %%

