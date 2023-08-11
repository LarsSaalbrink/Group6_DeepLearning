# %%
################### Imports ###################
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt

import Ex3_src as src

# %%
################### Training ###################

#Now with dropout

#Legend: [transformation_layer, activation_function]
layer_info_list = [
    [nn.Linear(in_features=28 * 28, out_features=40), nn.ReLU()],
    [nn.Dropout(p=0.2), nn.Identity()],
    [nn.Linear(in_features=40, out_features=20), nn.ReLU()],
    [nn.Dropout(p=0.2), nn.Identity()],
    [nn.Linear(in_features=20, out_features=10), nn.Identity()],
]
input_size = 28 * 28
num_epochs = 40

# Layers numbers are even, uneven numbers are activation functions
# Legend: [layer_index, regularization_type, lambda_val]
reg_info_list = []

model = src.MyNetwork(layer_info_list, input_size, num_epochs, reg_info_list)

optimizer = optim.SGD(model.parameters(), 
                      lr=0.01)
criterion = (
    # This invokes the softmax function at the end of the network
    nn.CrossEntropyLoss()
)  

loss_values, val_loss_values, accuracy_values = src.train(
    model, num_epochs, criterion, optimizer, "cpu"
)

# %%
src.plot(loss_values, val_loss_values, accuracy_values, num_epochs)

# %%
