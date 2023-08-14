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

#Now with:
# -Dropout

# -0.05 learning rate
# -momentum 0.4

# -Addition width and depth

# -Even more width and depth
# -Regularization
# -0.025 learning rate

#Legend: [transformation_layer, activation_function]
layer_info_list = [
    [nn.Linear(in_features=28 * 28, out_features=512), nn.ReLU()],
    [nn.Dropout(p=0.2), nn.Identity()],
    [nn.Linear(in_features=512, out_features=256), nn.ReLU()],
    [nn.Dropout(p=0.2), nn.Identity()],
    [nn.Linear(in_features=256, out_features=128), nn.ReLU()],
    [nn.Dropout(p=0.2), nn.Identity()],
    [nn.Linear(in_features=128, out_features=64), nn.ReLU()],
    [nn.Dropout(p=0.2), nn.Identity()],
    [nn.Linear(in_features=64, out_features=32), nn.ReLU()],
    [nn.Linear(in_features=32, out_features=32), nn.ReLU()],
    [nn.Linear(in_features=32, out_features=10), nn.Identity()],
]
input_size = 28 * 28
num_epochs = 40

# Layers numbers are even, uneven numbers are activation functions
# Legend: [layer_index, regularization_type, lambda_val]
reg_info_list = [[0, 2, 0.001],
                 [4, 2, 0.001],
                 [8, 2, 0.001],
                 [12, 2, 0.001],
                 [16, 2, 0.001]]

model = src.MyNetwork(layer_info_list, input_size, num_epochs, reg_info_list)

optimizer = optim.SGD(model.parameters(), 
                      lr=0.025,
                      momentum=0.4)
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
