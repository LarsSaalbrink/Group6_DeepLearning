# %%
################### Imports ###################
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt

# %%
################### Setup ###################
# Download the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=64, shuffle=False
)
# Dataset contains 28x28 images

class MyNetwork(nn.Module):
    def __init__(self, layer_info_list, input_size, num_epochs, reg_info_list):
        super(MyNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for layer_info in layer_info_list:
            self.layers.extend(layer_info)
        
        
        print(self.layers)

        self.num_epochs = num_epochs
        self.input_size = input_size
        self.reg_info_list = reg_info_list

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
        for layer in self.layers:
            x = layer(x)
        return x
        
def train(model, num_epochs, criterion, optimizer, HW_choice):
    print("CUDA available? ", torch.cuda.is_available())
    if HW_choice == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("HW choice: ", device)
    model.to(device)
    
    correct = 0
    total = 0
    val_loss_values = []
    loss_values = []
    accuracy_values = []
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch} / {num_epochs}")
        loss_value = 0
        val_loss_value = 0
        model.train()
    
        # Training loop & loss
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
    
            loss = criterion(outputs, targets)
            # Applying regularization
            for reg_info in model.reg_info_list:
                idx, type, lambda_val = reg_info
                layer = model.layers[idx]
                loss += lambda_val * torch.norm(layer.weight, p=type)  # L2 regularization
    
            loss.backward()
            optimizer.step()
            loss_value += loss.item()
    
        # Validation loss
        model.eval()
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
    
                # Validation loss
                val_loss = criterion(outputs, targets)
                val_loss_value += val_loss.item()
    
                # Accuracy
                _, predicted = torch.max(outputs.detach(), dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
    
        loss_values.append(loss_value / len(train_loader))
        print(f"Training loss: {loss_value/len(train_loader):.4f}")
    
        val_loss_values.append(val_loss_value / len(test_loader))
        print(f"Validation loss: {val_loss_value/len(train_loader):.4f}")
    
        accuracy = correct / total
        accuracy_values.append(accuracy)
        print(f"Accuracy: {(100*accuracy):.2f}%")

    return loss_values, val_loss_values, accuracy_values

# %%
################### Training ###################

layer_info_list = [
    [nn.Linear(in_features=28*28, out_features=512), nn.ReLU()],
    [nn.Dropout(p=0.2), nn.Identity()],
    [nn.Linear(in_features=512, out_features=256), nn.ReLU()],
    [nn.Dropout(p=0.2), nn.Identity()],
    [nn.Linear(in_features=256, out_features=128), nn.ReLU()],
    [nn.Linear(in_features=128, out_features=10), nn.Identity()]
]
input_size = 28*28
num_epochs = 10

#Layers numbers are even, uneven numbers are activation functions
# [layer_index, regularization_type, lambda_val]
reg_info_list = [[0, 2, 0.001],[4, 2, 0.001],[8, 2, 0.001],[10, 2, 0.001]]  

model = MyNetwork(layer_info_list, input_size, num_epochs, reg_info_list)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.4)
criterion = nn.CrossEntropyLoss() # This invokes the softmax function at the end of the network

loss_values, val_loss_values, accuracy_values = train(model, num_epochs, criterion, optimizer, "cpu")

# %%
################### Plotting ###################
fig, ax1 = plt.subplots()

# Set labels and ticks for the x-axis
ax1.set_xlabel("Epochs")
ax1.set_xticks(range(0, num_epochs))
ax1.set_ylabel("Loss")

# Plot training and validation loss on the first y-axis (ax1)
ax1.plot(loss_values, label="Training loss", color="blue")
ax1.plot(val_loss_values, label="Validation loss", color="orange")

# Create the second y-axis (ax2) for accuracy
ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy")
ax2.plot(accuracy_values, label="Accuracy", color="green")

# Combine the legends from both y-axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="upper center")

plt.show()

# %%
