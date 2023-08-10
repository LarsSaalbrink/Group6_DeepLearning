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
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform) 
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform) 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True) 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

input_size = 28*28  #Dataset contains 28x28 images
hidden_sizes = [40, 20]
output_size = 10
num_epochs = 3

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_sizes[0])
        self.fc2 = nn.Linear(in_features=hidden_sizes[0], out_features=hidden_sizes[1])
        self.fc3 = nn.Linear(in_features=hidden_sizes[1], out_features=output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = x.view(-1, input_size)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.logsoftmax(self.fc3(x))
        return x

# %%
################### Training ###################
model = MyNetwork()

print("CUDA available? ", torch.cuda.is_available())
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("HW choice: ", device)
model.to(device)

# loss_func = nn.CrossEntropyLoss()  #This invokes the softmax function at the end of the network
# optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_func = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

correct = 0
total = 0
val_loss_values = []
loss_values = []
accuracy_values = []
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch}")
    loss_value = 0
    val_loss_value = 0
    model.train() 

    #Training loop & loss
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_value += loss.item()

    #Validation loss
    model.eval()
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            #Validation loss
            val_loss = loss_func(outputs, targets)
            val_loss_value += val_loss.item()

            #Accuracy
            _, predicted = torch.max(outputs.detach(), dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    loss_values.append(loss_value/len(train_loader))

    print(f"Training loss: {loss_value/len(train_loader):.4f}")

    val_loss_values.append(val_loss_value/len(test_loader))
    print(f"Validation loss: {val_loss_value/len(train_loader):.4f}")

    accuracy = correct / total
    print(f"Accuracy: {(100*accuracy):.2f}%")
    accuracy_values.append(accuracy)

# %%
################### Plotting ###################
fig, ax1 = plt.subplots()

# Set labels and ticks for the x-axis
ax1.set_xlabel('Epochs')
ax1.set_xticks(range(0, num_epochs))
ax1.set_ylabel('Loss')

# Plot training and validation loss on the first y-axis (ax1)
ax1.plot(loss_values, label='Training loss', color='blue')
ax1.plot(val_loss_values, label='Validation loss', color='orange')

# Create the second y-axis (ax2) for accuracy
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy')
ax2.plot(accuracy_values, label='Accuracy', color='green')

# Combine the legends from both y-axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper center')

plt.show()

# %%
