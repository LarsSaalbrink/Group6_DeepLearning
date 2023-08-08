import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets, transforms 
import seaborn as sns 
import matplotlib.pyplot as plt  
# Download the MNIST dataset
transform = transforms.ToTensor() 
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform) 
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform) 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True) 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

################### Part 1 ###################
input_size = 28*28  #Dataset contains 28x28 images
hidden_sizes = [40, 20]
output_size = 10
num_epochs = 10

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

model = MyNetwork()

# loss_func = nn.CrossEntropyLoss()  #This invokes the softmax function at the end of the network
# optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_func = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

loss_values = []
correct = 0
total = 0
val_loss_values = []
for epoch in range(num_epochs):
    loss_value = 0
    val_loss_value = 0
    model.train() 

    #Training loop & loss
    for data, targets in train_loader:
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
            outputs = model(data)

            #Validation loss
            val_loss = loss_func(outputs, targets)
            val_loss_value += val_loss.item()

            #Accuracy
            _, predicted = torch.max(outputs.detach(), dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    loss_values.append(loss_value/len(train_loader))
    print(f"Training loss: {loss_value/len(train_loader)}")

    val_loss_values.append(val_loss_value/len(test_loader))
    print(f"Validation loss: {val_loss_value/len(train_loader)}")

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")

plt.plot(loss_values, label='Training loss')
plt.plot(val_loss_values, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(1,num_epochs))
plt.legend()
plt.show()
