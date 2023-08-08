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

#1.  Create a neural network:
# Initialize 3 layers
input_size = 784
hidden_sizes = [128, 64]
output_size = 10
epochs = 10

#Define the forward function:
# Reshape the data to a fully connected layer. Hint: Use .view(). 
# Let the input pass through the different layers.
# Consider what activation function you want to use in between the 
# layers, and for the final layer.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = x.view(-1, input_size)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.logsoftmax(self.fc3(x))
        return x

# Loss function and optimizer: 
# Consider what loss function and optimizer you want to use.
model = Net()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Create the training loop:
# Loop over the training data and pass it through the network.
# Consider how many epochs you want to train for.
# Print the training loss at each epoch.
running_loss_arr = []
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(train_loader)}")
        running_loss_arr.append((running_loss/len(train_loader)))

#Create the evaluation loop:
# Loop over the test data and pass it through the network.
# Print the test loss and the accuracy.
# Consider how you want to calculate the accuracy.
test_loss = 0
accuracy = 0
with torch.no_grad():
    for images, labels in test_loader:
        log_ps = model(images)
        test_loss += criterion(log_ps, labels)
        
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class==labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
    else:
        print(f"Test loss: {test_loss/len(test_loader)}")
        print(f"Accuracy: {accuracy/len(test_loader)}")

# Save the model
torch.save(model.state_dict(), 'checkpoint.pth')

# Report the accuracy of your model on the test data.
# Plot the loss curve
# epoch is integer
plt.figure(figsize=(12,6))
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(range(1,epochs))
plt.plot(running_loss_arr, label="Training Loss")
# plt.plot(test_loss/len(test_loader), label="Test Loss")
plt.show()



